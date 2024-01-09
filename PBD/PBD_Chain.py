import taichi as ti
import taichi.math as tm
from enum import Enum

_DEBUG = False
_INVALID_CONSTRAINT = -1

WindowRes = (1080, 720)
DT = 0.33
Substeps = 10
Gravity = -1.18
INF_MASS = -1.
Stiffness = 1.
ParticleRadius = WindowRes[0] / 50000
RopeCount = 1

# Taichi Initialisation
if _DEBUG:
    ti.init(arch=ti.gpu, kernel_profiler=True)      
else:
    ti.init(arch=ti.gpu)

PC = 25 #Particle Count
ConstraintCount = PC-1

@ti.dataclass
class Particle:
    Pos : tm.vec2
    PrevPos : tm.vec2
    Vel : tm.vec2
    Radius : ti.f16
    Mass : ti.f32
    InvMass : ti.f32

    @ti.func
    def Init(self):
        self.InvMass = 1.0 / self.Mass if self.Mass > 0 else 0
        self.PrevPos = self.Pos
        

Particles = Particle.field(shape=(PC))

Constraints = ti.Vector.field(n=2, dtype=ti.int32, shape=(ConstraintCount))
RestLengths = ti.field(dtype=ti.f32, shape=(ConstraintCount) )
ComplianceFactors = ti.field(dtype=ti.f16, shape=(ConstraintCount))

@ti.func
def SolveLinearConstraint( IndexA : ti.int32, IndexB : ti.int32, deltaTimeSqr : ti.f16, SelectedPointIdx : ti.i32 ):
    
    # Negative mass stands for Infinite mass
    w1 = Particles[IndexA].InvMass
    w2 = Particles[IndexB].InvMass

    if SelectedPointIdx == IndexA:
        w1 = 0
    elif SelectedPointIdx == IndexB:
        w2 = 0


    # Current Distance
    PosA = Particles[IndexA].Pos
    PosB = Particles[IndexB].Pos
    AToB = PosB-PosA
    DirAToB = tm.normalize(AToB)
    l = tm.length(AToB) # current length
    l0 = RestLengths[IndexA] #rest length
    lengthDiff = (l-l0)
    weightSum = w1+w2 + (ComplianceFactors[IndexA]/deltaTimeSqr)

    dX1 = w1/(weightSum) * lengthDiff * DirAToB
    dX2 = -w2/(weightSum) * lengthDiff * DirAToB

    Particles[IndexA].Pos += dX1
    Particles[IndexB].Pos += dX2

@ti.kernel
def Init():
    RopeCount = 1
    for i in range(PC):  # Parallelized over all particles
        ParticleMass = INF_MASS if (i == 0 or i == 6 or i == 14) else 1
        Particles[i] = Particle( Pos=tm.vec2(  0.1 + i * 0.052 , 0.7   ), Vel=tm.vec2(0,0), Radius=ParticleRadius, Mass=ParticleMass)
        Particles[i].Init()

    for i in Constraints:
        Constraints[i] = (i,i+1)
        RestLengths[i] = tm.length( Particles[i].Pos - Particles[i+1].Pos)
        ComplianceFactors[i] = ti.f16(1. - Stiffness) #Compliance is the inverse of (stiffness)

@ti.kernel
def IntegrateForces(deltaTime : ti.f16, SelectedPointIdx : ti.i32):
    for i in range(PC): # Parallelized over all particles
        if i != SelectedPointIdx: #don't integrate forces for point being dragged
            Particles[i].Vel += Particles[i].InvMass * tm.vec2(0.0, Gravity) * deltaTime
            Particles[i].PrevPos = Particles[i].Pos
            Particles[i].Pos += Particles[i].Vel * deltaTime

@ti.kernel
def SolveConstraints( deltaTimeSqr : ti.f16, SelectedPointIdx : ti.i32):
    ti.loop_config(serialize=True) # Serializes the next for loop
    for i in range(PC-1):
        if Constraints[i].y != _INVALID_CONSTRAINT:
            SolveLinearConstraint(Constraints[i].x, Constraints[i].y, deltaTimeSqr, SelectedPointIdx )
    
@ti.kernel
def SolveIntersections(deltaTime : ti.f16):
    
    TwoRadius = ParticleRadius * 2
    #TODO: potential to iterate through spatial partitioned cells instead of serially here
    ti.loop_config(serialize=True) # Serializes the next loop
    for i in range(PC-1):
        for j in range(i+1,PC):
            #A To B vector
            Dir = (Particles[j].Pos - Particles[i].Pos)
            LengthBetweenPositions = tm.length(Dir)
            if  LengthBetweenPositions < TwoRadius :
                #intersection
                DirAToB = tm.normalize(Dir)
                IntersectionAmount =  abs(TwoRadius - LengthBetweenPositions)
                w1 = Particles[i].InvMass
                w2 = Particles[j].InvMass

                weightSum = w1+w2
                dX1 = -w1/(weightSum) * IntersectionAmount * DirAToB
                dX2 = w2/(weightSum) * IntersectionAmount * DirAToB

                Particles[i].Pos += dX1
                Particles[j].Pos += dX2

@ti.kernel
def UpdateVelocities(deltaTime : ti.f16):
    for i in range(PC): # Parallelized over all particles
        Particles[i].Vel = (Particles[i].Pos - Particles[i].PrevPos) / deltaTime

#LineIndices = ti.field( dtype=ti.int32, shape=(1,ConstraintCount*2))
def GetRopePositionsAndIndices(ropeFirstNodeIdx):
    positions = ti.field( dtype=tm.vec2, shape=(PC))
    lineIndices = ti.field( dtype=ti.int32, shape=(ConstraintCount*2))

    currentConstraintIdx = ropeFirstNodeIdx
    idx = 0
    # Every rope needs at least two nodes. we add the first node here...
    positions[idx] = Particles[Constraints[currentConstraintIdx].x].Pos
    lineIndices[idx] = idx
    lineIndices[(idx*2)+1] = idx+1
    
    # the every subsequent node until we reach an invalid constraint idx and stop
    while Constraints[currentConstraintIdx].y != _INVALID_CONSTRAINT and currentConstraintIdx < ConstraintCount:
        idx+=1
        positions[idx] = Particles[Constraints[currentConstraintIdx].y].Pos
        currentConstraintIdx+=1
        if Constraints[currentConstraintIdx].y != _INVALID_CONSTRAINT:
            lineIndices[(idx*2)] = Constraints[currentConstraintIdx].x
            lineIndices[(idx*2)+1] = Constraints[currentConstraintIdx].y

    ropeLastNodeIdx = currentConstraintIdx
    return positions, lineIndices, ropeLastNodeIdx

def GenerateLineIndices():
    LineIndices = ti.field( dtype=ti.int32, shape=(RopeCount, ConstraintCount*2)) #TODO: this y dimension is now too large; worth trimming? 
    ropeIdx = 0
    currentRopeConstraintIdx = 0
    for i in range(ConstraintCount):
        if Constraints[i].x == _INVALID_CONSTRAINT:
            ropeIdx+=1
            currentRopeConstraintIdx = 0
        LineIndices[ropeIdx, currentRopeConstraintIdx*2] = currentRopeConstraintIdx
        LineIndices[ropeIdx, (currentRopeConstraintIdx*2)+1] = currentRopeConstraintIdx+1
        currentRopeConstraintIdx+=1

Positions = ti.field(dtype=tm.vec2, shape=(PC))
@ti.kernel
def ExtractParticlePositions():
    for i in range(PC): # Parallelized over all particles
        Positions[i] = Particles[i].Pos
    
SelectedPointIdx = -1
PointSelectionRangeSqr = 1
@ti.kernel
def FindNearbyPoint(cursorPos : tm.vec2) -> ti.i16:
    idxClosest = 0
    closestDistSqr = tm.dot( (Particles[0].Pos - cursorPos), (Particles[0].Pos - cursorPos))

    ti.loop_config(serialize=True) # Serializes the next for loop
    for i in range(1,PC):
        tmpVec = (Particles[i].Pos - cursorPos)
        tmpDistSqr = tm.dot( tmpVec, tmpVec) 
        if tmpDistSqr < closestDistSqr:
            idxClosest = i
            closestDistSqr = tmpDistSqr
    
    return idxClosest if closestDistSqr < PointSelectionRangeSqr else -1

def MoveSelectedPoint(cursorPos):
    Particles[SelectedPointIdx].Pos = cursorPos
    Particles[SelectedPointIdx].PrevPos = cursorPos

def BreakConstraint(ConstraintIdx, RopeCount):
    if Constraints[ConstraintIdx].y != _INVALID_CONSTRAINT:
        RopeCount+=1
        Constraints[ConstraintIdx].y = _INVALID_CONSTRAINT
    return RopeCount

window = ti.ui.Window(name='Position Based Dynamic - Chain', res = WindowRes, fps_limit=30, pos = (1050, 350))
canvas = window.get_canvas()

class SelectionMode(Enum):
    Unselected = 1
    Selecting = 2
    Dragging = 3

CurrentSelectionMode : SelectionMode = SelectionMode.Unselected

Init()

while window.running:
    if window.get_event( ti.ui.RELEASE):
        if window.event.key == ti.ui.ESCAPE:
            window.destroy()
            break
        if window.event.key == 'r':
            Init()
        if window.event.key == ti.ui.LMB:
            CurrentSelectionMode = SelectionMode.Unselected
            SelectedPointIdx = -1
        if window.event.key == 'c':
            SelectedPointIdx = FindNearbyPoint( ti.Vector(window.get_cursor_pos()))
            RopeCount = BreakConstraint(SelectedPointIdx, RopeCount)

    if window.get_event( ti.ui.PRESS ):
        if window.event.key == ti.ui.LMB:
            CurrentSelectionMode = SelectionMode.Selecting
            SelectedPointIdx = FindNearbyPoint( ti.Vector(window.get_cursor_pos()))
            if SelectedPointIdx != -1:
                CurrentSelectionMode = SelectionMode.Dragging
                

    if CurrentSelectionMode == SelectionMode.Dragging:
        MoveSelectedPoint( ti.Vector(window.get_cursor_pos()))

    ########
    # PDB
    #######
    substepDT = DT / Substeps
    substepDTSqr = substepDT * substepDT
    for n in range(Substeps):
        IntegrateForces(substepDT, SelectedPointIdx)
        SolveConstraints(substepDTSqr, SelectedPointIdx)
        SolveIntersections(substepDT)
        UpdateVelocities(substepDT)
    #######

    ropeFirstNodeIdx = 0
    for i in range(RopeCount):
        ropePositions, ropeIndices, lastRopeIdx = GetRopePositionsAndIndices(ropeFirstNodeIdx) -- This is too expensive to do every frame , only compute indices when a break happens. and still extract positions every frame
        canvas.circles(ropePositions, ParticleRadius) #TODO: report to taichi, circles radius is not in pixels unit
        canvas.lines(ropePositions, 0.01, ropeIndices)
        ropeFirstNodeIdx = lastRopeIdx+1
    window.show()