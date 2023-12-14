import taichi as ti
import taichi.math as tm
from enum import Enum

_DEBUG = False

WindowRes = (1080, 720)
DT = 0.33
Substeps = 10
Gravity = -1.18
INF_MASS = -1.
Stiffness = 1.
ParticleRadius = WindowRes[0] / 50000

# Taichi Initialisation
if _DEBUG:
    ti.init(arch=ti.gpu, kernel_profiler=True)      
else:
    ti.init(arch=ti.gpu)

PC = 10 #Particle Count
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
def SolveLinearConstraint( IndexA : ti.int32, IndexB : ti.int32, deltaTimeSqr : ti.f16 ):
    
    # Negative mass stands for Infinite mass
    w1 = Particles[IndexA].InvMass
    w2 = Particles[IndexB].InvMass

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
    for i in range(PC):  # Parallelized over all particles
        ParticleMass = INF_MASS if (i == 0 or i == 6) else 1
        Particles[i] = Particle( Pos=tm.vec2(  0.3 + i * 0.06 , 0.7   ), Vel=tm.vec2(0,0), Radius=ParticleRadius, Mass=ParticleMass)
        Particles[i].Init()

    for i in Constraints:
        Constraints[i] = (i,i+1)
        RestLengths[i] = tm.length( Particles[i].Pos - Particles[i+1].Pos)
        ComplianceFactors[i] = ti.f16(1. - Stiffness) #Compliance is the inverse of (stiffness)

@ti.kernel
def IntegrateForces(deltaTime : ti.f16):
    for i in range(PC): # Parallelized over all particles
        InvMass = (1./Particles[i].Mass)  if Particles[i].Mass != INF_MASS else 0.
        Particles[i].Vel += InvMass * tm.vec2(0.0, Gravity) * deltaTime
        Particles[i].PrevPos = Particles[i].Pos
        Particles[i].Pos += Particles[i].Vel * deltaTime

@ti.kernel
def SolveConstraints( deltaTimeSqr : ti.f16):
    ti.loop_config(serialize=True) # Serializes the next for loop
    for i in range(PC-1):
        SolveLinearConstraint(Constraints[i].x, Constraints[i].y, deltaTimeSqr )
    
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

LineIndices = ti.field( dtype=ti.int32, shape=(ConstraintCount*2))
def GenerateLineIndices():
    for i in range(ConstraintCount):
        LineIndices[i*2] = i
        LineIndices[(i*2)+1] = i+1
        

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

window = ti.ui.Window(name='Position Based Dynamic - Chain', res = WindowRes, fps_limit=30, pos = (1050, 350))
canvas = window.get_canvas()

class SelectionMode(Enum):
    Unselected = 1
    Selecting = 2
    Dragging = 3

CurrentSelectionMode : SelectionMode = SelectionMode.Unselected

Init()
GenerateLineIndices()

while window.running:
    if window.get_event( ti.ui.RELEASE):
        if window.event.key == ti.ui.ESCAPE:
            window.destroy()
            break
        if window.event.key == 'r':
            Init()
        if window.event.key == ti.ui.LMB:
            CurrentSelectionMode = SelectionMode.Unselected

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
        IntegrateForces(substepDT)
        SolveConstraints(substepDTSqr)
        SolveIntersections(substepDT)
        UpdateVelocities(substepDT)
    #######

    ExtractParticlePositions()

    canvas.circles(Positions, ParticleRadius) #TODO: report to taichi, circles radius is not in pixels unit
    canvas.lines(Positions, 0.01,LineIndices)
    window.show()