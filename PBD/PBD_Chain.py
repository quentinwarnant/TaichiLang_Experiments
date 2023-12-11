import taichi as ti
import taichi.math as tm
from enum import Enum

_DEBUG = False

DT = 0.33
Substeps = 10
Gravity = -1.18
INF_MASS = -1.
Stiffness = 0.98

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
    #Radius : ti.f16
    Mass : ti.f32

Particles = Particle.field(shape=(PC))

Constraints = ti.Vector.field(n=2, dtype=ti.int32, shape=(ConstraintCount))
RestLengths = ti.field(dtype=ti.f32, shape=(ConstraintCount) )
ComplianceFactors = ti.field(dtype=ti.f16, shape=(ConstraintCount))

@ti.func
def SolveLinearConstraint( IndexA : ti.int32, IndexB : ti.int32, deltaTimeSqr : ti.f16 ):
    
    #TODO: don't calculate this each time
    #inverse mass calc:
    ParticleAMass = Particles[IndexA].Mass
    ParticleBMass = Particles[IndexB].Mass

    # Negative mass stands for Infinite mass
    w1 = 1./ParticleAMass if ParticleAMass > 0 else 0
    w2 = 1./ParticleBMass if ParticleBMass > 0 else 0

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
        Particles[i] = Particle( Pos=tm.vec2(  0.3 + i * 0.06 , 0.7   ), Vel=tm.vec2(0,0), Mass=ParticleMass)
        Particles[i].PrevPos = Particles[i].Pos

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

window = ti.ui.Window(name='Position Based Dynamic - Chain', res = (1080, 720), fps_limit=30, pos = (1050, 350))
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
        UpdateVelocities(substepDT)
    #######

    ExtractParticlePositions()

    canvas.circles(Positions, 0.02)
    canvas.lines(Positions, 0.01,LineIndices)
    window.show()