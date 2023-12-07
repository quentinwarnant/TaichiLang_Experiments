import taichi as ti
import taichi.math as tm

_DEBUG = False

DT = 0.33
Substeps = 10
Gravity = -1.18

# Taichi Initialisation
if _DEBUG:
    ti.init(arch=ti.gpu, kernel_profiler=True)      
else:
    ti.init(arch=ti.gpu)

PC = 5 #Particle Count
ConstraintCount = PC-1

@ti.dataclass
class Particle:
    Pos : tm.vec2
    Vel : tm.vec2
    #Radius : ti.f16
    Mass : ti.f32



#TODO: move all these values into a class if it's more efficient?
Positions = ti.field(dtype=tm.vec2, shape=(PC))
Velocities = ti.field(dtype=tm.vec2, shape=(PC))
#Radii = ti.field(dtype=ti.f16, shape=(PC))
Masses = ti.field(dtype=ti.f16, shape=(PC) )
Particles = Particle.field(shape=(PC))

PrevPositions = ti.field(dtype=tm.vec2, shape=(PC))
Constraints = ti.Vector.field(n=2, dtype=ti.int32, shape=(ConstraintCount))
RestLengths = ti.field(dtype=ti.f32, shape=(ConstraintCount) )

@ti.func
def SolveLinearConstraint( IndexA : ti.int32, IndexB : ti.int32, deltaTime : ti.f16 ):
    
    #TODO: don't calculate this each time
    #inverse mass calc:
    w1 = 1./Masses[IndexA]
    w2 = 1./Masses[IndexB]

    # Current Distance
    Pos1 = Positions[IndexA]
    Pos2 = Positions[IndexB]
    l = tm.length(Pos1 - Pos2) # current length
    l0 = RestLengths[IndexA] #rest length

    #TODO: only compute parts once
    dX1 = w1/(w1+w2) * (l-l0) * tm.normalize(Pos2 - Pos1)
    dX2 = -w2/(w1+w2) * (l-l0) * tm.normalize(Pos2 - Pos1)

    Positions[IndexA] += dX1
    Positions[IndexB] += dX2

    #TODO: integrate deltaTime & stiffness factor


@ti.kernel
def Init():
    Velocities.fill((0,0))
    Masses.fill(ti.f16(1.))
    Masses[0] = ti.f16(300000.)
    #Masses[4] = ti.f16(900000.)
    
    for i in Positions:  # Parallelized over all particles
        Positions[i] = tm.vec2(  0.3 + i * 0.06 , 0.7   )
        #Radii[i] = ti.f16(0.05) + ti.random(dtype=ti.f16) * ti.f16(0.2)

    for i in Constraints:
        Constraints[i] = (i,i+1)
        RestLengths[i] = tm.length( Positions[i] - Positions[i+1])

@ti.kernel
def IntegrateForces(deltaTime : ti.f16):
    for i in Positions: # Parallelized over all particles
        #HACK: cheap temp hack for intermediate testing
        if Masses[i] < 2. :
            Velocities[i] += tm.vec2(0.0, Gravity) * deltaTime
        PrevPositions[i] = Positions[i]
        Positions[i] += Velocities[i] * deltaTime

@ti.kernel
def SolveConstraints(deltaTime : ti.f16):
    ti.loop_config(serialize=True) # Serializes the next for loop
    for i in range(PC-1):
        SolveLinearConstraint(Constraints[i].x, Constraints[i].y, deltaTime )
    

@ti.kernel
def UpdateVelocities(deltaTime : ti.f16):
    for i in Velocities: # Parallelized over all particles
        Velocities[i] = (Positions[i] - PrevPositions[i]) / deltaTime



LineIndices = ti.field( dtype=ti.int32, shape=(ConstraintCount*2))

def GenerateLineIndices():
    for i in range(ConstraintCount):
        LineIndices[i*2] = i
        LineIndices[(i*2)+1] = i+1
        
    

window = ti.ui.Window(name='Position Based Dynamic - Chain', res = (1080, 720), fps_limit=30, pos = (1050, 350))
canvas = window.get_canvas()
Init()

GenerateLineIndices()

while window.running:
    if window.get_event( ti.ui.RELEASE):
        if window.event.key == ti.ui.ESCAPE:
            window.destroy()
            break
        if window.event.key == 'r':
            Init()

    ########
    # PDB
    #######
    substepDT = DT / Substeps
    for n in range(Substeps):
        IntegrateForces(substepDT)
        SolveConstraints(substepDT)
        UpdateVelocities(substepDT)
    #######


    canvas.circles(Positions, 0.02)
    canvas.lines(Positions, 0.01,LineIndices)
    window.show()