import taichi as ti
import taichi.math as tm

_DEBUG = False

# Configuration
dt = 16.6666e-3
dim = 400
Viscocity = 0.1
PressureIterationCount = 60
VelocityDiffusionIterationCount = 60


# Taichi Initialisation
if _DEBUG:
    ti.init(arch=ti.gpu, kernel_profiler=True)
else:
    ti.init(arch=ti.gpu)

# Buffer creation
VelocityField = ti.Vector.field(n=2, dtype=float, shape=(dim, dim))
VelocityField_Old = ti.Vector.field(n=2, dtype=float, shape=(dim, dim))
if _DEBUG:
    DebugVelocityField = ti.Vector.field(n=2, dtype=float, shape=(dim, dim))

DivergenceField = ti.field(ti.f32, shape=(dim, dim))
PressureField = ti.field(ti.f32, shape=(dim, dim))
PressureField_Old = ti.field(ti.f32, shape=(dim, dim))

if _DEBUG:
    DebugPressureField = ti.field(ti.f32, shape=(dim,dim))

DyeField = ti.Vector.field(n=3,dtype=ti.f32, shape=(dim,dim))
DyeField_Old = ti.Vector.field(n=3,dtype=ti.f32, shape=(dim,dim))

# Functions and Kernels
@ti.func
def ResetFields():
    VelocityField.fill([0.0,0.0])
    DivergenceField.fill(0)

    SetInitialDyeSetup()

@ti.func
def ClearPressure():  # complex square of a 2D vector
    VelocityField.fill([0.0,0.0])

@ti.func
def SetInitialDyeSetup():
    for i, j in DyeField:  # Parallelized over all pixels
        DyeField[i,j] = tm.vec3( float(i) / dim,  float(j) / dim, 0.)

def ReadInput(gui : ti.GUI, PrevFrameCursorPos : tm.vec2):
    CursorPos = ti.Vector( gui.get_cursor_pos() )
    CursorVelocity = (CursorPos - PrevFrameCursorPos)
    return CursorPos, CursorVelocity

@ti.func
def BilinearInterpolate(Coord, SampledField): 
    BL = SampledField[ tm.clamp( int( Coord.x - 0.5), 0, dim-1), tm.clamp( int( Coord.y - 0.5), 0, dim-1)]
    TL = SampledField[ tm.clamp( int(Coord.x - 0.5), 0, dim-1), tm.clamp( int( Coord.y  + 0.5), 0, dim-1)]
    BR = SampledField[ tm.clamp( int( Coord.x + 0.5), 0, dim-1), tm.clamp( int( Coord.y - 0.5), 0, dim-1)]
    TR = SampledField[ tm.clamp( int( Coord.x + 0.5), 0, dim-1), tm.clamp( int(Coord.y + 0.5), 0, dim-1)]

    HoriBottom = tm.mix(BL, BR, tm.fract( Coord.x  - 0.5 ) )
    HoriTop = tm.mix(TL, TR, tm.fract( Coord.x  - 0.5 ) )
    return tm.mix(HoriBottom, HoriTop, tm.fract( Coord.y - 0.5 ) )

@ti.kernel
def Init():
    ResetFields()

@ti.kernel
def Reset():
    ResetFields()

InputVelocityStampSize = 40
VelocityStampField = ti.field(ti.i16, shape=(InputVelocityStampSize,InputVelocityStampSize))
@ti.kernel
def AddInputVelocity(Pos: tm.vec2, Velocity : tm.vec2):
    for i,j in VelocityStampField:
        Strength = tm.sin( (float(i) / InputVelocityStampSize) * tm.pi) * tm.sin( (float(j) / InputVelocityStampSize) * tm.pi)
        VelocityField[tm.clamp(int(Pos.x * dim) + (i-(InputVelocityStampSize//2)), 0, dim-1), tm.clamp(int(Pos.y * dim) + (j-(InputVelocityStampSize//2)), 0, dim-1)] += Strength * Velocity * 20.
    
@ti.kernel
def AdvectVelocity():
    for i, j in VelocityField:  # Parallelized over all pixels
        CurrCellVel = VelocityField_Old[i,j]
        VelocityField[i,j] = BilinearInterpolate( tm.vec2(float(i) + 0.5 - (dim * CurrCellVel.x * dt), float(j) + 0.5 - (dim * CurrCellVel.y * dt) ), VelocityField_Old)

@ti.kernel
def AdvectDye():
    for i, j in VelocityField:  # Parallelized over all pixels
        CurrCellVel = VelocityField[i,j]
        DyeField[i,j] = BilinearInterpolate( tm.vec2(float(i) + 0.5 - (dim * CurrCellVel.x * dt), float(j) + 0.5 - (dim * CurrCellVel.y * dt)) , DyeField_Old) 

@ti.kernel
def CalculateDivergence():
    for i, j in VelocityField:  # Parallelized over all pixels
        if i == 0 or i == (dim-1) or j == 0 or j == (dim-1):
            DivergenceField[i,j] = 0
        else:     
            DivergenceField[i,j] = ((VelocityField[i+1,j].x - VelocityField[i-1,j].x)  + (VelocityField[i,j+1].y - VelocityField[i,j-1].y) ) / 2. 

@ti.func
def Jacobi(Coord : tm.vec2, Alpha, Beta, FieldX, FieldB): # X & B refer to Ax = b
    return ((FieldX[Coord.x+1,Coord.y] + FieldX[Coord.x-1,Coord.y] + FieldX[Coord.x,Coord.y+1] + FieldX[Coord.x,Coord.y-1]) + ( Alpha * FieldB[Coord.x, Coord.y])) * Beta


@ti.kernel
def ComputePressure():
    Alpha = -1.
    Beta = 0.25

    for i,j in PressureField:
        if not (i == 0 or i == (dim-1) or j == 0 or j == (dim-1)) : #not an edge
            PressureField[i,j] = Jacobi( tm.ivec2(i,j), Alpha, Beta, PressureField_Old, DivergenceField) 

@ti.kernel
def DiffuseVelocity():
    Alpha = 1. / (Viscocity * dt)
    Beta = 1. / (4. + Alpha)

    for i,j in VelocityField_Old:
        if not (i == 0 or i == (dim-1) or j == 0 or j == (dim-1)) : #not an edge
            VelocityField[i,j] = Jacobi( tm.ivec2(i,j), Alpha, Beta, VelocityField_Old, VelocityField_Old) 


# @ti.kernel
# def DiffuseDye():
#     Alpha = 1. / (Viscocity * dt)
#     Beta = 1. / (4 + Alpha)

#     for i,j in DyeField_Old:
#         if not (i == 0 or i == (dim-1) or j == 0 or j == (dim-1)) : #not an edge
#             DyeField[i,j] = Jacobi( tm.ivec2(i,j), Alpha, Beta, DyeField_Old, DyeField_Old) 


@ti.kernel
def RemoveDivergenceFromVelocity():
    # compute gradient
    for i, j in PressureField:
        Grad = tm.vec2(0.0,0.0)
        if i == 0 or i == (dim-1) or j == 0 or j == (dim-1):
            # edge
            Grad = tm.vec2(0,0)
        else:
            Grad = tm.vec2((PressureField[i+1,j] - PressureField[i-1,j]) / 2. , (PressureField[i,j+1] - PressureField[i,j-1]) / 2. )
        
        VelocityField[i,j] -= Grad

@ti.kernel
def EnforceBoundaryConditions_Pressure():
    for i,j in ti.ndrange(4,dim):
        if i == 0: #Left Boundary
            PressureField[0,j] = PressureField[1,j]
        elif i == 1: #Right Boundary
            PressureField[dim-1,j] = PressureField[dim-2,j]
        elif i == 2: #Bottom Boundary
            PressureField[j,0] = PressureField[j,1]
        elif i == 3: #Top Boundary
            PressureField[j,dim-1] = PressureField[j,dim-2]

@ti.kernel
def EnforceBoundaryConditions_Velocity():
     for i,j in ti.ndrange(4,dim):
        if i == 0:  #Left Boundary
            VelocityField[0,j] = -VelocityField[1,j]
        elif i == 1: #Right Boundary
            VelocityField[dim-1,j] = -VelocityField[dim-2,j]
        elif i == 2: #Bottom Boundary
            VelocityField[j,0] = -VelocityField[j,1]
        elif i ==  3: #Top Boundary
            VelocityField[j,dim-1] = -VelocityField[j,dim-2]

@ti.kernel
def EnforceBoundaryConditions_Dye():
     for i,j in ti.ndrange(4,dim):
        if i == 0:  #Left Boundary
            DyeField[0,j] = tm.vec3(0.,0.,0.)
        elif i == 1: #Right Boundary
            DyeField[dim-1,j] = tm.vec3(0.,0.,0.)
        elif i == 2: #Bottom Boundary
            DyeField[j,0] = tm.vec3(0.,0.,0.)
        elif i ==  3: #Top Boundary
            DyeField[j,dim-1] = tm.vec3(0.,0.,0.)


if _DEBUG:
    @ti.kernel
    def GenerateDebugVelocityField():
        for i, j in VelocityField:  # Parallelized over all pixels
            DebugVelocityField[i,j] = ((VelocityField[i,j] ) * 0.5) + 0.5

    @ti.kernel
    def GenerateDebugPressureField():
        for i,j in PressureField:
            DebugPressureField[i,j] = (PressureField[i,j] * 0.5) + 0.5


###############################################################################################
# Main Program
###############################################################################################

PrevFrameCursorPos = tm.vec2(0.0,0.0) 
gui = ti.GUI("Eulerian Fluid Sim", res=(dim, dim))
Init()
DisplayedBuffer = 0

while gui.running:
    ##############################
    ## Input
    ##############################
    OverrideVelocity = False
    PrintScopedProfilerInfo = False
    #Keyboard input (espace to close, space to reset, A/Z/E/R to swap between buffer visualisation)
    if gui.get_event(ti.GUI.RELEASE, ti.GUI.PRESS):
        if gui.event.type == ti.GUI.RELEASE:
            if gui.event.key == ti.GUI.ESCAPE:
                gui.close()
                break
            if gui.event.key == ti.GUI.SPACE:
                Reset()
            if  gui.event.key == 'a':
                DisplayedBuffer = 0
            if  gui.event.key == 'z':
                DisplayedBuffer = 1
            if  gui.event.key == 'e':
                DisplayedBuffer = 2
            if  gui.event.key == 'r':
                DisplayedBuffer = 3
            if  gui.event.key == 'd':
                PrintScopedProfilerInfo = True
        elif gui.event.type == ti.GUI.PRESS:
            if gui.event.key == 'Control_L':
                OverrideVelocity = True


    if _DEBUG and PrintScopedProfilerInfo:
        ti.profiler.clear_kernel_profiler_info() 
        ti.sync()

    ##############################
    ## Fluid Sim
    ##############################
    
    # Add velocity with cursor input
    PrevFrameCursorPos, Velocity = ReadInput(gui, PrevFrameCursorPos)
    if OverrideVelocity:
        Velocity = tm.vec2(0.,.2)

    AddInputVelocity(PrevFrameCursorPos, Velocity)
    EnforceBoundaryConditions_Velocity()

    # Advect Velocity
    VelocityField_Old.copy_from(VelocityField)
    AdvectVelocity()
    EnforceBoundaryConditions_Velocity()

    # Diffuse Velocity
    VelocityField_Old.copy_from(VelocityField)
    for i in range( VelocityDiffusionIterationCount):
        DiffuseVelocity()
        EnforceBoundaryConditions_Velocity()
        lastIteration = (i == VelocityDiffusionIterationCount-1) 
        if not lastIteration:
            VelocityField_Old.copy_from(VelocityField)

    #Calculate divergence from velocity field
    CalculateDivergence()

    # Solve Pressure
    PressureField.fill(0)
    PressureField_Old.fill(0)
    for i in range(PressureIterationCount): 
        ComputePressure()
        EnforceBoundaryConditions_Pressure()
        lastIteration = (i == PressureIterationCount-1) 
         #copy new to old - for next iteration
        if not lastIteration:
            PressureField_Old.copy_from(PressureField) #TODO: Ping pong instead of copy

    #Project Pressure / remove Divergence (gradient of Pressure) from velocity field - incompressible fluids
    RemoveDivergenceFromVelocity()
    EnforceBoundaryConditions_Velocity()

    # Advect Dye
    DyeField_Old.copy_from(DyeField)
    AdvectDye()
    EnforceBoundaryConditions_Dye()


    ##############################
    ## Render
    ##############################
    match DisplayedBuffer:
        case 0:
            gui.set_image(DyeField)
        case 1:
            if _DEBUG:
                GenerateDebugVelocityField()
                gui.set_image(DebugVelocityField)
            else:
                gui.vector_field(VelocityField)
        case 2:
            gui.set_image(DivergenceField)
        case 3:
            if _DEBUG:
                GenerateDebugPressureField()
                gui.set_image(DebugPressureField)
            else:
                gui.set_image(PressureField)
        case _:
            gui.set_image(DyeField)

    gui.show()

    if _DEBUG and PrintScopedProfilerInfo:
        ti.profiler.print_kernel_profiler_info()
