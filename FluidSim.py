import taichi as ti
import taichi.math as tm

import sys
#sys.tracebacklimit=0 #shorter callstacks

ti.init(arch=ti.gpu)

dt = 16.6666e-3
dim = 420
Viscocity = 500.4

VelocityField = ti.Vector.field(n=2, dtype=float, shape=(dim, dim))
VelocityField_Old = ti.Vector.field(n=2, dtype=float, shape=(dim, dim))
DebugVelocityField = ti.Vector.field(n=2, dtype=float, shape=(dim, dim))

DivergenceField = ti.field(ti.f32, shape=(dim, dim))
PressureField = ti.field(ti.f32, shape=(dim, dim))
PressureField_Old = ti.field(ti.f32, shape=(dim, dim))
DebugPressureField = ti.field(ti.f32, shape=(dim,dim))

DieField = ti.Vector.field(n=3,dtype=ti.f32, shape=(dim,dim))
DieField_Old = ti.Vector.field(n=3,dtype=ti.f32, shape=(dim,dim))
@ti.func
def ResetFields():
    VelocityField.fill([0.0,0.0])
    DivergenceField.fill(0)

    SetInitialDieSetup()

@ti.func
def ClearPressure():  # complex square of a 2D vector
    VelocityField.fill([0.0,0.0])

@ti.func
def SetInitialDieSetup():
    for i, j in DieField:  # Parallelized over all pixels
        DieField[i,j] = tm.vec3( float(i) / dim, float(j) / dim, 0.5)

def ReadInput(gui : ti.GUI, PrevFrameCursorPos : tm.vec2):
    CursorPos = ti.Vector( gui.get_cursor_pos() )
    CursorVelocity = (CursorPos - PrevFrameCursorPos)
    return CursorPos, CursorVelocity

@ti.func
def BilinearInterpolate(Coord, SampledField): 
    BL = SampledField[ tm.clamp( int(tm.floor( Coord.x)), 0, dim-1), tm.clamp( int(tm.floor( Coord.y)), 0, dim-1)]
    TL = SampledField[ tm.clamp( int(tm.floor( Coord.x)), 0, dim-1), tm.clamp( int(tm.ceil( Coord.y)), 0, dim-1)]
    BR = SampledField[ tm.clamp( int(tm.ceil( Coord.x)), 0, dim-1), tm.clamp( int(tm.floor( Coord.y)), 0, dim-1)]
    TR = SampledField[ tm.clamp( int(tm.ceil( Coord.x)), 0, dim-1), tm.clamp( int(tm.ceil( Coord.y)), 0, dim-1)]

    HoriBottom = tm.mix(BL, BR, tm.fract(Coord.x))
    HoriTop = tm.mix(TL, TR, tm.fract(Coord.x))
    return tm.mix(HoriBottom, HoriTop, tm.fract(Coord.y))

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
        VelocityField[tm.clamp(int(Pos.x * dim) + (i-(InputVelocityStampSize//2)), 0, dim-1), tm.clamp(int(Pos.y * dim) + (j-(InputVelocityStampSize//2)), 0, dim-1)] += Strength * Velocity * 200.
    
@ti.kernel
def AdvectVelocity():
    for i, j in VelocityField:  # Parallelized over all pixels
        CurrCellVel = VelocityField_Old[i,j]
        VelocityField[i,j] = BilinearInterpolate( tm.vec2(float(i) - (dim * CurrCellVel.x * dt), float(j) - (dim * CurrCellVel.y * dt) ), VelocityField_Old)

@ti.kernel
def AdvectDie():
    for i, j in VelocityField:  # Parallelized over all pixels
        CurrCellVel = VelocityField[i,j]
        DieField[i,j] = BilinearInterpolate( tm.vec2(float(i) - (dim * CurrCellVel.x * dt), float(j) - (dim * CurrCellVel.y * dt)) , DieField_Old) 

@ti.kernel
def CalculateDivergence():
    for i, j in VelocityField:  # Parallelized over all pixels
        if i == 0 or i == (dim-1) or j == 0 or j == (dim-1):
            DivergenceField[i,j] = 0
        else:     
            DivergenceField[i,j] = ((VelocityField[i+1,j].x - VelocityField[i-1,j].x)  + (VelocityField[i,j+1].y - VelocityField[i,j-1].y) / 2.) 

@ti.func
def Jacobi(Coord : tm.vec2, Alpha, Beta, FieldX, FieldB): # X & B refer to Ax = b
    return ((FieldX[Coord.x+1,Coord.y] + FieldX[Coord.x-1,Coord.y] + FieldX[Coord.x,Coord.y+1] + FieldX[Coord.x,Coord.y-1]) + ( Alpha * FieldB[Coord.x, Coord.y])) * Beta


@ti.kernel
def ComputePressure():
    Alpha = -1
    Beta = 0.25

    for i,j in PressureField:
        if not (i == 0 or i == (dim-1) or j == 0 or j == (dim-1)) : #not an edge
            PressureField[i,j] = Jacobi( tm.ivec2(i,j), Alpha, Beta, PressureField_Old, DivergenceField) 

@ti.kernel
def DiffuseVelocity():
    Alpha = 1. / Viscocity * dt
    Beta = 1. / (4 + Alpha)

    for i,j in VelocityField_Old:
        if not (i == 0 or i == (dim-1) or j == 0 or j == (dim-1)) : #not an edge
            VelocityField[i,j] = Jacobi( tm.ivec2(i,j), Alpha, Beta, VelocityField_Old, VelocityField_Old) 


# @ti.kernel
# def DiffuseDie():
#     Alpha = 1. / Viscocity * dt
#     Beta = 1. / (4 + Alpha)

#     for i,j in DieField_Old:
#         if not (i == 0 or i == (dim-1) or j == 0 or j == (dim-1)) : #not an edge
#             DieField[i,j] = Jacobi( tm.ivec2(i,j), Alpha, Beta, VelocityField_Old, VelocityField_Old) 


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
    for i,j in PressureField:
        if i == 0:
            PressureField[i,j] = PressureField[i+1,j]
        elif i == (dim-1):
            PressureField[i,j] = PressureField[i-1,j]
        elif j == 0:
            PressureField[i,j] = PressureField[i,j+1]
        elif j == (dim-1):
            PressureField[i,j] = PressureField[i,j-1]

@ti.kernel
def EnforceBoundaryConditions_Velocity():
    for i,j in VelocityField:
        if i == 0:
            VelocityField[i,j] = -VelocityField[i+1,j]
        elif i == (dim-1):
            VelocityField[i,j] = -VelocityField[i-1,j]
        elif j == 0:
            VelocityField[i,j] = -VelocityField[i,j+1]
        elif j == (dim-1):
            VelocityField[i,j] = -VelocityField[i,j-1]

@ti.kernel
def EnforceBoundaryConditions_Die():
    for i,j in VelocityField:
        if (i == 0 or i == (dim-1) or j == 0 or j == (dim-1)) : #if edge
            DieField[i,j] = tm.vec3(0.,0.,0.)

@ti.kernel
def GenerateDebugVelocityField():
    for i, j in VelocityField:  # Parallelized over all pixels
        DebugVelocityField[i,j] = ((VelocityField[i,j] ) * 0.5) + 0.5


@ti.kernel
def GenerateDebugPressureField():
    for i,j in PressureField:
        DebugPressureField[i,j] = (PressureField[i,j] * 0.5) + 0.5

@ti.kernel
def StampDebugVelocity():
    for i in range(100):
        for j in range(20):
            VelocityField[200+i,dim-1 - j] = tm.vec2(0.,1.) * 2000. 

###################
###################
###################
###################
PrevFrameCursorPos = tm.vec2(0.0,0.0) 
gui = ti.GUI("Julia Set", res=(dim, dim))
Init()
PressureIterationCount = 60
VelocityDiffusionIterationCount = 60

DisplayedBuffer = 0

while gui.running:

    ##############################
    ## Input
    ##############################
    OverrideVelocity = False
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
        elif gui.event.type == ti.GUI.PRESS:
            if gui.event.key == 'Control_L':
                OverrideVelocity = True
            elif gui.event.key == 'w':
                StampDebugVelocity()


    ##############################
    ## Fluid Sim
    ##############################
    PrevFrameCursorPos, Velocity = ReadInput(gui, PrevFrameCursorPos)
    if OverrideVelocity:
        Velocity = tm.vec2(0.,1.)
    AddInputVelocity(PrevFrameCursorPos, Velocity)
    EnforceBoundaryConditions_Velocity()
    

    VelocityField_Old.copy_from(VelocityField)
    AdvectVelocity()
    EnforceBoundaryConditions_Velocity()

    # print('Velocity Edge:', VelocityField[210,dim-1], 'Velocity neighbour', VelocityField[210,dim-2], )

    VelocityField_Old.copy_from(VelocityField)
    for i in range( VelocityDiffusionIterationCount):
        DiffuseVelocity()
        EnforceBoundaryConditions_Velocity()
        lastIteration = (i == VelocityDiffusionIterationCount-1) 
        if not lastIteration:
            VelocityField_Old.copy_from(VelocityField)
    
    #AddExternalForces()
    #EnforceBoundaryConditions_Velocity()

    #Calculate divergence from velocity field
    CalculateDivergence()

    PressureField.fill(0)
    PressureField_Old.fill(0)
    #Solve Pressure
    for i in range(PressureIterationCount): 
        ComputePressure()
        EnforceBoundaryConditions_Pressure()
        lastIteration = (i == PressureIterationCount-1) 
         #copy new to old - for next iteration
        if not lastIteration:
            PressureField_Old.copy_from(PressureField) #TODO: Ping pong instead of copy
    #remove gradient of Pressure from velocity (ie: remove divergence)
    RemoveDivergenceFromVelocity()
    EnforceBoundaryConditions_Velocity()


    DieField_Old.copy_from(DieField)
    AdvectDie()
    EnforceBoundaryConditions_Die()


    ##############################
    ## Render
    ##############################
    match DisplayedBuffer:
        case 0:
            gui.set_image(DieField)
        case 1:
            GenerateDebugVelocityField()
            #gui.vector_field(VelocityField)
            gui.set_image(DebugVelocityField)
        case 2:
            gui.set_image(DivergenceField)
        case 3:
            GenerateDebugPressureField()
            gui.set_image(DebugPressureField)
        case _:
            gui.set_image(DieField)


    gui.show()
