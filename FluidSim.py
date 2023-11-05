import taichi as ti
import taichi.math as tm

import sys
#sys.tracebacklimit=0 #shorter callstacks

ti.init(arch=ti.gpu)

dim = 320
VelocityField = ti.Vector.field(n=2, dtype=float, shape=(dim, dim))
DebugVelocityField = ti.Vector.field(n=2, dtype=float, shape=(dim, dim))

DivergenceField = ti.field(ti.f32, shape=(dim, dim))
PressureField = ti.field(ti.f32, shape=(dim, dim))
PressureField_Old = ti.field(ti.f32, shape=(dim, dim))
DebugPressureField = ti.field(ti.f32, shape=(dim,dim))

DieField = ti.field(ti.f32, shape=(dim,dim))

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
    DieField.fill(0.0)
    for i, j in DieField:  # Parallelized over all pixels
        if i > ((dim/2)-20) and i < ((dim/2)+20) and j > ((dim/2)-20) and j < ((dim/2)+20):
            DieField[i,j] = 1.0

def ReadInput(gui : ti.GUI, PrevFrameCursorPos : tm.vec2):
    CursorPos = ti.Vector( gui.get_cursor_pos() )
    CursorVelocity = (CursorPos - PrevFrameCursorPos)
    return CursorPos, CursorVelocity

@ti.kernel
def Init():
    ResetFields()

@ti.kernel
def Reset():
    ResetFields()

velocityStampField = ti.field(ti.i16, shape=(20,20))
@ti.kernel
def AddInputVelocity(Pos: tm.vec2, Velocity : tm.vec2):
    for i,j in velocityStampField:
        VelocityField[tm.clamp(int(Pos[0] * dim) + (i-10), 0, dim), tm.clamp(int(Pos[1] * dim) + (j-10), 0, dim)] += Velocity * 30.
    
@ti.kernel
def AdvectVelocity():
    for i, j in VelocityField:  # Parallelized over all pixels
        CurrCellVel = VelocityField[i,j]
        VelocityField[i,j] = VelocityField[tm.clamp( i-int(CurrCellVel.x), 0, dim), tm.clamp(j-int(CurrCellVel.y), 0, dim)] #should do interpolation


@ti.kernel
def AdvectDie():
    for i, j in VelocityField:  # Parallelized over all pixels
        CurrCellVel = VelocityField[i,j]
        DieField[i,j] = DieField[tm.clamp( i-int(CurrCellVel.x), 0, dim), tm.clamp(j-int(CurrCellVel.y), 0, dim)] #should do interpolation

@ti.kernel
def CalculateDivergence():
    for i, j in VelocityField:  # Parallelized over all pixels
        if i == 0 or i == (dim-1) or j == 0 or j == (dim-1):
            DivergenceField[i,j] = 0
        else:     
            DivergenceField[i,j] = ((VelocityField[i+1,j].x - VelocityField[i-1,j].x) / 2. ) + ((VelocityField[i,j+1].y - VelocityField[i,j-1].y) / 2.) 

@ti.kernel
def ComputePressure():
    #alpha = (1.0 / dim) * (1.0/dim)
    for i,j in PressureField:
        if not (i == 0 or i == (dim-1) or j == 0 or j == (dim-1)) : #not an edge
            PressureField[i,j] = ((PressureField_Old[i+1,j] + PressureField_Old[i-1,j] + PressureField_Old[i,j+1] + PressureField_Old[i,j-1]) + (-DivergenceField[i,j])) * 0.25

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
            PressureField[i,j] = PressureField_Old[i+1,j]
        elif i == (dim-1):
            PressureField[i,j] = PressureField_Old[i-1,j]
        elif j == 0:
            PressureField[i,j] = PressureField_Old[i,j+1]
        elif j == (dim-1):
            PressureField[i,j] = PressureField_Old[i,j-1]

@ti.kernel
def EnforceBoundaryConditions_Velocity():
    for i,j in VelocityField:
        if i == 0:
            VelocityField[i,j] = tm.vec2(-VelocityField[i+1,j].x, VelocityField[i+1,j].y)
        elif i == (dim-1):
            VelocityField[i,j] = tm.vec2(-VelocityField[i-1,j].x, VelocityField[i-1,j].y)
        elif j == 0:
            VelocityField[i,j] = tm.vec2(VelocityField[i,j+1].x, -VelocityField[i,j+1].y)
        elif j == (dim-1):
            VelocityField[i,j] = tm.vec2(VelocityField[i,j-1].x, -VelocityField[i,j-1].y)

@ti.kernel
def EnforceBoundaryConditions_Die():
    for i,j in VelocityField:
        if (i == 0 or i == (dim-1) or j == 0 or j == (dim-1)) : #if edge
            DieField[i,j] = 0

@ti.kernel
def GenerateDebugVelocityField():
    for i, j in VelocityField:  # Parallelized over all pixels
        DebugVelocityField[i,j] = ((VelocityField[i,j] ) * 0.5) + 0.5


@ti.kernel
def GenerateDebugPressureField():
    for i,j in PressureField:
        DebugPressureField[i,j] = (PressureField[i,j] * 0.5) + 0.5

###################
###################
###################
###################
PrevFrameCursorPos = tm.vec2(0.0,0.0) 
gui = ti.GUI("Julia Set", res=(dim, dim))
Init()
JacobiIterationCount = 60

DisplayedBuffer = 0

while gui.running:

    #Keyboard input (espace to close, space to reset, A/Z/E/R to swap between buffer visualisation)
    if gui.get_event(ti.GUI.RELEASE):
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

    print(DisplayedBuffer)

    PrevFrameCursorPos, Velocity = ReadInput(gui, PrevFrameCursorPos)
    AddInputVelocity(PrevFrameCursorPos, Velocity)
    EnforceBoundaryConditions_Velocity()

    #Calculate divergence from velocity field
    CalculateDivergence()

    PressureField.fill(0)
    PressureField_Old.fill(0)
    #Solve Pressure
    for i in range(JacobiIterationCount):
        ComputePressure()
        EnforceBoundaryConditions_Pressure()
        lastIteration = (i == JacobiIterationCount-1) 
         #copy new to old - for next iteration
        if not lastIteration:
            PressureField_Old.copy_from(PressureField)
    #remove gradient of Pressure from velocity (ie: remove divergence)
    RemoveDivergenceFromVelocity()

    #print(VelocityField)
    AdvectVelocity()
    EnforceBoundaryConditions_Velocity()

    AdvectDie()
    EnforceBoundaryConditions_Die()

    match DisplayedBuffer:
        case 0:
            gui.set_image(DieField)
        case 1:
            GenerateDebugVelocityField()
            gui.set_image(DebugVelocityField)
        case 2:
            gui.set_image(DivergenceField)
        case 3:
            GenerateDebugPressureField()
            gui.set_image(DebugPressureField)
        case _:
            gui.set_image(DieField)


    gui.show()
