import taichi as ti
import taichi.math as tm

import sys
sys.tracebacklimit=0 #shorter callstacks

ti.init(arch=ti.gpu)

dim = 320
VelocityField = ti.Vector.field(n=2, dtype=float, shape=(dim, dim))
DebugVelocityField = ti.Vector.field(n=2, dtype=float, shape=(dim, dim))
DivergenceField = ti.Vector.field(n=1, dtype=float, shape=(dim, dim))
PressureField = ti.Vector.field(n=1, dtype=float, shape=(dim, dim))

DieField = ti.field(ti.f32, shape=(dim,dim))

@ti.func
def ResetFields():
    ClearVelocityField()
    SetInitialDieSetup()

@ti.func
def ClearVelocityField():  # complex square of a 2D vector
    VelocityField.fill([0.0,0.0])

@ti.func
def SetInitialDieSetup():
    DieField.fill(0.0)
    for i, j in DieField:  # Parallelized over all pixels
        if i > ((dim/2)-20) and i < ((dim/2)+20) and j > ((dim/2)-20) and j < ((dim/2)+20):
            DieField[i,j] = 1.0


def ReadInput(gui : ti.GUI, PrevFrameCursorPos : tm.vec2):
    CursorPos = ti.Vector( gui.get_cursor_pos() )
    CursorVelocity = (CursorPos - PrevFrameCursorPos) * 20.0
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
        VelocityField[tm.clamp(int(Pos[0] * dim) + (i-10), 0, dim), tm.clamp(int(Pos[1] * dim) + (j-10), 0, dim)] += Velocity * 20
    
@ti.kernel
def AdvectVelocity():
    for i, j in VelocityField:  # Parallelized over all pixels
        CurrCellVel = VelocityField[i,j]
        VelocityField[i,j] = VelocityField[tm.clamp( i-int(CurrCellVel.x), 0, dim), tm.clamp(j-int(CurrCellVel.y), 0, dim)] #should do interpolation


@ti.kernel
def AdvectDensity():
    for i, j in VelocityField:  # Parallelized over all pixels
        CurrCellVel = VelocityField[i,j]
        DieField[i,j] = DieField[tm.clamp( i-int(CurrCellVel.x), 0, dim), tm.clamp(j-int(CurrCellVel.y), 0, dim)] #should do interpolation



@ti.kernel
def CalculateDivergence():
    for i, j in VelocityField:  # Parallelized over all pixels
        DivergenceField[i,j] = VelocityField[TODO] - VelocityField[TODO]

@ti.kernel
def ComputePressure():


@ti.kernel
def RemoveDivergenceFromVelocity():




@ti.kernel
def GenerateDebugVelocityField():
    for i, j in VelocityField:  # Parallelized over all pixels
        DebugVelocityField[i,j] = ((VelocityField[i,j] ) * 0.5) + 0.5

###################
PrevFrameCursorPos = tm.vec2(0.0,0.0) 
gui = ti.GUI("Julia Set", res=(dim, dim))
Init()

while gui.running:
    if gui.get_event((ti.GUI.RELEASE, ti.GUI.SPACE)):
        Reset()

    PrevFrameCursorPos, Velocity = ReadInput(gui, PrevFrameCursorPos)
    #if gui.is_pressed("A"):
    AddInputVelocity(PrevFrameCursorPos, Velocity)

    CalculateDivergence()
    ComputePressure()
    RemoveDivergenceFromVelocity()

    #print(VelocityField)
    AdvectVelocity()
    AdvectDensity()

    gui.set_image(DieField)

    # GenerateDebugVelocityField()
    # gui.set_image(DebugVelocityField)

    gui.show()