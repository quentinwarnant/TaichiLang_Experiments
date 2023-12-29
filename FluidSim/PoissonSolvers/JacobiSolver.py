import taichi as ti
import taichi.math as tm
from enum import Enum
import math
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

_DEBUG = False

# Configuration
dim = 64
dimSqr = dim * dim

class Solver(Enum):
    Jacobi = 1
    GaussSeidelRB = 2

ActiveSolver = Solver.GaussSeidelRB

# Taichi Initialisation
if _DEBUG:
    ti.init(arch=ti.gpu, kernel_profiler=True)
else:
    ti.init(arch=ti.gpu)

# Buffer creation

#Resolution level 1    
VHField_Ping = ti.field(ti.f32, shape=(dim, dim))      # approximation of "u" (unknown) field at a given iteration
VHField_Pong = ti.field(ti.f32, shape=(dim, dim))      # approximation of "u" (unknown) field at a given iteration
FHField = ti.field(ti.f32, shape=(dim, dim))
ResidualField = ti.field(ti.f32, shape=(dim, dim))

@ti.func
def Reset():
    VHField_Ping.fill(0.)
    VHField_Pong.fill(0.)

    FHField.fill(0.)

    #b aka F aka fh
    # for i in range(0,dim):
    #     FHField[i,0] = 60.
    # for i in range (60, dim-60):   
    #     for j in range(0,10):
    #         FHField[i,dim-2-j] = -15.
    #         FHField[j,i] = -15.

    for i in range(0,dim):
        FHField[i,3] = 60.
    for i in range (3, dim-3):   
        for j in range(0,13):
            FHField[i,dim-2-j] = -15.
            FHField[j,1+i] = -15.





    # for x,y in FHField:
    #     FHField[x,y] = -5. * tm.exp(x) * tm.exp(-2. * y)


    ResidualField.fill(0.)

@ti.kernel
def Init():
    Reset()

@ti.func
def SampleField(f, coord : tm.vec2) -> ti.f32:
    extents = tm.vec2(f.shape[0]-1, f.shape[1]-1)
    val = 0.
    if coord.x < 0 or coord.x >= extents.x or coord.y < 0 or coord.y >= extents.y:
        val = 0.
    else:
        val = f[ti.i32(coord.x), ti.i32(coord.y)]
    return val

@ti.func
def TwoDCoordToOneD(coord : tm.ivec2) -> ti.i32:
    return (coord.y * dim) + coord.x

@ti.func
def OneDCoordToTwoD(coord : ti.i32) -> tm.ivec2:
    y = tm.floor(coord / dim)
    x = coord - (y*dim)
    return tm.ivec2(x,y)

@ti.func
def OffsetTwoDCoord(coord : tm.ivec2, offset : ti.i32) -> tm.ivec2:
    return OneDCoordToTwoD( TwoDCoordToOneD(coord) + offset)

@ti.kernel
def Jacobi(fh : ti.template(), vh : ti.template(), vvh : ti.template(), h2 : ti.f32, residualField : ti.template()): # fh = solution field, vh = old approximation value (being read), vvh = new approximation value, h2 = cell width squared  
    for i, j in vvh:  # Parallelized over all pixels
        if i != 0 and i != dim-1 and j != 0 and j != dim-1:
            s = ( SampleField(vh, tm.vec2(i-1,j) ) + SampleField(vh, tm.vec2(i+1,j) ) + SampleField(vh, tm.vec2(i,j-1) ) + SampleField(vh, tm.vec2(i,j+1) ) - h2 * SampleField(fh, tm.vec2(i,j)) ) / 4.
            vvh[i,j] = s
            residualField[i,j] = tm.pow(vh[i,j] - vvh[i,j], 2.)

@ti.kernel
def GaussSeidel(fh : ti.template(), vh : ti.template(), vvh : ti.template(), h2 : ti.f32, residualField : ti.template(), red : ti.i8): # fh = solution field, vh = old approximation value (being read), vvh = new approximation value, h2 = cell width squared  
    halfDim = dim/2
    halfGrid = dim * halfDim
    for index in range(halfGrid): # Parallelized
        j = tm.floor(index/halfDim, ti.i32)
        tmpI = ti.i32(index - (j * halfDim))

        offset = 0
        jIsEven = (j % 2 == 0)
        if red == 1:
            offset = 0 if jIsEven else 1
        else:
            offset = 1 if jIsEven else 0

        i = (tmpI * 2) + offset

        
        # not working properly yet...
        if i != 0 and i != dim-1 and j != 0 and j != dim-1:
            s = ( SampleField(vh, tm.vec2(i-1,j) ) + SampleField(vh, tm.vec2(i+1,j) ) + SampleField(vh, tm.vec2(i,j-1) ) + SampleField(vh, tm.vec2(i,j+1) ) - h2 * SampleField(fh, tm.vec2(i,j)) ) / 4.


            vvh[i,j] = s * 0.5
            residualField[i,j] = tm.pow(vh[i,j] - vvh[i,j], 2.)

AbsVField = ti.Vector.field(2, ti.f32, shape=(dim, dim)) 
AbsFHField = ti.Vector.field(2, ti.f32, shape=(dim, dim))  
@ti.kernel
def AbsVals(vField : ti.template(), fhField : ti.template()):
    for i,j in vField:
        AbsVField[i,j].x = (vField[i,j]) * 100 #Positive values
        AbsVField[i,j].y = -(vField[i,j]) * 100 #Negative Values
    for i,j in fhField:
        AbsFHField[i,j].x = (fhField[i,j]) * 10 #Positive values
        AbsFHField[i,j].y = -(fhField[i,j]) * 10 #Negative Values


###############################################################################################
# Main Program
###############################################################################################
Init()
PrevFrameCursorPos = tm.vec2(0.0,0.0) 
gui = ti.GUI("Poisson Equation Solver", res=(dim, dim))

# Au = f        # A is a sparse matrix .  u is the unknown vector. and f is the right hand side of the equation (ie: the solution) 
# u = A^-1 · f  # Invert of A multiplied by f equals the unknown u. But inverse of A is complex to calculate & takes a lot of memory.
# vα = approximation of unknown "u", at iteration α, which converges to A^-1 · f

##############################
## Solver
##############################
CellSize = 1. /  dim
CellSizeSqr = CellSize * CellSize

ping = True
residualThreshold = 1e-6
i = 0
while True:
    i+=1
    if ActiveSolver == Solver.Jacobi:
        if ping:
            Jacobi(FHField, VHField_Pong, VHField_Ping, CellSizeSqr, ResidualField )
        else:
            Jacobi(FHField, VHField_Ping, VHField_Pong, CellSizeSqr, ResidualField )
    elif ActiveSolver == Solver.GaussSeidelRB:
            GaussSeidel(FHField, VHField_Ping, VHField_Pong, CellSizeSqr, ResidualField, 1 )
            GaussSeidel(FHField, VHField_Pong, VHField_Ping, CellSizeSqr, ResidualField, 0 )

    ping = not ping

    numpResidualArray = ResidualField.to_numpy()
    residualSum = numpResidualArray.sum() 
    residual = math.sqrt(residualSum) / math.sqrt(dimSqr)
    if residual  < residualThreshold: # use L2Norm(residual, fine.size) // sqrt(sum(r*r))
        print("performed ", i, " iterations before reaching residual threshold")
        break

ping = not ping #flip back to last written ping/pong 
if ActiveSolver == Solver.Jacobi:
    AbsVals(VHField_Ping if ping else VHField_Pong, FHField)
elif ActiveSolver == Solver.GaussSeidelRB:
    AbsVals(VHField_Ping, FHField)

DisplayedBuffer = 2

while gui.running:
    ##############################
    ## Render
    ##############################
    if gui.get_event(ti.GUI.RELEASE, ti.GUI.PRESS):
        if gui.event.type == ti.GUI.RELEASE:
            if gui.event.key == ti.GUI.ESCAPE:
                gui.close()
                break
            if gui.event.key == ti.GUI.SPACE:
                Reset()
            if  gui.event.key == 'a':
                DisplayedBuffer = 1
            if gui.event.key == 'z':
                DisplayedBuffer = 2

    if DisplayedBuffer == 1:
        gui.set_image(AbsVField)
    elif DisplayedBuffer == 2:
        gui.set_image(AbsFHField)

    gui.show()
