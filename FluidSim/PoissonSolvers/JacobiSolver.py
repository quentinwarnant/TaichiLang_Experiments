import taichi as ti
import taichi.math as tm

_DEBUG = False

# Configuration
dim = 256
dimSqr = dim * dim

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
    for i in range(0,dim):
        FHField[i,0] = 60.
    for i in range (60, dim-60):   
        for j in range(0,10):
            FHField[i,dim-2-j] = -15.
            FHField[j,i] = -15.

    # for x,y in FHField:
    #     FHField[x,y] = -5. * tm.exp(x) * tm.exp(-2. * y)


    ResidualField.fill(0.)

@ti.kernel
def Init():
    Reset()

@ti.func
def SampleField(f, coord : tm.vec2) -> ti.f32:
    extents = tm.vec2(f.shape[0]-1, f.shape[1]-1)
    #coord = tm.clamp(coord, tm.vec2(0,0), extents)
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

@ti.func
def ComputeResidual(GridIndexX, GridIndexY, currentXGuess : ti.f32, b : ti.template()) -> ti.f32:
    #construct A row which represents
    LinearGridIndex = (GridIndexY * dim) + GridIndexX
    # Matrix A coeficients for this linear system of equations line
    coefLeftMostDiagonal = 0 if GridIndexY < 3 else 1
    coefOffDiagonalLeft = 0 if GridIndexY < 1 else 1
    coefDiagonal = -4
    coefOffDiagonalRight = 0 if GridIndexY < (dim - 2) else 1
    coefRightMostDiagonal = 0 if GridIndexY < (dim - 4) else 1

    coord = tm.ivec2(GridIndexX, GridIndexY)
    total = b[coord] * coefDiagonal
    if coefLeftMostDiagonal != 0:
        total += b[OffsetTwoDCoord(coord,-3)] * coefLeftMostDiagonal
    if coefOffDiagonalLeft != 0:
        total += b[OffsetTwoDCoord(coord,-1)] * coefOffDiagonalLeft
    if coefOffDiagonalRight != 0:
        total += b[OffsetTwoDCoord(coord,1)] * coefOffDiagonalRight
    if coefRightMostDiagonal != 0:
        total += b[OffsetTwoDCoord(coord,3)] * coefRightMostDiagonal

    #if abs(currentXGuess - total) > 0:
    #    print("residual[",GridIndexX,",",GridIndexY,"] = ", currentXGuess - total)
    return ti.abs(currentXGuess - total)

@ti.kernel
def Jacobi(fh : ti.template(), vh : ti.template(), vvh : ti.template(), h2 : ti.f32, residualField : ti.template()): # fh = solution field, vh = old approximation value (being read), vvh = new approximation value, h2 = cell width squared  
    for i, j in vvh:  # Parallelized over all pixels
        s = ( SampleField(vh, tm.vec2(i-1,j) ) + SampleField(vh, tm.vec2(i+1,j) ) + SampleField(vh, tm.vec2(i,j-1) ) + SampleField(vh, tm.vec2(i,j+1) ) - h2 * SampleField(fh, tm.vec2(i,j)) ) / 4.
        vvh[i,j] = s
        residualField[i,j] = ti.abs(vh[i,j] - vvh[i,j])
        #residualField[i,j] = ComputeResidual(i,j, s, fh) #not working correctly...


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
gui = ti.GUI("Jacobi Solver", res=(dim, dim))

# Au = f        # A is a sparse matrix .  u is the unknown vector. and f is the right hand side of the equation (ie: the solution) 
# u = A^-1 · f  # Invert of A multiplied by f equals the unknown u. But inverse of A is complex to calculate & takes a lot of memory.
# vα = approximation of unknown "u", at iteration α, which converges to A^-1 · f

##############################
## Solver
##############################
CellSize = 1. /  dim
CellSizeSqr = CellSize * CellSize

ping = True
residualThreshold = 1e-1
i = 0
while True:
    i+=1
    if ping:
        Jacobi(FHField, VHField_Pong, VHField_Ping, CellSizeSqr, ResidualField )
    else:
        Jacobi(FHField, VHField_Ping, VHField_Pong, CellSizeSqr, ResidualField )

    ping = not ping

    numpResidualArray = ResidualField.to_numpy()
    residualSum = numpResidualArray.sum()
    if residualSum  < residualThreshold:
        print("performed ", i, " jacobi iterations before reaching residual threshold")
        break

ping = not ping #flip back to last written ping/pong 

AbsVals(VHField_Ping if ping else VHField_Pong, FHField)
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
