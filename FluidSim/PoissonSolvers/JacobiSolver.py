import taichi as ti
import taichi.math as tm

_DEBUG = False

# Configuration
dim = 4
DimLvl1 = dim


# Taichi Initialisation
if _DEBUG:
    ti.init(arch=ti.gpu, kernel_profiler=True)
else:
    ti.init(arch=ti.gpu)

# Buffer creation

#Resolution level 1    
VHField_Ping = ti.field(ti.f32, shape=(DimLvl1, DimLvl1))      # approximation of "u" (unknown) field at a given iteration
VHField_Pong = ti.field(ti.f32, shape=(DimLvl1, DimLvl1))      # approximation of "u" (unknown) field at a given iteration
FHField = ti.field(ti.f32, shape=(DimLvl1, DimLvl1))
ResidualField = ti.field(ti.f32, shape=(DimLvl1, DimLvl1))

# @ti.dataclass
# class PoissonProblem:
#     Size : tm.vec2
#     def __init__(self, size) -> None:
#         self.Size = size

@ti.func
def Reset():
    VHField_Ping.fill(0.)
    VHField_Pong.fill(0.)
#    FHField.fill(0.)
    FHField[0,0] = 7
    FHField[0,1]= 10
    FHField[0,2]= 13
    FHField[0,3]= 16

    FHField[1,0]= 20
    FHField[1,1]= 24
    FHField[1,2]= 28
    FHField[1,3]= 32

    FHField[2,0]= 36
    FHField[2,1]= 40
    FHField[2,2]= 44
    FHField[2,3]= 48

    FHField[3,0]= 35
    FHField[3,1]= 38
    FHField[3,2]= 41
    FHField[3,3]= 44

    ResidualField.fill(0.)

@ti.kernel
def Init():
    Reset()

@ti.func
def SampleField(f, coord : tm.vec2) -> ti.f32:
    extents = tm.vec2(f.shape[0]-1, f.shape[1]-1)
    coord = tm.clamp(coord, tm.vec2(0,0), extents)
    return f[ti.i32(coord.x), ti.i32(coord.y)]

@ti.kernel
def Jacobi(fh : ti.template(), vh : ti.template(), vvh : ti.template(), h2 : ti.f32): # fh = solution field, vh = old approximation value (being read), vvh = new approximation value, h2 = cell width squared  
    for i, j in vvh:  # Parallelized over all pixels
        s = ( SampleField(vh, tm.vec2(i-1,j) ) + SampleField(vh, tm.vec2(i+1,j) ) + SampleField(vh, tm.vec2(i,j-1) ) + SampleField(vh, tm.vec2(i,j+1) ) - h2 * SampleField(fh, tm.vec2(i,j)) ) / 4.
        vvh[i,j] = s

# @ti.kernel
# def Residual( fh, vh, rh):
#     for i, j in fh:  # Parallelized over all pixels
#         rh[i,j] = - SampleField(fh, (i,j)) - ()

###############################################################################################
# Main Program
###############################################################################################
Init()
PrevFrameCursorPos = tm.vec2(0.0,0.0) 
gui = ti.GUI("Jacobi Solver", res=(DimLvl1, DimLvl1))

# Au = f        # A is a sparse matrix .  u is the unknown vector. and f is the right hand side of the equation (ie: the solution) 
# u = A^-1 · f  # Invert of A multiplied by f equals the unknown u. But inverse of A is complex to calculate & takes a lot of memory.
# vα = approximation of unknown "u", at iteration α, which converges to A^-1 · f

## Example:
# Original matrix A:
# 2 1 0 0
# 1 2 1 0
# 0 1 2 1
# 0 0 1 2

#Inverse of matrix A:
# 4/5 -3/5 2/5 -1/5
# -3/5 6/5 -4/5 2/5
# 2/5 -4/5 6/5 -3/5
# -1/5 2/5 -3/5 4/5

# unknown U:
# 1 2 3 4
# 5 6 7 8
# 9 10 11 12
# 13 14 15 16

# solution f: 
# 7 10 13 16
# 20 24 28 32
# 36 40 44 48
# 35 38 41 44

##############################
## Solver
##############################
CellSize = 1. #/ DimLvl1

ping = True
IterationCount = 60
for i in range(IterationCount):
    if ping:
        Jacobi(FHField, VHField_Ping, VHField_Pong, CellSize )
    else:
        Jacobi(FHField, VHField_Pong, VHField_Ping, CellSize )

    ping = not ping

ping = not ping #flip back to last written ping/pong 

while gui.running:


    ##############################
    ## Render
    ##############################
    gui.set_image(VHField_Ping if ping else VHField_Pong)
    gui.show()
