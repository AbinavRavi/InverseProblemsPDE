from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

T = 5.0            # final time
num_steps = 500   # number of time steps
dt = T / num_steps # time step size
nx = ny = 50

#define the mesh
mesh = UnitSquareMesh(nx,ny)
V = VectorFunctionSpace(mesh,'P',1)

#define boundary conditions
u_D = Expression('cos(x[0]+x[1])+sin(x[0]+x[1])',degree=1)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V,u_D,boundary)
# (u0,u1,u2) = split(Function(V))
u = TrialFunction(V)
# (v1,v2,v3) = TestFunctions(V)
v = TestFunction(V)
f = Constant(0.0)
u_n = Function(V)
w =Function(V)
# + dot(w,grad(u))*v*dx 
#define the variational problem
a = ((u-u_n)/dt)*v*dx + dot(grad(u),grad(v))*dx
# a = lhs(F)
L = f*v*dx

#solve the variational problem
u = Function(V)
# solve(a == L,u,bc)

#time stepping of the problem
t = 0

for n in range(num_steps):
    t+=dt
    b = assemble(L)
    bc.apply(b)
    solve(a == L,u,bc)
    plt.figure(n)
    plot(u)
    plt.show()

def write(u,nx,ny):
    a = np.reshape(u.compute_vertex_values(),[nx+1,ny+1])
    np.savetxt("sol_cda.csv",a,delimiter=",")

write(u,nx,ny)
