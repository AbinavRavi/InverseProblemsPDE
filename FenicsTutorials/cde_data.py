from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

T = 5.0            # final time
num_steps = 500   # number of time steps
dt = T / num_steps # time step size
nx = ny = 50

#define the mesh
mesh = UnitSquareMesh(nx,ny)
V = FunctionSpace(mesh)

#define boundary conditions
u_D = Expression('for(int i=0;i<9;i++){std::normal_distribution<double> distribution(0.0,0.02)*(cos(x[0]+x[1])+sin(x[0]+x[1]))}')

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V,u_D,boundary)

u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0.0)

#define the variational problem
F = ((u-u_n)/dt)*v*dx + dot(w,grad(u))*v*dx + dot(grad(u),grad(v))*dx - dot(f,v)*dx

#solve the variational problem
u = Function(V)
solve(a == L,u,bc)
