from fenics import *
import matplotlib.pyplot as plt
import numpy as np

# Set the constants
T = 10
num_steps = 100
dt = T/num_steps
nx = ny = 10

# set the mesh and geometry
P = FiniteElement('P',triangle,1)
mesh = UnitSquareMesh(nx,ny)
V = FunctionSpace(mesh,P)
W = VectorFunctionSpace(mesh,'P',1)

# set the boundary conditions
u_D = Expression('cos(x[0]+x[1])+sin(x[0]+x[1])',degree=1)
def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V,u_D,boundary)

# set the test and trial functions
u = TrialFunction(V)
v = TestFunction(V)
w = Function(W)
u_n = Function(V)
f = Constant(0.0)
F = ((u - u_n) /dt)*v*dx + dot(w, grad(u))*v*dx + dot(grad(u), grad(v))*dx - f*v*dx 
a = lhs(F)
L = rhs(F)




