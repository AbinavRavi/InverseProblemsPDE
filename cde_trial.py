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
# u_D = Expression('cos(x[0]+x[1])+sin(x[0]+x[1])',degree=1)
# def boundary(x, on_boundary):
#     return on_boundary

# bc = DirichletBC(V,u_D,boundary)

# set the test and trial functions
u = TrialFunction(V)
v = TestFunction(V)
w = Function(W)
u = Function(V)
u_n = Function(V)
f = Constant(0.0)
F = ((u - u_n) /dt)*v*dx + dot(w, grad(u))*v*dx + dot(grad(u), grad(v))*dx - f*v*dx 
# a = lhs(F)
# L = rhs(F)
# u = Function(V)

#create time series
timeseries = TimeSeries('timeseries_cde')

t = 0
for n in range(num_steps):

    # Update current time
    t += dt

    # b = assemble(lhs(F))
    # bc.apply(b)

    # Read velocity from file
    timeseries.store(w.vector(), t)

    # Solve variational problem for time step
    solve(F == 0, u)

    # Save solution to file (VTK)
    # _u_1, _u_2, _u_3 = u.split()
    # vtkfile_u_1 << (_u_1, t)
    # vtkfile_u_2 << (_u_2, t)
    # vtkfile_u_3 << (_u_3, t)

    # Update previous solution
    u_n.assign(u)
    plot(u)

    # Update progress bar
    # progress.update(t / T)

# Hold plot
plt.show()
