from fenics import *
import matplotlib.pyplot as plt
import numpy as np

#create mesh and define function space
mesh = UnitSquareMesh(100,100)
V = FunctionSpace(mesh,'P',1)

#define boundary conditions
u_D = Expression('1+x[0]*x[0]+ 2*x[1]*x[1]',degree=2)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V,u_D,boundary)

#define the variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0.0)
a = dot(grad(u),grad(v))*dx
L = f*v*dx

#compute solution
u = Function(V)
solve(a == L, u,bc)

#plot solution and mesh
plot(u)
plot(mesh)

#write to vtk file
# vtkfile = File('solution.pvd')
# vtkfile << u
x = V.tabulate_dof_coordinates()
x = x.reshape((-1,mesh.geometry().dim()))
# print(x)
print(u.vector()[:])
np.savetxt("sol100.txt",u.vector()[:])
#compute the error in L2 norm
error_L2 = errornorm(u_D, u, 'L2')

#print error
print('error_L2 = ',error_L2)
plt.show()