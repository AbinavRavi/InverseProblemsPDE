from fenics import *
import matplotlib.pyplot as plt


#create mesh and define function space
mesh = UnitSquareMesh(10,10)
V = FunctionSpace(mesh,'P',3)

#define boundary conditions
u_D = Expression('1+x[0]*x[0]+ 2*x[1]*x[1]',degree=3)

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
vtkfile = File('poisson/solution.pvd')
vtkfile << u

#compute the error in L2 norm
error_L2 = errornorm(u_D, u, 'L2')

#print error
print('error_L2 = ',error_L2)
plt.show()