from fenics import *
import matplotlib.pyplot as plt
import numpy as np

#generate a mesh
mesh = UnitSquareMesh(10,10)
V = FunctionSpace(mesh,'P',1)

#define the boundary conditions
u_D = Expression('1+x[0]*x[0]+ 2*x[1]*x[1]',degree=2)
def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V,u_D,boundary)

#define the variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0.0)
u_n = Function(V)
w = Function(V)
#other conditions
T = 5.0
num_steps = 100
dt = Constant(T/num_steps)

#define the varialtional problem
a = ((u-u_n)/dt)*v*dx + dot(w,grad(u))*v*dx + dot(grad(u),grad(v))*dx
L = f*v*dx

u = Function(V)
solve(a == L,u,bc)

# Create progress bar
progress = Progress('Time-stepping')
set_log_level(PROGRESS)

#create timeseries for the scheme of convection diffusion equation
timeseries_w = TimeSeries('velocity series') 
#plot solution and mesh
plot(u)
plot(mesh)

#writing the time stepping 
t=0
for n in range(num_steps):
    t+=dt
    timeseries_w.retrieve(w.vector(),t)
    solve(a==0,u)

#data writing
def write(u,nx,ny):
    a = np.reshape(u.compute_vertex_values(),[nx+1,ny+1])
    np.savetxt("sol.csv",a,delimiter=",")

write(u,nx,ny)

plt.show()

