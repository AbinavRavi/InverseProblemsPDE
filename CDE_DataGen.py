from fenics import *
import matplotlib.pyplot as plt
import numpy as np

# Set the constants
T = 10
num_steps = 100
dt = T/num_steps
nx = ny = 64

# set the mesh and geometry
P = FiniteElement('P',triangle,1)
mesh = UnitSquareMesh(nx,ny)
V = FunctionSpace(mesh,P)
W = VectorFunctionSpace(mesh,'P',1)

#set the boundary conditions as random initialisations
def U0xy(V):
    u_n = interpolate(Constant(0), V)

    #random distribution
    mu, sigma = 0, 0.02 # mean and standard deviation

    for m in range(9):
        for n in range(9):

            #initial value
            u_0 = Expression('(rand_lambda*cos(k*x[0]+l*x[1]) + rand_gamma*sin(k*x[0]+l*x[1]))',degree=5,
                             k=np.random.uniform(0,9), l=np.random.uniform(0,9),rand_lambda = np.random.normal(mu, sigma),
                             rand_gamma = np.random.normal(mu, sigma))

            u_t = interpolate(u_0, V)

            u_n.vector().set_local(u_t.vector().get_local()+u_n.vector().get_local())
    return u_n

# set the boundary conditions 
u_D = U0xy(V) #Expression('cos(x[0]+x[1])+sin(x[0]+x[1])',degree=1)
def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V,u_D,boundary)

# set the test and trial functions
u = TrialFunction(V)
v = TestFunction(V)
w = Function(W)
u = Function(V)
u_n = Function(V)
f = Constant(0.0)
eps = Constant(1.0)
F = ((u - u_n) /dt)*v*dx + dot(w, grad(u))*v*dx + eps*dot(grad(u), grad(v))*dx - f*v*dx 
# a = lhs(F)
# L = rhs(F)
# u = Function(V)

#create time series
timeseries = TimeSeries('timeseries_cde')
# Function to store data in np.array
def write(u,nx,ny,timestep):
    a = np.reshape(u.compute_vertex_values(),[nx+1,ny+1])
    np.savetxt('results/64grid/'+ str(timestep)+".csv",a,delimiter=",")

t = 0
for n in range(num_steps):

    # Update current time
    t += dt
    A = assemble(lhs(F))
    b = assemble(rhs(F))
    bc.apply(b)

    # Read velocity from file
    timeseries.store(w.vector(), t)

    # Solve variational problem for time step
    solve(F == 0, u,bc)

    # Update previous solution
    u_n.assign(u)
    write(u,nx,ny,n)
    plot(u)
    # plt.savefig('results/64grid/'+str(n)+'.png')
    # Update progress bar
    # progress.update(t / T)

# Hold plot
plt.axis('off')
plt.show()
