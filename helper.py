from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import os

#Function for setting the boundary condition as random initialisation
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

def boundary(x, on_boundary):
    return on_boundary

# function to write the simulated quantities in a csv file
# """TO DO: Modify the write file to store the multiple timesteps as a single tensor along with the velocity vector value and the diffusivity coefficient value """
def write(u,nx,ny,timestep,num_steps,sim_number):
    a = np.reshape(u.compute_vertex_values(),[nx+1,ny+1])
    if not os.path.exists('results/64grid/'+str(sim_number)+'/'):
        os.makedirs('results/64grid/'+str(sim_number)+'/')
    np.savetxt('results/64grid/'+str(sim_number)+'/'+str(timestep)+".csv",a,delimiter=",")

def Simulation(nx,ny,T,num_steps,sim_number):
    dt = T/num_steps
    P = FiniteElement('P',triangle,1)
    mesh = UnitSquareMesh(nx,ny)
    V = FunctionSpace(mesh,P)
    W = VectorFunctionSpace(mesh,'P',1)

    u_D = U0xy(V) #Expression('cos(x[0]+x[1])+sin(x[0]+x[1])',degree=1)
    bc = DirichletBC(V,u_D,boundary)

    # set the test and trial functions
    u = TrialFunction(V)
    v = TestFunction(V)
    w = Function(W)
    u = Function(V)
    u_n = Function(V)
    f = Constant(0.0)
    eps = Constant(random.uniform(0.2,0.8))
    F = ((u - u_n) /dt)*v*dx + dot(w, grad(u))*v*dx + eps*dot(grad(u), grad(v))*dx - f*v*dx 

    timeseries = TimeSeries('timeseries_cde')
    t = 0
    for n in range(1,num_steps+1):
    
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
        write(u,nx,ny,n,num_steps,sim_number)
        plot(u)
        plt.axis('off')
    plt.savefig('results/64grid/'+str(sim_number)+'/'+str(n)+'.png')
    # Update progress bar
    # progress.update(t / T)

# Hold plot


for i in range(1,51):
    Simulation(64,64,10,100,i)