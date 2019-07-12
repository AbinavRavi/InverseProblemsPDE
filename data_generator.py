from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from tqdm import tqdm

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
def write_function_values(u,nx,ny,timestep,num_steps,sim_number):
    a = np.reshape(u.compute_vertex_values(),[nx+1,ny+1])
    if not os.path.exists('data/'+str(sim_number)+'/'):
        os.makedirs('data/'+str(sim_number)+'/')
    np.savetxt('data/'+str(sim_number)+'/'+str(timestep)+".csv",a,delimiter=",")
    

def write_parameters(array,sim_number):
    np.savetxt('data/'+str(sim_number)+'_parameters'+".csv",array,delimiter=",")

    
def Simulation(nx,ny,T,num_steps,sim_number):
    dt = T/num_steps
    P = FiniteElement('P',triangle,2)
    mesh = UnitSquareMesh(nx,ny)
    V = FunctionSpace(mesh,P)
    # W = VectorFunctionSpace(mesh,'P',2)
    w_lower = Expression('-2.0',degree=2)
    w_upper = Expression('2.0',degree=2)
    eps_lower = Expression('0.2',degree=6)
    eps_upper = Expression('0.8',degree=6)
    u_D = U0xy(V) #Expression('cos(x[0]+x[1])+sin(x[0]+x[1])',degree=1)
    bc = DirichletBC(V,u_D,boundary)

    # set the test and trial functions
    u = TrialFunction(V)
    v = TestFunction(V)
    # w = Function(W)
    u = Function(V)
    u_n = Function(V)
    f = Constant(0.0)
    # k = Constant(np.random.uniform(-2,2))
    # eps = Constant(random.uniform(0.2,0.8))
    w = as_vector([w_lower,w_upper])
    eps = as_matrix([[eps_lower,0],[0,eps_upper]])
    DiffTerm=dot(dot(grad(u),eps),grad(v))*dx
    Convterm=dot(grad(u),w)*v*dx
    F=(u-u_n)*v*dx + dt*DiffTerm + dt*Convterm
    # F = ((u - u_n) /dt)*v*dx + k*dot(w, grad(u))*v*dx + eps*dot(grad(u), grad(v))*dx - f*v*dx 
    array = [w,eps]
    timeseries = TimeSeries('timeseries_cde')
    t = 0
    for n in range(1,num_steps+1):
    
    # Update current time
        t += dt
        u_D.t = t 
        A = assemble(lhs(F))
        b = assemble(rhs(F))
        bc.apply(b)

    # Read velocity from file
        # timeseries.store(w.vector(), t)

    # Solve variational problem for time step
        solve(F == 0, u,bc)

    # Update previous solution
        u_n.assign(u)
        write_function_values(u,nx,ny,n,num_steps,sim_number)
        # plot(u,mode='color')
        # plt.axis('off')
        
        # plt.savefig('data/'+str(sim_number)+'/'+str(n)+'.png')
    # write_parameters(array,sim_number)  


for i in range(1):
    Simulation(64,64,10,100,i)