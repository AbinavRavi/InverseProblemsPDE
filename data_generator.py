from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import random
import os
# import pandas as pd

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
    np.savetxt('data/'+str(sim_number)+'_parameters'+".csv",array,delimiter=",",fmt='%s')
    # np.savetxt('data/'+str(sim_number)+'_eps.csv',eps,delimiter=",",fmt='%s')

    
def Simulation(nx,ny,T,num_steps,sim_number):
    dt = T/num_steps
    Xmax = Ymax = 2*3.14           # range of 'x' and 'y'
    mesh = RectangleMesh(Point(0, 0), Point(Xmax, Ymax), nx, ny)
    poly=5
    V = FunctionSpace(mesh, 'P', poly)
    # P = FiniteElement('P',triangle,2)
    # mesh = UnitSquareMesh(nx,ny)
    # V = FunctionSpace(mesh,P)
    # W = VectorFunctionSpace(mesh,'P',2)
    # sample = np.random.uniform(-2.0,2.0)
    a = random.uniform(-2.0,2.0)
    b = random.uniform(-2.0,2.0)
    limit_A = Expression('a',degree=2, a = a)
    limit_B = Expression('b',degree=2, b = b)
    # sample2 = np.random.uniform(0.2,0.8)
    c = random.uniform(0.2,0.8)
    d = random.uniform(0.2,0.8)
    eps_lower = Expression('c',degree=2, c = c)
    eps_upper = Expression('d',degree=2, d = d)
    # u_D = U0xy(V)
    #Expression('cos(x[0]+x[1])+sin(x[0]+x[1])',degree=1)
    # bc = DirichletBC(V,u_D,boundary)
    bc = DirichletBC(V, Constant(0), boundary)

    # set the test and trial functions
    u = TrialFunction(V)
    v = TestFunction(V)
    # w = Function(W)
    u = Function(V)
    u_n =  U0xy(V)
    f = Constant(0.0)
    # k = Constant(np.random.uniform(-2,2))
    # eps = Constant(random.uniform(0.2,0.8))
    w = as_vector([limit_A,limit_B])
    eps = as_matrix([[eps_lower,0],[0,eps_upper]])
    # print(eps.shape)
    DiffTerm=dot(dot(grad(u),eps),grad(v))*dx
    Convterm=dot(grad(u),w)*v*dx
    F=(u-u_n)*v*dx + dt*DiffTerm + dt*Convterm
    array = [a,b,c,d]
    # F = ((u - u_n) /dt)*v*dx + k*dot(w, grad(u))*v*dx + eps*dot(grad(u), grad(v))*dx - f*v*dx 
    t = 0
    for n in range(1,num_steps+1):
    
    # Update current time
        t += dt
        u_n.t = t 
        # A = assemble(lhs(F))
        # b = assemble(rhs(F))
        # bc.apply(b)

    # Solve variational problem for time step
        solve(F == 0, u,bc)

    # Update previous solution
        u_n.assign(u)
        write_function_values(u,nx,ny,n,num_steps,sim_number)
        # plot(u,mode='color')
        # plt.axis('off')
        
        # plt.savefig('data/'+str(sim_number)+'/'+str(n)+'.png')
    write_parameters(array,sim_number)  
# Hold plot


for i in range(1,11):
    Simulation(128,128,10,50,i)