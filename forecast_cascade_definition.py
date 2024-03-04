import control as ct
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime



k = 1 # this is the "speed" of decay
dimension = 3 + 1 # time of peak + 1
# define a matrix A of dimension "dimension" with -k on the diagonal and +k on the subdiagonal
A = -k*np.eye(dimension) + k*np.eye(dimension, k=-1)

# define a matrix B of dimension "dimension" with 1 on the first row and 0 elsewhere
B = np.zeros((dimension,1))
B[0] = 1
# define a matrix C of dimension "dimension" with 1 on the last row and 0 elsewhere
C = np.zeros((1,dimension))
C[0,dimension-1] = 1
# define a matrix D of dimension 1x1 with 0
D = np.zeros((1,1))
# create a state space model with A, B, C, and D
sys = ct.ss(A,B,C,D)
# create a time vector from 0 to 10 with 1000 points
t_f = 7
dt = 0.01
t = np.linspace(0,t_f,int(t_f/dt))


# simulate the state space model with the step input
t,y = ct.impulse_response(sys,t)

# what is the sum of y? (scaled by the timestep)
print(np.sum(y)*dt)

# plot the output
plt.plot(t,y)
# plot a vertical line
plt.axvline(x=dimension-1, color='r', linestyle='--')

plt.show()

# just using k=1 preserves the mass of the impulse which is definitely desired. 
# it smears the impulse to 1.5% of its peak value after 2 days.
# does that smearing result in a wrong decision on the part of the controller if the mass is preserved?
# I feel like the distributed controller should be still making the approximtaely right decision. even though it will have smoother disturbances than the cnetralized controller.
# the mass balance is identical, it's only the pace of the disturbance that is different.
