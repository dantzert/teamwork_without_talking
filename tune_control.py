import sys
sys.path.append("C:/modpods")
import modpods
import pystorms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import control as ct
import dill as pickle
import pyswmm
import swmm
import datetime
import control.matlab as matlab


np.set_printoptions(precision=3,suppress=True)

# options are: 'centralized', 'hi-fi', 'lo-fi', and 'local'
control_scenario = 'centralized' 
verbose = True

print("evaluating ", control_scenario)

# project file is in english units
cfs2cms = 35.315
ft2meters = 3.281
basin_max_depths =  [10.0, 10.0, 20.0, 10.0, 10.0, 13.72] # feet
flow_threshold = np.ones(6)*3.9 # cfs

# load the discrete time model we've already trained
with open("C:/teamwork_without_talking/plant_approx_discrete_w_predictions.pickle", 'rb') as handle:
    lti_plant_approx = pickle.load(handle)
    
    
# make sure everything is a float
lti_plant_approx.A = lti_plant_approx.A.astype(float)
lti_plant_approx.B = lti_plant_approx.B.astype(float) 
#lti_plant_approx.C = lti_plant_approx.C.astype(float) 

A = lti_plant_approx.A
B = lti_plant_approx.B
C = lti_plant_approx.C
'''
# divide A by its largest eigenvalue to make it stable (if not already so)
if np.max(np.abs(np.linalg.eig(A)[0])) > 1.0:
    A = A / (np.max(np.abs(np.linalg.eig(A)[0]))*1.01) # divide by 1.01 to make sure it's stable
'''  

# define the observer based compensator (LQR + LQE)

# define the cost function
Q = np.eye(len(lti_plant_approx.state_labels)) *0  # don't want to penalize the transition or prediction states, only the basin depths


for asset_index in range(len(basin_max_depths)):
    # bryson's rule based on the maximum depth of each basin
    Q[lti_plant_approx.state_labels.index(lti_plant_approx.output_labels[asset_index]),lti_plant_approx.state_labels.index(lti_plant_approx.output_labels[asset_index])] = 1 / ((basin_max_depths[asset_index])**2 )
    # use below code if you want to weight some basins more than others
    if asset_index == 4: # node 5
        Q[lti_plant_approx.state_labels.index(lti_plant_approx.output_labels[asset_index])
            ,lti_plant_approx.state_labels.index(lti_plant_approx.output_labels[asset_index])] = Q[lti_plant_approx.state_labels.index(lti_plant_approx.output_labels[asset_index])
            ,lti_plant_approx.state_labels.index(lti_plant_approx.output_labels[asset_index])]*1
    if asset_index == 8: # node 9
        Q[lti_plant_approx.state_labels.index(lti_plant_approx.output_labels[asset_index])
            ,lti_plant_approx.state_labels.index(lti_plant_approx.output_labels[asset_index])] = Q[lti_plant_approx.state_labels.index(lti_plant_approx.output_labels[asset_index])
            ,lti_plant_approx.state_labels.index(lti_plant_approx.output_labels[asset_index])]*1

        


R = np.eye(len(lti_plant_approx.input_labels)) / (3.9**2) # bryson's rule based on the maximum flow of each valve

state_weighting = 0.2 # weight of state penalties (flooding) vs input penalties (flows)
# this weighting is necessary because control violations are transient while state violations are persistent

Q = Q * state_weighting 


# if the feedback gain is already calculated, we can load it from a file

print("calculating feedback gain")
# find the state feedback gain for the linear quadratic regulator
#print("defining controller")
# need to use discrete time LQR - https://python-control.readthedocs.io/en/latest/generated/control.dlqr.html#control.dlqr
K,S,E = ct.dlqr(A,B,Q,R) 

feedback_poles = E

# convert K to a pandas dataframe with the appropriate labels
K = pd.DataFrame(K,columns=lti_plant_approx.state_labels,index=lti_plant_approx.input_labels)
# save K to a csv file
K.to_csv("C:/teamwork_without_talking/feedback_gain_K.csv")
# convert K back to a numpy array
K = K.to_numpy()

# reference tracking
# abbreviate just for legibility of the reference gain calculation


# the precompensation gain calculation has a singular matrix (either A-BK or the entire product).
# this is likely because of the prediction shift matrices. We'll just specify a negative flow bias directly which will have very similar impact to using the Gr bias
# reference commands
#G = np.linalg.inv( C @ np.linalg.inv(-A + B@K) @ B )    
  
# define the reference command as emptying the basins
#r = 0.0*np.array(basin_max_depths).reshape(-1,1) # desire less than zero depth to more aggressively empty the basins

# if any of the basins are flooding, making the reference command slightly negative will help to empty the basins more aggressively

#precompensation = G@r
precompensation = -0.1*flow_threshold.reshape(-1,1) # negative flow bias will create a fixed depth in each basin


#print(precompensation) # this is in cfs

# define the observer gain

print("calculating observer gain")
# find the observer gain for the linear quadratic estimator
measurement_noise = 1
process_noise = 200
'''
L, S, E = ct.dlqr(np.transpose(lti_plant_approx.A),
                    np.transpose(lti_plant_approx.C),
                    process_noise*np.eye(len(lti_plant_approx.state_labels)),
                    measurement_noise*np.eye(len(lti_plant_approx.output_labels)))
L = np.transpose(L)
'''
L,S,E = ct.dlqe(A, np.eye(len(lti_plant_approx.state_labels)), C, 
    process_noise*np.eye(len(lti_plant_approx.state_labels)), 
    measurement_noise*np.eye(len(lti_plant_approx.output_labels)) )

observer_poles = E
#print(observer_poles)

# convert L to a pandas dataframe with the appropriate labels
L = pd.DataFrame(L,columns=lti_plant_approx.output_labels,index=lti_plant_approx.state_labels)
# save L to a csv file
L.to_csv("C:/teamwork_without_talking/observer_gain_L.csv")
# convert L back to a numpy array
L = L.to_numpy()


A = lti_plant_approx.A.astype(float)
B = lti_plant_approx.B.astype(float)
C = lti_plant_approx.C.astype(float)
L = L.astype(float)
K = K.astype(float)


# is the observer internally stable?  (are the poles of the observer within the unit circle?)

feedback_poles = np.linalg.eig(A - B@K)[0]
observer_poles = np.linalg.eig(A - L@C)[0]
open_loop_poles = np.linalg.eig(A)[0]
obc_dynamics = A - B@K - L@C
obc_poles = np.linalg.eig(A - B@K - L@C)[0] 
   
# convert control command (flow) into orifice open percentage
# per the EPA-SWMM user manual volume ii hydraulics, orifices (section 6.2, page 107) - https://nepis.epa.gov/Exe/ZyPDF.cgi/P100S9AS.PDF?Dockey=P100S9AS.PDF 
# all orifices in gamma are "bottom"
Cd = 0.65 # same for all valves
Ao = 1 # area is one square foot
g = 32.2 # ft / s^2
# the expression for discharge is found using Torricelli's equation: Q = Cd * (Ao*open_pct) sqrt(2*g*H_e)
# H_e is the effective head in meters, which is just the depth in the basin as the orifices are "bottom"
# to get the action command as a percent open, we solve as: open_pct = Q_desired / (Cd * Ao * sqrt(2*g*H_e))

env = pystorms.scenarios.gamma()
env.env.sim = pyswmm.simulation.Simulation(r"C:\\teamwork_without_talking\\gamma.inp")
# if you want a shorter timeframe than the entire summer so you can debug the controller
env.env.sim.start_time = datetime.datetime(2020,6,10,12,0)
env.env.sim.end_time = datetime.datetime(2020,7,10,12,0) # one month with some good storms
env.env.sim.start()
done = False

# edit the visible states and action space
# controlled and observed basins will be: 1, 4, 6, 7, 8, and 10
env.config['action_space'] = [env.config['action_space'][i] for i in [0,3,5,6,7,9]]
env.config['states'] = [env.config['states'][i] for i in [0,3,5,6,7,9]]

print(env.config['action_space'])
print(env.config['states'])

# grab the rain gages
# could define them as states through pystorms, but this will be easier
for raingage in pyswmm.RainGages(env.env.sim):
    if raingage.raingageid == 'J':
        raingage_J = raingage
    elif raingage.raingageid == 'C':
        raingage_C = raingage
    elif raingage.raingageid == 'N':
        raingage_N = raingage
    elif raingage.raingageid == 'S':
        raingage_S = raingage
    else: 
        pass
    
raingage_J_data = pd.read_csv("C:/teamwork_without_talking/raingage_J_summer_2020.dat",index_col=0,parse_dates=True,sep='\t')
raingage_C_data = pd.read_csv("C:/teamwork_without_talking/raingage_C_summer_2020.dat",index_col=0,parse_dates=True,sep='\t')
raingage_N_data = pd.read_csv("C:/teamwork_without_talking/raingage_N_summer_2020.dat",index_col=0,parse_dates=True,sep='\t')
raingage_S_data = pd.read_csv("C:/teamwork_without_talking/raingage_S_summer_2020.dat",index_col=0,parse_dates=True,sep='\t')

u = -1.0*np.ones((len(env.config['action_space']),1)) # begin all open
u_open_pct = np.ones((len(env.config['action_space']),1)) # begin all open
xhat = np.zeros((len(lti_plant_approx.state_labels),1)) # initial state estimate for modpods

xhat = np.ones((len(lti_plant_approx.state_labels),1))*0.01

last_eval = env.env.sim.start_time - datetime.timedelta(days=1) # init to a time before the simulation starts

while not done:

    # take control actions?
    if env.env.sim.current_time.minute % 5 == 0 and (env.env.sim.current_time > last_eval + datetime.timedelta(minutes=2)):
        last_eval = env.env.sim.current_time

        if control_scenario == 'centralized':
            # update the rainfall predictions
            # find the index of the current time in the datetime index of raingage_J_data
            current_idx = raingage_J_data.index[raingage_J_data.index >= env.env.sim.current_time]            



            #y_measured = env.state().reshape(-1,1)
            y_measured = env.state()
            # append to y_measured the rainfall at the four rain gages (the order is J, C, N, and S)
            # append the "rainfall" attribute of raingage_J to y_measured
            y_measured = np.r_[y_measured,raingage_J.rainfall]
            y_measured = np.r_[y_measured,raingage_C.rainfall]
            y_measured = np.r_[y_measured,raingage_N.rainfall]
            y_measured = np.r_[y_measured,raingage_S.rainfall]
            y_measured = y_measured.reshape(-1,1)

             # for updating the plant, calculate the "u" that is actually applied to the plant, not the desired control input
            for idx in range(len(u)):
                u[idx,0] = Cd*Ao*u_open_pct[idx,0]*np.sqrt(2*g*env.state()[idx]) # calculate the actual flow through the orifice
                
            # update the observer based on these measurements -> xhat_dot = A xhat + B u + L (y_m - C xhat)
            state_evolution = A @ xhat 
            impact_of_control = B @ u 
            yhat = C @ xhat # just for reference, could be useful for plotting later
            y_error =  y_measured - yhat # cast observables to be 2 dimensional
            output_updating = L @ y_error 
            xhat_tp1 =  state_evolution + impact_of_control + output_updating # xhat_dot = A xhat + B u + L (y_m - C xhat)
            yhat_tp1 = C @ xhat_tp1
            
            xhat = xhat_tp1 # try dividing the plant approximation instead

            u = -K @ xhat 
            u = u + precompensation 

            u_open_pct = u*-1
    
            for idx in range(len(u)):
                head = 2*g*env.state()[idx]
                if head == 0:
                    head = 10e-6 # avoid divide by zero errors
        
                u_open_pct[idx,0] = u[idx,0] / (Cd*Ao * np.sqrt(head)) # open percentage for desired flow rate
        
                if u_open_pct[idx,0] > 1: # if the calculated open percentage is greater than 1, the orifice is fully open
                    u_open_pct[idx,0] = 1
                elif u_open_pct[idx,0]< 0: # if the calculated open percentage is less than 0, the orifice is fully closed
                    u_open_pct[idx,0] = 0
            # cast u_open_pct to be a numpy array
            u_open_pct = np.array(u_open_pct)

            if verbose and env.env.sim.current_time.minute == 0 and env.env.sim.current_time.hour % 2 == 0:
                # make u_open_pct_print by appending four nan values to u_open_pct
                u_open_pct_print = np.r_[u_open_pct,np.nan*np.ones((4,1))]
                print("output_labels , u_open_pct , yhat , y_measured , y_error")
                #print(np.c_[u_open_pct_print,yhat, y_measured, y_error])
                # format numpy outputs to be scientific with 3 decimal places
                np.set_printoptions(precision=3,suppress=True)
                print(np.c_[lti_plant_approx.output_labels,u_open_pct_print,yhat, y_measured, y_error])
                print("current time, end time")
                print(env.env.sim.current_time, env.env.sim.end_time)
                print("\n")  

        else:
            pass
                 
    done = env.step(u_open_pct.flatten())