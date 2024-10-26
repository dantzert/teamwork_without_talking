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

# set the random seed for replicability
np.random.seed(0)

# options are: 'centralized', 'hi-fi', 'lo-fi', and 'local' ('uncontrolled')
control_scenario = 'hi-fi' 
year = '2021' # '2020' or '2021'
verbose = True

print("evaluating ", control_scenario)

# project file is in english units
cfs2cms = 35.315
ft2meters = 3.281
basin_max_depths =  [10.0, 10.0, 20.0, 10.0, 10.0, 13.72] # feet
flow_threshold_value = 0.5 # cfs
flow_threshold = np.ones(6)*flow_threshold_value # cfs

if control_scenario == 'lo-fi':
    with open("C:/teamwork_without_talking/plant_approx_discrete_w_predictions_lofi.pickle", 'rb') as handle:
        lti_plant_approx = pickle.load(handle)
else:
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
    if asset_index == 1: # basin 4
        Q[lti_plant_approx.state_labels.index(lti_plant_approx.output_labels[asset_index])
            ,lti_plant_approx.state_labels.index(lti_plant_approx.output_labels[asset_index])] = Q[lti_plant_approx.state_labels.index(lti_plant_approx.output_labels[asset_index])
            ,lti_plant_approx.state_labels.index(lti_plant_approx.output_labels[asset_index])]*2
    if asset_index == 3: # basin 7
        Q[lti_plant_approx.state_labels.index(lti_plant_approx.output_labels[asset_index])
            ,lti_plant_approx.state_labels.index(lti_plant_approx.output_labels[asset_index])] = Q[lti_plant_approx.state_labels.index(lti_plant_approx.output_labels[asset_index])
            ,lti_plant_approx.state_labels.index(lti_plant_approx.output_labels[asset_index])]*0.005

        


R = np.eye(len(lti_plant_approx.input_labels)) / (flow_threshold_value**2) # bryson's rule based on the maximum flow of each valve
# penalize flows out of O1 more (that's the outlet)
# R[0,0] = R[0,0]*10

state_weighting = 3.0 # weight of state penalties (flooding) vs input penalties (flows)
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
precompensation = -0.2*flow_threshold.reshape(-1,1) # negative flow bias will create a fixed depth in each basin


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

if control_scenario == 'centralized' or control_scenario == 'uncontrolled':
    outage_lengths = [0.0]
else:
    #packet_loss_chances = [0.0,0.2,0.5,0.8,0.9,0.95,0.98,0.99,0.999,0.9993,0.9995,0.9997,0.9999] # as originally submitted
    #packet_loss_chances = [0.0,0.2,0.5,0.8,0.9,0.95,0.98,0.99,0.999,0.9993,0.9995,0.9997,0.9999, 1.0] # with "never"
    # make outage_lengths an array of timedeltas ranging from 5 minutes to 4 months, all divisible by 5 minutes, log spaced
    outage_lengths = [datetime.timedelta(minutes=5),datetime.timedelta(minutes=10),datetime.timedelta(minutes=15),
                      datetime.timedelta(minutes=30),datetime.timedelta(hours=1),datetime.timedelta(hours=2),
                      datetime.timedelta(hours=4),datetime.timedelta(hours=8),datetime.timedelta(days=1),
                      datetime.timedelta(days=2),datetime.timedelta(days=3),datetime.timedelta(days=7),
                      datetime.timedelta(days=14),datetime.timedelta(days=30),datetime.timedelta(days=60),
                      datetime.timedelta(days=90),datetime.timedelta(days=120)]
    outage_lengths = outage_lengths[-3:] # just for testing


for outage_length in outage_lengths:
    print("\nevaluating outage length: ", outage_length)

    env = pystorms.scenarios.gamma()
    env.env.sim = pyswmm.simulation.Simulation(r"C:\\teamwork_without_talking\\gamma.inp") # 2.0.0 release raises a spurious multi-sim error. revert to 1.5.1
    # if you want a shorter timeframe than the entire summer so you can debug the controller
    env.env.sim.start_time = datetime.datetime(2021,6,10,0,0)
    #env.env.sim.end_time = datetime.datetime(2020,5,20,0,0) 
    #env.env.sim.end_time = datetime.datetime(2020,6,15,0,0) 
    env.env.sim.start()
    done = False

    # edit the visible states and action space
    # controlled and observed basins will be: 1, 4, 6, 7, 8, and 10
    env.config['action_space'] = [env.config['action_space'][i] for i in [0,3,5,6,7,9]]
    env.config['states'] = [env.config['states'][i] for i in [0,3,5,6,7,9]]

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
    
    raingage_J_data = pd.read_csv(str("C:/teamwork_without_talking/raingage_J_summer_"+str(year) + ".dat"),index_col=0,parse_dates=True,sep='\t')
    raingage_C_data = pd.read_csv(str("C:/teamwork_without_talking/raingage_C_summer_"+str(year) + ".dat"),index_col=0,parse_dates=True,sep='\t')
    raingage_N_data = pd.read_csv(str("C:/teamwork_without_talking/raingage_N_summer_"+str(year) + ".dat"),index_col=0,parse_dates=True,sep='\t')
    raingage_S_data = pd.read_csv(str("C:/teamwork_without_talking/raingage_S_summer_"+str(year) + ".dat"),index_col=0,parse_dates=True,sep='\t')

    u = -1.0*np.ones((len(env.config['action_space']),1)) # begin all shut
    u_open_pct = 0.0*np.ones((len(env.config['action_space']),1)) # begin all shut
    xhat = np.zeros((len(lti_plant_approx.state_labels),1)) # initial state estimate for modpods

    last_eval = env.env.sim.start_time - datetime.timedelta(days=1) # init to a time before the simulation starts

    if control_scenario == 'hi-fi' or control_scenario == 'lo-fi':
        # initialize the distributed system estimates 
        system_estimates = {'server':xhat.copy(), '1':xhat.copy(), '4':xhat.copy(), '6':xhat.copy(), 
                            '7':xhat.copy(), '8':xhat.copy(), '10':xhat.copy()}
    elif control_scenario == 'local':
        system_estimates = {'server':xhat.copy()} # microcontrollers don't maintain system estimates
                

    while not done:
        if control_scenario == 'uncontrolled':
            # make u_open_pct all ones (fully open) for the uncontrolled scenario
            u_open_pct = np.ones((len(env.config['action_space']),1))
            total_TSS_loading = pyswmm.Links(env.env.sim)["O1"].total_loading['TSS']
            done = env.step(u_open_pct.flatten())
            continue

        # take control actions?
        if env.env.sim.current_time.minute % 5 == 0 and (env.env.sim.current_time > last_eval + datetime.timedelta(minutes=2)):
            last_eval = env.env.sim.current_time

            # find indices for updating rainfall predictions
            try:
                # find the index of the current time in the datetime index of raingage_J_data
                current_rain_idx = raingage_J_data.index[raingage_J_data.index >= env.env.sim.current_time][0]    
                # what are the indices in state_labels corresponding the the raingage_J prediction states?
                J_pred_start = lti_plant_approx.state_labels.index('raingage_J_0H5M')
                J_pred_end = lti_plant_approx.state_labels.index('raingage_J_06H00M')  # pred_start is a larger index than pred_end
                # do the same for the other raingages
                C_pred_start = lti_plant_approx.state_labels.index('raingage_C_0H5M')
                C_pred_end = lti_plant_approx.state_labels.index('raingage_C_06H00M')  # pred_start is a larger index than pred_end
                N_pred_start = lti_plant_approx.state_labels.index('raingage_N_0H5M')
                N_pred_end = lti_plant_approx.state_labels.index('raingage_N_06H00M')  # pred_start is a larger index than pred_end
                S_pred_start = lti_plant_approx.state_labels.index('raingage_S_0H5M')
                S_pred_end = lti_plant_approx.state_labels.index('raingage_S_06H00M')  # pred_start is a larger index than pred_end
            except:
                pass


            if control_scenario == 'centralized':
                # update the rainfall predictions
                try:
                    xhat[J_pred_end:J_pred_start+1] = np.array(raingage_J_data.loc[current_rain_idx+datetime.timedelta(minutes=5):current_rain_idx +datetime.timedelta(minutes=5*72)].values)[::-1].reshape(-1,1) # reverese it
                    xhat[C_pred_end:C_pred_start+1] = np.array(raingage_C_data.loc[current_rain_idx+datetime.timedelta(minutes=5):current_rain_idx +datetime.timedelta(minutes=5*72)].values)[::-1].reshape(-1,1) # reverese it
                    xhat[N_pred_end:N_pred_start+1] = np.array(raingage_N_data.loc[current_rain_idx+datetime.timedelta(minutes=5):current_rain_idx +datetime.timedelta(minutes=5*72)].values)[::-1].reshape(-1,1) # reverese it
                    xhat[S_pred_end:S_pred_start+1] = np.array(raingage_S_data.loc[current_rain_idx+datetime.timedelta(minutes=5):current_rain_idx +datetime.timedelta(minutes=5*72)].values)[::-1].reshape(-1,1) # reverese it
                except:
                    pass # near the end of the precip record, already have all the data

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
                
                # update the observer based on these measurements -> xhat_tp1 = A xhat + B u + L (y_m - C xhat)
                state_evolution = A @ xhat 
                impact_of_control = B @ u 
                yhat = C @ xhat # just for reference, could be useful for plotting later
                y_error =  y_measured - yhat # cast observables to be 2 dimensional
                output_updating = L @ y_error 
                xhat_tp1 =  state_evolution + impact_of_control + output_updating 
                yhat_tp1 = C @ xhat_tp1
            
                xhat = xhat_tp1 # xhat_tp1 is the state estimate for this time step. xhat is the system estimate we came into this timestep with (from last timestep)

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
            
                total_TSS_loading = pyswmm.Links(env.env.sim)["O1"].total_loading['TSS']

                if verbose and env.env.sim.current_time.minute == 0 and env.env.sim.current_time.hour == 0:
                    # make u_open_pct_print by appending four nan values to u_open_pct
                    u_open_pct_print = np.r_[u_open_pct,np.nan*np.ones((4,1))]
                    print("output_labels , u_open_pct , yhat , y_measured , y_error")
                    #print(np.c_[u_open_pct_print,yhat, y_measured, y_error])
                    # format numpy outputs to be scientific with 3 decimal places
                    np.set_printoptions(precision=3,suppress=True)
                    #print(np.c_[lti_plant_approx.output_labels,u_open_pct_print ,yhat, y_measured, y_error])
                    # do the above, but print no more than three decimal places for any of those arrays
                    print(np.c_[lti_plant_approx.output_labels,np.round(u_open_pct_print,3) ,np.round(yhat,3), np.round(y_measured,3), np.round(y_error,3)])
                    print("current time, end time")
                    print(env.env.sim.current_time, env.env.sim.end_time)
                    print("\n")  

            elif control_scenario == 'hi-fi' or control_scenario == 'lo-fi':
                # which control points report this timestep?
                # define a dictionary whose keys are the states and raingages, their values will be booleans indicating whether they report this timestep
                reported = {'1':True,'4':True,'6':True,'7':True, '8':True,'10':True}
                for key in reported.keys():
                    # if the date is in between june 17th, noon, and june 17th, noon + outage_length, set reported[key] to False
                    if (env.env.sim.current_time >= datetime.datetime(2021,6,17,12,0) and env.env.sim.current_time <= datetime.datetime(2021,6,17,12,0) + outage_length):
                        reported[key] = False
                    else:
                        reported[key] = True
                        
                # the beginning looks very similar to the centralized case as we do the server-side system estimate
                # the only difference is that we'll be missing some depth measurements from the control points
                # update the rainfall predictions for the server state estimate
                try:
                    system_estimates['server'][J_pred_end:J_pred_start+1] = np.array(raingage_J_data.loc[current_rain_idx+datetime.timedelta(minutes=5):current_rain_idx +datetime.timedelta(minutes=5*72)].values)[::-1].reshape(-1,1) # reverese it
                    system_estimates['server'][C_pred_end:C_pred_start+1] = np.array(raingage_C_data.loc[current_rain_idx+datetime.timedelta(minutes=5):current_rain_idx +datetime.timedelta(minutes=5*72)].values)[::-1].reshape(-1,1) # reverese it
                    system_estimates['server'][N_pred_end:N_pred_start+1] = np.array(raingage_N_data.loc[current_rain_idx+datetime.timedelta(minutes=5):current_rain_idx +datetime.timedelta(minutes=5*72)].values)[::-1].reshape(-1,1) # reverese it
                    system_estimates['server'][S_pred_end:S_pred_start+1] = np.array(raingage_S_data.loc[current_rain_idx+datetime.timedelta(minutes=5):current_rain_idx +datetime.timedelta(minutes=5*72)].values)[::-1].reshape(-1,1) # reverese it
                except:
                    pass # near the end of the precip record, already have all the data
                
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
                    if (idx == 0 and reported['1']) or (idx == 1 and reported['4']) or (idx == 2 and reported['6']) or (idx == 3 and reported['7']) or (idx == 4 and reported['8']) or (idx == 5 and reported['10']):
                        u[idx,0] = Cd*Ao*u_open_pct[idx,0]*np.sqrt(2*g*env.state()[idx]) # calculate the actual flow through the orifice

                # update the observer based on these measurements -> xhat_tp1 = A xhat + B u + L (y_m - C xhat)
                state_evolution = A @ system_estimates['server']
                impact_of_control = B @ u 
                yhat = C @ system_estimates['server'] # just for reference, could be useful for plotting later
                y_error =  y_measured - yhat # cast observables to be 2 dimensional
                # make the rows of y_error corresponding to the unreporting control points zero
                for idx in range(len(lti_plant_approx.output_labels)):
                    if (idx == 0 and not reported['1']) or (idx == 1 and not reported['4']) or (idx == 2 and not reported['6']) or (idx == 3 and not reported['7']) or (idx == 4 and not reported['8']) or (idx == 5 and not reported['10']):
                        y_error[idx,0] = 0
                # this implies we're not using the information in our system update 
                output_updating = L @ y_error 
                xhat_tp1 =  state_evolution + impact_of_control + output_updating 
                yhat_tp1 = C @ xhat_tp1
            
                system_estimates['server'] = xhat_tp1 # xhat_tp1 is the state estimate for this time step. xhat is the system estimate we came into this timestep with (from last timestep)

                # server just generates a state estimate. doesn't make any feedback control decisions

                # now we'll update the control points and make feedback control decisions

                for idx, key in enumerate(reported):
                    if reported[key]: # if the control point reported this timestep, give it the server system estimate
                        system_estimates[key] = system_estimates['server'].copy()
                    else: # if the control point didn't report this timestep, it will use its own system estimate from last timestep and the local measurement
                        # calculate the flow through our orifice (for all others we'll assume what "should have" happened, happened because we're not observing it)
                        u[idx,0] = Cd*Ao*u_open_pct[idx,0]*np.sqrt(2*g*env.state()[idx])

                        state_evolution = A @ system_estimates[key]
                        impact_of_control = B @ u 
                        yhat = C @ system_estimates[key] 
                        y_error =  y_measured - yhat
                        # the only y_error available is our local measurement. set all other entries to zero
                        # set all rows except "idx" of y_error to zero
                        for j in range(len(lti_plant_approx.output_labels)):
                            if j != idx:
                                y_error[j,0] = 0
                        output_updating = L @ y_error
                        xhat_tp1 =  state_evolution + impact_of_control + output_updating 
                        yhat_tp1 = C @ xhat_tp1
            
                        system_estimates[key] = xhat_tp1 # xhat_tp1 is the state estimate for this time step. xhat is the system estimate we came into this timestep with (from last timestep)
                    
                    # now caluclate the control action for this control point
                    u = -K @ system_estimates[key] # our idea of what the whole system should be doing
                    u = u + precompensation
                    # we may not have access to other depths, but we know flows won't be negative
                    for j in range(len(u)):
                        if u[j,0] < 0:
                            u[j,0] = 0

                    # we only need to worry about calculating the u_open_pct entry corresponding to the control point we're updating
                    head = 2*g*env.state()[idx] # our local depth measurement, always available
                    if head == 0:
                        head = 10e-6
                    u_open_pct[idx,0] = u[idx,0] / (Cd*Ao * np.sqrt(head)) # open percentage for desired flow rate
                    if u_open_pct[idx,0] > 1: # if the calculated open percentage is greater than 1, the orifice is fully open
                        u_open_pct[idx,0] = 1
                    elif u_open_pct[idx,0]< 0: # if the calculated open percentage is less than 0, the orifice is fully closed
                        u_open_pct[idx,0] = 0

                # cast u_open_pct to be a numpy array
                u_open_pct = np.array(u_open_pct)
                total_TSS_loading = pyswmm.Links(env.env.sim)["O1"].total_loading['TSS']
                        
                if verbose and env.env.sim.current_time.minute == 0 and env.env.sim.current_time.hour == 0 and env.env.sim.current_time.day % 10 == 0:
                    # make a dataframe with columns: value, server, 1, 4, 6, 7, 8, 10 and rows: the output labels
                    # fill the entries with the system estimates for each control point
                    estimates_to_print = pd.DataFrame(columns = ['truth','server','1','4','6','7','8','10'],index=lti_plant_approx.output_labels)
                    for key in system_estimates.keys():
                        estimates_to_print[key] = (C@system_estimates[key])
                    estimates_to_print['truth'] = y_measured
                    # print all columns of pandas dataframes
                    pd.set_option('display.max_columns', None)
                    # print to 2 decimal places
                    pd.set_option('display.float_format', lambda x: '%.2f' % x)

                    print(estimates_to_print)
                    '''
                    # print out the different estimates for y side by side
                    print("value            | server   , 1 , 4 , 6 , 7 , 8 , 10")
                    print(np.c_[lti_plant_approx.output_labels, np.round(C@system_estimates['server'],2),np.round(C@system_estimates['1'],2),
                                np.round(C@system_estimates['4'],2),
                                np.round(C@system_estimates['6'],2),np.round(C@system_estimates['7'],2),
                                np.round(C@system_estimates['8'],2),
                                np.round(C@system_estimates['10'],2)])
                    '''
                    print("current time, end time")
                    print(env.env.sim.current_time, env.env.sim.end_time)
                    print("\n")

            elif control_scenario == 'local':
                # only the server maintains a system estimate. the other assets will do a zero-order hold if below 25% full and activate a level controller otherwise
                # which control points report this timestep?
                # define a dictionary whose keys are the states and raingages, their values will be booleans indicating whether they report this timestep
                reported = {'1':True,'4':True,'6':True,'7':True, '8':True,'10':True}
                for key in reported.keys():
                    # if the date is in between june 17th, noon, and june 17th, noon + outage_length, set reported[key] to False
                    if (env.env.sim.current_time >= datetime.datetime(2021,6,17,12,0) and env.env.sim.current_time <= datetime.datetime(2021,6,17,12,0) + outage_length):
                        reported[key] = False
                    else:
                        reported[key] = True
                        
                # the beginning looks very similar to the centralized case as we do the server-side system estimate
                # the only difference is that we'll be missing some depth measurements from the control points
                # update the rainfall predictions for the server state estimate
                try:
                    system_estimates['server'][J_pred_end:J_pred_start+1] = np.array(raingage_J_data.loc[current_rain_idx+datetime.timedelta(minutes=5):current_rain_idx +datetime.timedelta(minutes=5*72)].values)[::-1].reshape(-1,1) # reverese it
                    system_estimates['server'][C_pred_end:C_pred_start+1] = np.array(raingage_C_data.loc[current_rain_idx+datetime.timedelta(minutes=5):current_rain_idx +datetime.timedelta(minutes=5*72)].values)[::-1].reshape(-1,1) # reverese it
                    system_estimates['server'][N_pred_end:N_pred_start+1] = np.array(raingage_N_data.loc[current_rain_idx+datetime.timedelta(minutes=5):current_rain_idx +datetime.timedelta(minutes=5*72)].values)[::-1].reshape(-1,1) # reverese it
                    system_estimates['server'][S_pred_end:S_pred_start+1] = np.array(raingage_S_data.loc[current_rain_idx+datetime.timedelta(minutes=5):current_rain_idx +datetime.timedelta(minutes=5*72)].values)[::-1].reshape(-1,1) # reverese it
                except:
                    pass # near the end of the precip record, already have all the data
                
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
                    if (idx == 0 and reported['1']) or (idx == 1 and reported['4']) or (idx == 2 and reported['6']) or (idx == 3 and reported['7']) or (idx == 4 and reported['8']) or (idx == 5 and reported['10']):
                        u[idx,0] = Cd*Ao*u_open_pct[idx,0]*np.sqrt(2*g*env.state()[idx]) # calculate the actual flow through the orifice

                # update the observer based on these measurements -> xhat_tp1 = A xhat + B u + L (y_m - C xhat)
                state_evolution = A @ system_estimates['server']
                impact_of_control = B @ u 
                yhat = C @ system_estimates['server'] # just for reference, could be useful for plotting later
                y_error =  y_measured - yhat # cast observables to be 2 dimensional
                # make the rows of y_error corresponding to the unreporting control points zero
                for idx in range(len(lti_plant_approx.output_labels)):
                    if (idx == 0 and not reported['1']) or (idx == 1 and not reported['4']) or (idx == 2 and not reported['6']) or (idx == 3 and not reported['7']) or (idx == 4 and not reported['8']) or (idx == 5 and not reported['10']):
                        y_error[idx,0] = 0
                # this implies we're not using the information in our system update 
                output_updating = L @ y_error 
                xhat_tp1 =  state_evolution + impact_of_control + output_updating 
                yhat_tp1 = C @ xhat_tp1
            
                system_estimates['server'] = xhat_tp1 # xhat_tp1 is the state estimate for this time step. xhat is the system estimate we came into this timestep with (from last timestep)
                
                # server just generates a state estimate. doesn't make any feedback control decisions

                # now we'll update the control points and make feedback control decisions
                filling_degree_threshold = 0.5
                minimum_percent_open_local = 0.02

                for idx, key in enumerate(reported):
                    if reported[key]: # if the control point reported this timestep, do what the server says
                        u = -K @ system_estimates['server'] # our idea of what the whole system should be doing
                        u = u + precompensation
                        # we may not have access to other depths, but we know flows won't be negative
                        for j in range(len(u)):
                            if u[j,0] < 0:
                                u[j,0] = 0
                        # we only need to worry about calculating the u_open_pct entry corresponding to the control point we're updating
                        head = 2*g*env.state()[idx] # our local depth measurement, always available
                        if head == 0:
                            head = 10e-6
                        u_open_pct[idx,0] = u[idx,0] / (Cd*Ao * np.sqrt(head)) # open percentage for desired flow rate
                        if u_open_pct[idx,0] > 1: # if the calculated open percentage is greater than 1, the orifice is fully open
                            u_open_pct[idx,0] = 1
                        elif u_open_pct[idx,0] < 0: # if the calculated open percentage is less than 0, the orifice is fully closed
                            u_open_pct[idx,0] = 0

                    elif not reported[key] and env.state()[idx] / basin_max_depths[idx] < filling_degree_threshold: # if the control point didn't report this timestep and is less than X% full
                        # don't change u_open_pct from the last time step  - zero order hold on open percentage (not discharge)
                        if u_open_pct[idx,0] < minimum_percent_open_local: # if zero order hold would result in being completely shut
                            u_open_pct[idx,0] = minimum_percent_open_local # always allow some slow draining
                        else:
                            pass
                    elif not reported[key] and env.state()[idx] / basin_max_depths[idx] >= filling_degree_threshold: # we didn't connect and we're more than X full
                        u_open_pct[idx,0] = minimum_percent_open_local +  (env.state()[idx] / basin_max_depths[idx] - filling_degree_threshold) / (1-filling_degree_threshold) 
                        # level controller which is nearly shut at X% full and fully open at 100% full
                    else:
                        print("error in control logic. should never reach this point")

                # cast u_open_pct to be a numpy array
                u_open_pct = np.array(u_open_pct)
                total_TSS_loading = pyswmm.Links(env.env.sim)["O1"].total_loading['TSS']
                        
                if verbose and env.env.sim.current_time.minute == 0 and env.env.sim.current_time.hour == 0 and env.env.sim.current_time.day % 10 == 0:
                    # make u_open_pct_print by appending four nan values to u_open_pct
                    u_open_pct_print = np.r_[u_open_pct,np.nan*np.ones((4,1))]
                    print("output_labels , u_open_pct , yhat , y_measured , y_error")
                    #print(np.c_[u_open_pct_print,yhat, y_measured, y_error])
                    # format numpy outputs to be scientific with 3 decimal places
                    np.set_printoptions(precision=3,suppress=True)
                    #print(np.c_[lti_plant_approx.output_labels,u_open_pct_print ,yhat, y_measured, y_error])
                    # do the above, but print no more than three decimal places for any of those arrays
                    print(np.c_[lti_plant_approx.output_labels,np.round(u_open_pct_print,3) ,np.round(yhat,3), np.round(y_measured,3), np.round(y_error,3)])
                    print("current time, end time")
                    print(env.env.sim.current_time, env.env.sim.end_time)
                    print("\n")  

            else:
                pass
                 
        done = env.step(u_open_pct.flatten())
    

    # only penalize flow and flood at controlled assets
    valves = ["O1","O4","O6","O7","O8","O10"]
    storage_nodes = ["1","4","6","7","8","10"]

    flood_cost = 0
    for key,value in env.data_log['flooding'].items():    
        if sum(value) > 0 and key in storage_nodes:
            # the total cost is the number of positive entries in value multiplied by the cost of flooding (10e6)
            # find the number of positive entries in value
            num_positive = sum([1 for x in value if x > 0])
            print(key,"{:.4e}".format(num_positive*(10**1)))
            flood_cost += num_positive*(10**1)

    flow_cost = 0
    for key,value in env.data_log['flow'].items():
        if key in valves:
            for flow in value:
                if flow > flow_threshold_value:
                    flow_cost += (flow-flow_threshold_value)**2

    print("flood cost: {:.4e}".format(flood_cost))
    print("flow cost: {:.4e}".format(flow_cost))
    print("TSS loading (kg): {:.4e}".format(total_TSS_loading/2.2))
    print("total cost: {:.4e}".format(flood_cost + flow_cost + (total_TSS_loading/2.2)*10**3))
    # save the costs to a csv
    with open(str("C:/teamwork_without_talking/results/" + control_scenario + "_synchronized_" + str(outage_length) + "_summer_" +str(year) + "_costs.csv"), 'w') as f:
        f.write("flood cost, flow cost, TSS loading (kg), total cost\n")
        f.write("{:.4e},{:.4e},{:.4e},{:.4e}\n".format(flood_cost,flow_cost,total_TSS_loading/2.2,flood_cost + flow_cost + (total_TSS_loading/2.2)*10**3))

    fig,axes = plt.subplots(6,2,figsize=(16,8))
    axes[0,0].set_title("Valves",fontsize='xx-large')
    axes[0,1].set_title("Storage Nodes",fontsize='xx-large')


    cfs2cms = 35.315
    ft2meters = 3.281
    # plot the valves
    for idx in range(6):
        axes[idx,0].plot(env.data_log['simulation_time'],np.array(env.data_log['flow'][valves[idx]])/cfs2cms,label=control_scenario,color='g',linewidth=2)
        # add a dotted red line indicating the flow threshold
        axes[idx,0].hlines(flow_threshold_value/cfs2cms, env.data_log['simulation_time'][0],env.data_log['simulation_time'][-1],label='Threshold',colors='r',linestyles='dashed',linewidth=2)
        #axes[idx,0].set_ylabel( str(  str(valves[idx]) + " Flow" ),rotation='horizontal',labelpad=8)
        axes[idx,0].annotate(str(  str(valves[idx]) + " Flow" ),xy=(0.5,0.8),xycoords='axes fraction',fontsize='xx-large')
        if idx == 0:
            axes[idx,0].legend(fontsize='xx-large')
        if idx != 5:
            axes[idx,0].set_xticks([])
        if idx == 2: # valve 6
            # write a text box indicating the costs
            axes[idx,0].text(0.3,0.75,"Flood Cost: {:.4e}".format(flood_cost),fontsize='large',horizontalalignment='center',verticalalignment='center',transform=axes[idx,0].transAxes)
            axes[idx,0].text(0.3,0.55,"Flow Cost: {:.4e}".format(flow_cost),fontsize='large',horizontalalignment='center',verticalalignment='center',transform=axes[idx,0].transAxes)
            axes[idx,0].text(0.3,0.35,"TSS Loading (kg): {:.4e}".format(total_TSS_loading/2.2),fontsize='large',horizontalalignment='center',verticalalignment='center',transform=axes[idx,0].transAxes)
            axes[idx,0].text(0.3,0.15,"Total Cost: {:.4e}".format(flood_cost + flow_cost + (total_TSS_loading/2.2)*10**3),fontsize='large',horizontalalignment='center',verticalalignment='center',transform=axes[idx,0].transAxes)
       

    # plot the storage nodes
    for idx in range(6):
        axes[idx,1].plot(env.data_log['simulation_time'],np.array(env.data_log['depthN'][storage_nodes[idx]])/ft2meters,label=control_scenario,color='g',linewidth=2)
        #axes[idx,1].set_ylabel( str( str(storage_nodes[idx]) + " Depth"),rotation='horizontal',labelpad=8)
        axes[idx,1].annotate( str( str(storage_nodes[idx]) + " Depth"),xy=(0.5,0.8),xycoords='axes fraction',fontsize='xx-large')
    
        # add a dotted red line indicating the depth threshold
        axes[idx,1].hlines(basin_max_depths[idx]/ft2meters,env.data_log['simulation_time'][0],env.data_log['simulation_time'][-1],label='Threshold',colors='r',linestyles='dashed',linewidth=2)
        if idx != 5:
            axes[idx,1].set_xticks([])

    plt.tight_layout()
    plt.savefig(str("C:/teamwork_without_talking/results/" + control_scenario + "_synchronized_" + str(outage_length) + "_summer_"+str(year) + ".png"),dpi=450)
    plt.savefig(str("C:/teamwork_without_talking/results/" + control_scenario + "_synchronized_" + str(outage_length)  + "_summer_"+str(year) + ".svg"),dpi=450)
    #plt.show(block=True)
    #plt.close('all')

    # save the data log
    with open(str("C:/teamwork_without_talking/results/" + control_scenario + "_synchronized_" + str(outage_length)  + "_summer_"+str(year) + ".pkl"), 'wb') as f:
        pickle.dump(env.data_log, f)
    # this is only the "truth" data log, the values which actually happened.
    # when you want to track teammate state inference errors, you'll need to save the system estimates as well

            
