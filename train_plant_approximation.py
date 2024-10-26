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

# to get rid of pysindy clarabel warning. comment this out if something isn't working as expected
import warnings
warnings.filterwarnings(action='ignore')

# set the numpy random seed
np.random.seed(42)

# print all columns of pandas dataframes
pd.options.display.max_columns = None

# system is in US units of feet and cubic feet per second
env = pystorms.scenarios.gamma()
env.env.sim = pyswmm.simulation.Simulation(r"C:\\teamwork_without_talking\\gamma.inp")
env.env.sim.start_time = datetime.datetime(2020,6,10,12,0)
#env.env.sim.end_time = datetime.datetime(2020,6,16,12,0)  # shorter for debugging / dev
env.env.sim.end_time = datetime.datetime(2020,7,10,12,0) # train on one month

env.env.sim.start()
done = False

actions_characterize = np.ones(len(env.config['action_space']))*0.1 # all a bit open

last_eval = env.env.sim.start_time + datetime.timedelta(days=1,hours=6) 
i = 0 # for iterating through the action space

# toricelli's equation to convert depths and open percentages to flows
Cd = 0.65 # same for both valves
Ao = 1 # area is one square foot
g = 32.2 # ft / s^2

# edit the visible states and action space
# controlled and observed basins will be: 1, 4, 6, 7, 8, and 10
env.config['action_space'] = [env.config['action_space'][i] for i in [0,3,5,6,7,9]]
env.config['states'] = [env.config['states'][i] for i in [0,3,5,6,7,9]]

print(env.config['action_space'])
print(env.config['states'])

# start all a bit open
actions_characterize = np.ones(len(env.config['action_space']))*0.1 # all a bit open

while not done:
    if env.env.sim.current_time.hour == 12 and env.env.sim.current_time.day == env.env.sim.start_time.day:
            actions_characterize = np.ones(len(env.config['action_space']))*0.0 # all closed
        
    if env.env.sim.current_time.hour % 6 == 0 and env.env.sim.current_time.minute == 0 and (env.env.sim.current_time > last_eval + datetime.timedelta(minutes=2)):
        last_eval = env.env.sim.current_time
        # select i to be a random integer amongst the action space
        i = np.random.randint(0, len(env.config['action_space']))
        actions_characterize = np.ones(len(env.config['action_space']))*0.0 # all closed
        actions_characterize[i] = np.random.rand() # except for one

    done = env.step(actions_characterize)
    

cfs2cms = 35.315
ft2meters = 3.281

basin_max_depths =  [10.0, 10.0, 20.0, 10.0, 10.0, 13.72] # feet
flow_threshold = np.ones(6)*3.9 # cfs

plots_high = max(len(env.config['action_space']) , len(env.config['states']))
fig, axes = plt.subplots(plots_high, 2, figsize=(10,1*plots_high))

axes[0,0].set_title("flows")
axes[0,1].set_title("depths")
# plot the actions
for idx in range(len(env.config['action_space'])):
    axes[idx,0].plot(env.data_log['simulation_time'], np.array(env.data_log['flow'][env.config['action_space'][idx]])/cfs2cms, label=env.config['action_space'][idx])
    axes[idx,0].axhline(y=flow_threshold[idx]/cfs2cms, color='r', linestyle='-')
    axes[idx,0].set_ylabel("flow (cms)")
    #axes[idx,0].set_xlabel("time")
    #axes[idx,0].legend()
    if idx != len(env.config['action_space']) - 1: # not the last row
        axes[idx,0].set_xticklabels([])
    axes[idx,0].annotate(str(env.config['action_space'][idx]), xy=(0.5, 0.8), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')

# plot the states
for idx in range(len(env.config['states'])):
    axes[idx,1].plot(env.data_log['simulation_time'], np.array(env.data_log['depthN'][env.config['states'][idx][0]])/ft2meters, label=env.config['states'][idx])
    axes[idx,1].axhline(y=basin_max_depths[idx] / ft2meters, color='r', linestyle='-')
    axes[idx,1].set_ylabel("depth (m)")
    #axes[idx,1].set_xlabel("time")
    #axes[idx,1].legend()
    if idx != len(env.config['states']) - 1:
        axes[idx,1].set_xticklabels([])
    axes[idx,1].annotate(str(env.config['states'][idx]), xy=(0.5, 0.8), xycoords='axes fraction', ha='center', va='center',fontsize='xx-large')

plt.tight_layout()
plt.savefig("C:/teamwork_without_talking/characterization_experiment.png",dpi=450)
plt.savefig("C:/teamwork_without_talking/characterization_experiment.svg",dpi=450)
#plt.show()
plt.close('all')


flows = pd.DataFrame.from_dict(env.data_log['flow']).iloc[:,[0,3,5,6,7,9]] # select only the [0,3,5,6,7,9] entries for the flows 
depthN = pd.DataFrame.from_dict(env.data_log['depthN']).iloc[:,[0,3,5,6,7,9]]
depthN.columns = env.config['states']
flows.columns = env.config['action_space'] # to match the naming conventions on the subway map
response = pd.concat([flows, depthN], axis=1)
response.index = env.data_log['simulation_time']

raingage_J = pd.read_csv("C:\\teamwork_without_talking\\raingage_J_summer_2020.dat", header=None, sep='\t')
# set the first columns as the index
raingage_J = raingage_J.set_index(0)
# convert the index to a datetime index
raingage_J.index = pd.to_datetime(raingage_J.index)
raingage_J.columns = ['raingage_J']
#raingage_J = raingage_J.resample('5T').asfreq()
raingage_C = pd.read_csv("C:\\teamwork_without_talking\\raingage_C_summer_2020.dat", header=None, sep='\t')
raingage_C = raingage_C.set_index(0)
raingage_C.index = pd.to_datetime(raingage_C.index)
raingage_C.columns = ['raingage_C']
#raingage_C = raingage_C.resample('5T').asfreq()
raingage_N = pd.read_csv("C:\\teamwork_without_talking\\raingage_N_summer_2020.dat", header=None, sep='\t')
raingage_N = raingage_N.set_index(0)
raingage_N.index = pd.to_datetime(raingage_N.index)
raingage_N.columns = ['raingage_N']
#raingage_N = raingage_N.resample('5T').asfreq()
raingage_S = pd.read_csv("C:\\teamwork_without_talking\\raingage_S_summer_2020.dat", header=None, sep='\t')
raingage_S = raingage_S.set_index(0)
raingage_S.index = pd.to_datetime(raingage_S.index)
raingage_S.columns = ['raingage_S']
#raingage_S = raingage_S.resample('5T').asfreq()

#print(raingage_J)

#print(response)

response = response.resample('5T').mean().copy(deep=True)

# put the raingages and response together into one dataframe
response = pd.concat([response, raingage_J, raingage_C, raingage_N, raingage_S], axis=1)
# drop any rows with na's in them
response = response.dropna()

print(response)

connectivity = modpods.topo_from_pystorms(env)


# add in the raingage connectivity
connectivity['raingage_J'] = 'n'
# J goes to 8 and 9
connectivity['raingage_J'][('8', 'depthN')] = 'd'
connectivity['raingage_C'] = 'n'
# C goes to 5, 6, and 7
connectivity['raingage_C'][('6', 'depthN')] = 'd'
connectivity['raingage_C'][('7', 'depthN')] = 'd'
connectivity['raingage_N'] = 'n'
# N goes to 10 and 11
connectivity['raingage_N'][('10', 'depthN')] = 'd'
connectivity['raingage_S'] = 'n'
# S goes to 1 and 4
connectivity['raingage_S'][('1', 'depthN')] = 'd'
connectivity['raingage_S'][('4', 'depthN')] = 'd'

# add rows for each of the raingages, but fill them with 'n'
connectivity.loc['raingage_J'] = 'n'
connectivity.loc['raingage_C'] = 'n'
connectivity.loc['raingage_N'] = 'n'
connectivity.loc['raingage_S'] = 'n'

# they need to be defined within the state variables in order to accomodate the disturbance predictions
# but we won't actaully learn any model for their dynamics as they are not caused by anything else and have no autocorrelation


print(connectivity)

# make a list called dependent_columns which is the columns of connectivity less env.config['action_space']
dependent_columns = list(connectivity.columns)
for action in env.config['action_space']:
    dependent_columns.remove(action)
    
print(dependent_columns)
# independent columns are the action space
independent_columns = env.config['action_space']


plot_approx_validation = True

if plot_approx_validation:
    ## validation - forward simulate over subset of training data

    # simulate the first couple days by specifying the rainfall forecast as initial conditions
    # and compare this to the swmm model output

    # only use the first 5 days
    response_for_sim = response.iloc[:5*24*12]
    # reindex the response to an integer step
    response_for_sim.index = np.arange(0,len(response_for_sim),1)

    # load the discrete time model we've already trained
    with open("C:/teamwork_without_talking/plant_approx_discrete_w_predictions.pickle", 'rb') as handle:
        lti_plant_approx_discrete_w_predictions = pickle.load(handle)

    # construct the initial condition X0
    # first, start with a dataframe of zeros with the same index as the state space
    X0 = pd.DataFrame(columns=lti_plant_approx_discrete_w_predictions.state_labels,index=[0])
    # now set the initial conditions for the depths to the first row of the training response
    for idx in range(len(env.config['states'])):
        X0[str(dependent_columns[idx])] = response[dependent_columns[idx]].iloc[0]

    # now initialize the raingage states
    # the initial state of 'raingage_X' will be the first entry in response
    # the initial state of 'raingage_X_00H05M' will be the second entry in response
    # the initial state of 'raingage_X_00H10M' will be the third entry in response
    # and so on up to 'raingage_X_48H00M' which will be the 72th entry in response
    # so, the initial conditions for the raingage states are the first 72 entries of response
    #raingage_J_shift_names = ['raingage_J_' + str(i) + 'H' + str(j) + 'M' for i in range(48) for j in range(0,60,5)]
    raingage_J_shift_names = ['raingage_J_' + str(i) + 'H' + str(j) + 'M' for i in range(6) for j in range(0,60,5)]
    # get rid of 0H0M and add 48H0M
    raingage_J_shift_names = raingage_J_shift_names[1:]
    #raingage_J_shift_names.append('raingage_J_48H00M')
    raingage_J_shift_names.append('raingage_J_06H00M')
    # do the same for raingage_C, N and S
    #raingage_C_shift_names = ['raingage_C_' + str(i) + 'H' + str(j) + 'M' for i in range(48) for j in range(0,60,5)]
    raingage_C_shift_names = ['raingage_C_' + str(i) + 'H' + str(j) + 'M' for i in range(6) for j in range(0,60,5)]
    raingage_C_shift_names = raingage_C_shift_names[1:]
    #raingage_C_shift_names.append('raingage_C_48H00M')
    raingage_C_shift_names.append('raingage_C_06H00M')
    #raingage_N_shift_names = ['raingage_N_' + str(i) + 'H' + str(j) + 'M' for i in range(48) for j in range(0,60,5)]
    raingage_N_shift_names = ['raingage_N_' + str(i) + 'H' + str(j) + 'M' for i in range(6) for j in range(0,60,5)]
    raingage_N_shift_names = raingage_N_shift_names[1:]
    #raingage_N_shift_names.append('raingage_N_48H00M')
    raingage_N_shift_names.append('raingage_N_06H00M')
    #raingage_S_shift_names = ['raingage_S_' + str(i) + 'H' + str(j) + 'M' for i in range(48) for j in range(0,60,5)]
    raingage_S_shift_names = ['raingage_S_' + str(i) + 'H' + str(j) + 'M' for i in range(6) for j in range(0,60,5)]
    raingage_S_shift_names = raingage_S_shift_names[1:]
    #raingage_S_shift_names.append('raingage_S_48H00M')
    raingage_S_shift_names.append('raingage_S_06H00M')
    # set the entries in X0 corresponding to raingage_J_shift names to 72 entries in response['raingage_J'] starting at the second timestep (index 1) and ending at the 73th timestep (index 72)
    print(X0.loc[0,raingage_J_shift_names])
    print(len(X0.loc[0,raingage_J_shift_names]))
    print(response_for_sim['raingage_J'].iloc[1:73]) 

    X0.loc[0,raingage_J_shift_names] = np.array(response_for_sim['raingage_J'].iloc[1:73])

    print(max(X0.loc[0,raingage_J_shift_names]))
    print(max(response_for_sim['raingage_J'].iloc[1:73]))

    X0.loc[0,raingage_C_shift_names] = np.array(response_for_sim['raingage_C'].iloc[1:73])
    X0.loc[0,raingage_N_shift_names] = np.array(response_for_sim['raingage_N'].iloc[1:73])
    X0.loc[0,raingage_S_shift_names] = np.array(response_for_sim['raingage_S'].iloc[1:73])

    # fill any na's with zero
    X0.fillna(0.0,inplace=True)   
    # convert X0 to a numpy array
    X0 = X0.to_numpy().flatten()

    approx_response = ct.forced_response(lti_plant_approx_discrete_w_predictions, T = response_for_sim.index.values, 
                                         U = np.transpose(response_for_sim[independent_columns].values), X0 = X0)


    approx_data = pd.DataFrame(index=response_for_sim.index.values)
    for idx in range(len(dependent_columns)):
        approx_data[dependent_columns[idx]] = approx_response.outputs[idx][:]

    # print the max of each column in approx_data
    print(approx_data.max())

    fig, axes = plt.subplots(len(dependent_columns), 1, figsize=(10, 10))

    for idx in range(len(dependent_columns)):
        axes[idx].plot(response_for_sim[dependent_columns[idx]],label='actual')
        axes[idx].plot(approx_data[dependent_columns[idx]],label='approx')
        if idx == 0:
            axes[idx].legend(fontsize='x-large',loc='best')
        axes[idx].set_ylabel(dependent_columns[idx],fontsize='small')
        if idx == len(dependent_columns)-1:
            axes[idx].set_xlabel("time",fontsize='x-large')
    axes[0].set_title("outputs",fontsize='xx-large')
    plt.tight_layout()
    plt.savefig("C:/teamwork_without_talking/plant_approx.png")
    plt.savefig("C:/teamwork_without_talking/plant_approx.svg")

    plt.show()
    #plt.close()

    # end validation





### TRAIN model approximation

# reindex the response to integer timestep
response.index = np.arange(0,len(response),1)

max_iter = 250 # 250
max_transition_state_dim = 25 # 25


# learn the dynamics
lti_plant_approx = modpods.lti_system_gen(connectivity,response,independent_columns = env.config['action_space'],
                                          dependent_columns = dependent_columns, max_iter = max_iter,
                                          swmm=True,bibo_stable=True,max_transition_state_dim=max_transition_state_dim)


if max_iter < 5:
    # pickle the plant approximation to load later
    with open("C:/teamwork_without_talking/plant_approx_continuous_lofi.pickle", 'wb') as handle:
        pickle.dump(lti_plant_approx, handle)
else:
    # pickle the plant approximation to load later
    with open("C:/teamwork_without_talking/plant_approx_continuous.pickle", 'wb') as handle:
        pickle.dump(lti_plant_approx, handle)
'''
# load the plant approximation
with open("C:/teamwork_without_talking/plant_approx_continuous.pickle", 'rb') as handle:
    lti_plant_approx = pickle.load(handle)
'''
# is the plant approximation internally stable?
plant_eigenvalues,_ = np.linalg.eig(lti_plant_approx['A'].values)
    

# convert the plant approximation to discrete time
lti_plant_approx_discrete = matlab.c2d(lti_plant_approx['system'], Ts = 1, method = 'impulse') 

# include the shift matrices upstream of the rainfall states
# going to use a 2 day prediction window, timestep is 5 minutes. that gives 2 days / 5 minutes = 576 timesteps
# change to a 6 hour prediction window, timestep is 5 minutes. that gives 6 hours / 5 minutes = 72 timesteps
# the shift matrix will be 72x72 and then we'll add a 1 from the raingage_X_00H05M state to the raingage_X_now state
# that gives an A matrix of about 3000x3000. 9000 floats in C is 35.16 KB out of 256 KB flash - https://github.com/open-storm/open-storm-hardware 
# could also put that onto an SD card if it's too big which is no problem
shift = np.eye(72)
shift = np.roll(shift,-1,axis=1)
shift[0,:] = 0
#print(shift)

'''
# the names for the states of that matrix are formatted as: raingage_X_00H05M, raingage_X_00H10M, raingage_X_00H15M, ... raingage_X_47H55M, raingage_X_48H00M
# make a list of these names. separate for each raingage
raingage_J_shift_names = ['raingage_J_' + str(i) + 'H' + str(j) + 'M' for i in range(48) for j in range(0,60,5)]
# get rid of 0H0M and add 48H0M
raingage_J_shift_names = raingage_J_shift_names[1:]
raingage_J_shift_names.append('raingage_J_48H00M')
# reverse the names so that the farthest in the future is at the top
raingage_J_shift_names = raingage_J_shift_names[::-1]


#print(raingage_J_shift_names)

# add the shift matrices to the plant approximation
# just for debugging, only use the last couple entries in raingage_J_shift_names
raingage_J_shift_names = raingage_J_shift_names[-5:]
# make shift smaller to match
shift = shift[-5:,-5:]
print(shift)

# for raingage J
before_raingage_J_index = list(lti_plant_approx_discrete.state_labels[:lti_plant_approx_discrete.state_labels.index('raingage_J')])
after_raingage_J_index = list(lti_plant_approx_discrete.state_labels[lti_plant_approx_discrete.state_labels.index('raingage_J')+1:])
states = before_raingage_J_index + raingage_J_shift_names + ['raingage_J'] + after_raingage_J_index # state dimension expands by 72
newA = pd.DataFrame(index=states,columns=states)
for row in newA.index: # can you vectorize these operations? would likely be much quicker if you did. currently takes almost five minutes when adding the last raingage (S)
    for col in newA.columns:
        if row in lti_plant_approx_discrete.state_labels and col in lti_plant_approx_discrete.state_labels:
            row_idx = lti_plant_approx_discrete.state_labels.index(row)
            col_idx = lti_plant_approx_discrete.state_labels.index(col)
            newA.loc[row,col] = lti_plant_approx_discrete.A[row_idx,col_idx] # copy over the original entries
newA.loc[raingage_J_shift_names,raingage_J_shift_names] = shift # add the shift matrix
# make sure the row 'raingage_J' is all zeros 
newA.loc['raingage_J',:] = 0
# why is this here? -> sindy's best guess is a constant value (dx/dt = 0) when fitting which is converted to 1.0 when we make the discrete time conversion

# add a "1" in the row "raingage_J" and the column "raingage_J_00H05M"
newA[raingage_J_shift_names[-1]]['raingage_J'] = 1
# fill na's with zero
newA = newA.fillna(0)

# print all the entries of newA which include 'raingage_J'
#print(newA.loc[newA.index.str.contains('raingage_J'),newA.columns.str.contains('raingage_J')])

# the whole thing becomes a cascade. there's clean discrete transitions in the prediction propagation (shift matrix), and then you have a linear reservoir cascade for the unit hydrograph
#print(newA.loc[newA.index.str.contains('raingage_J'),newA.columns.str.contains('raingage_J')])
# print the statement above but just as a matrix, float precision two decimal places
#print(newA.loc[newA.index.str.contains('raingage_J'),newA.columns.str.contains('raingage_J')].values.round(2))

#print("done")

# keep this code snippet above just to look back at
'''

# do raingage_J for real now with the full shift matrix
#raingage_J_shift_names = ['raingage_J_' + str(i) + 'H' + str(j) + 'M' for i in range(48) for j in range(0,60,5)]
raingage_J_shift_names = ['raingage_J_' + str(i) + 'H' + str(j) + 'M' for i in range(6) for j in range(0,60,5)]
# get rid of 0H0M and add 48H0M
raingage_J_shift_names = raingage_J_shift_names[1:]
#raingage_J_shift_names.append('raingage_J_48H00M')
raingage_J_shift_names.append('raingage_J_06H00M')

# reverse the names so that the farthest in the future is at the top
raingage_J_shift_names = raingage_J_shift_names[::-1]
shift = np.eye(72)
shift = np.roll(shift,-1,axis=1)
shift[0,:] = 0

# for raingage J
before_raingage_J_index = list(lti_plant_approx_discrete.state_labels[:lti_plant_approx_discrete.state_labels.index('raingage_J')])
after_raingage_J_index = list(lti_plant_approx_discrete.state_labels[lti_plant_approx_discrete.state_labels.index('raingage_J')+1:])
states = before_raingage_J_index + raingage_J_shift_names + ['raingage_J'] + after_raingage_J_index # state dimension expands by 72
newA = pd.DataFrame(index=states,columns=states)
for row in newA.index:
    for col in newA.columns:
        if row in lti_plant_approx_discrete.state_labels and col in lti_plant_approx_discrete.state_labels:
            row_idx = lti_plant_approx_discrete.state_labels.index(row)
            col_idx = lti_plant_approx_discrete.state_labels.index(col)
            newA.loc[row,col] = lti_plant_approx_discrete.A[row_idx,col_idx] # copy over the original entries
newA.loc[raingage_J_shift_names,raingage_J_shift_names] = shift # add the shift matrix
# make sure the row 'raingage_J' is all zeros 
newA.loc['raingage_J',:] = 0
# why is this here? -> sindy's best guess is a constant value (dx/dt = 0) when fitting which is converted to 1.0 when we make the discrete time conversion

# add a "1" in the row "raingage_J" and the column "raingage_J_00H05M"
newA[raingage_J_shift_names[-1]]['raingage_J'] = 1
# fill na's with zero
newA = newA.fillna(0)

# print all the entries of newA which include 'raingage_J'
#print(newA.loc[newA.index.str.contains('raingage_J'),newA.columns.str.contains('raingage_J')])


# now update B accordingly
newB = pd.DataFrame(index=states,columns=lti_plant_approx_discrete.input_labels) # columns will remain the same
# insert rows of zeros for all the new states. otherwise, copy over the original B matrix
for row in newB.index:
    if row in lti_plant_approx_discrete.state_labels:
        row_idx = lti_plant_approx_discrete.state_labels.index(row)
        newB.loc[row,:] = lti_plant_approx_discrete.B[row_idx,:]
    else:
        newB.loc[row,:] = 0

# now update C
newC = pd.DataFrame(index=lti_plant_approx_discrete.output_labels,columns=states) # rows will remain the same
# insert columns of zeros for all the new states. otherwise, copy over the original C matrix
for col in newC.columns:
    if col in lti_plant_approx_discrete.state_labels:
        col_idx = lti_plant_approx_discrete.state_labels.index(col)
        newC.loc[:,col] = lti_plant_approx_discrete.C[:,col_idx]
    else:
        newC.loc[:,col] = 0

# after A, B, and C have all been redefined, update the system approximation object

# cast newA, newB, and newC to floats
newA = newA.astype(float)
newB = newB.astype(float)
newC = newC.astype(float)

# reality check, make sure you didn't break anything
# print out the column names which have nonzero entries in A for the row for (1,depthN). include what the coefficients are
print(newA.loc["('1', 'depthN')"].loc[newA.loc["('1', 'depthN')"] != 0])
# do the same for the B matrix
print(newB.loc["('1', 'depthN')"].loc[newB.loc["('1', 'depthN')"] != 0])

lti_plant_approx_discrete = ct.ss(newA, newB, newC, 0, inputs = newB.columns, outputs=newC.index, states=newA.columns,dt = 1)

# do the same for raingage_C
#raingage_C_shift_names = ['raingage_C_' + str(i) + 'H' + str(j) + 'M' for i in range(48) for j in range(0,60,5)]
raingage_C_shift_names = ['raingage_C_' + str(i) + 'H' + str(j) + 'M' for i in range(6) for j in range(0,60,5)]
# get rid of 0H0M and add 48H0M
raingage_C_shift_names = raingage_C_shift_names[1:]
#raingage_C_shift_names.append('raingage_C_48H00M')
raingage_C_shift_names.append('raingage_C_06H00M')

# reverse the names so that the farthest in the future is at the top
raingage_C_shift_names = raingage_C_shift_names[::-1]
shift = np.eye(72)
shift = np.roll(shift,-1,axis=1)
shift[0,:] = 0

# for raingage C
before_raingage_C_index = list(lti_plant_approx_discrete.state_labels[:lti_plant_approx_discrete.state_labels.index('raingage_C')])
after_raingage_C_index = list(lti_plant_approx_discrete.state_labels[lti_plant_approx_discrete.state_labels.index('raingage_C')+1:])
states = before_raingage_C_index + raingage_C_shift_names + ['raingage_C'] + after_raingage_C_index # state dimension expands by 72
newA = pd.DataFrame(index=states,columns=states)
for row in newA.index:
    for col in newA.columns:
        if row in lti_plant_approx_discrete.state_labels and col in lti_plant_approx_discrete.state_labels:
            row_idx = lti_plant_approx_discrete.state_labels.index(row)
            col_idx = lti_plant_approx_discrete.state_labels.index(col)
            newA.loc[row,col] = lti_plant_approx_discrete.A[row_idx,col_idx] # copy over the original entries
newA.loc[raingage_C_shift_names,raingage_C_shift_names] = shift # add the shift matrix
# make sure the row 'raingage_C' is all zeros
newA.loc['raingage_C',:] = 0
# add a "1" in the row "raingage_C" and the column "raingage_C_00H05M"
newA[raingage_C_shift_names[-1]]['raingage_C'] = 1
# fill na's with zero
newA = newA.fillna(0)

# now update B accordingly
newB = pd.DataFrame(index=states,columns=lti_plant_approx_discrete.input_labels) # columns will remain the same
# insert rows of zeros for all the new states. otherwise, copy over the original B matrix
for row in newB.index:
    if row in lti_plant_approx_discrete.state_labels:
        row_idx = lti_plant_approx_discrete.state_labels.index(row)
        newB.loc[row,:] = lti_plant_approx_discrete.B[row_idx,:]
    else:
        newB.loc[row,:] = 0

# now update C
newC = pd.DataFrame(index=lti_plant_approx_discrete.output_labels,columns=states) # rows will remain the same
# insert columns of zeros for all the new states. otherwise, copy over the original C matrix
for col in newC.columns:
    if col in lti_plant_approx_discrete.state_labels:
        col_idx = lti_plant_approx_discrete.state_labels.index(col)
        newC.loc[:,col] = lti_plant_approx_discrete.C[:,col_idx]
    else:
        newC.loc[:,col] = 0
        
# after A, B, and C have all been redefined, update the system approximation object

# cast newA, newB, and newC to floats
newA = newA.astype(float)
newB = newB.astype(float)
newC = newC.astype(float)

# reality check, make sure you didn't break anything
# print out the column names which have nonzero entries in A for the row for (4,depthN). include what the coefficients are
print(newA.loc["('4', 'depthN')"].loc[newA.loc["('4', 'depthN')"] != 0])
# do the same for the B matrix
print(newB.loc["('4', 'depthN')"].loc[newB.loc["('4', 'depthN')"] != 0])

lti_plant_approx_discrete = ct.ss(newA, newB, newC, 0, inputs = newB.columns, outputs=newC.index, states=newA.columns,dt = 1)

# do the same for raingage_N
#raingage_N_shift_names = ['raingage_N_' + str(i) + 'H' + str(j) + 'M' for i in range(48) for j in range(0,60,5)]
raingage_N_shift_names = ['raingage_N_' + str(i) + 'H' + str(j) + 'M' for i in range(6) for j in range(0,60,5)]
# get rid of 0H0M and add 48H0M
raingage_N_shift_names = raingage_N_shift_names[1:]
#raingage_N_shift_names.append('raingage_N_48H00M')
raingage_N_shift_names.append('raingage_N_06H00M')

# reverse the names so that the farthest in the future is at the top
raingage_N_shift_names = raingage_N_shift_names[::-1]
# don't need to define shift again, it's the same as before

# for raingage N
before_raingage_N_index = list(lti_plant_approx_discrete.state_labels[:lti_plant_approx_discrete.state_labels.index('raingage_N')])
after_raingage_N_index = list(lti_plant_approx_discrete.state_labels[lti_plant_approx_discrete.state_labels.index('raingage_N')+1:])
states = before_raingage_N_index + raingage_N_shift_names + ['raingage_N'] + after_raingage_N_index # state dimension expands by 72
newA = pd.DataFrame(index=states,columns=states)
for row in newA.index:
    for col in newA.columns:
        if row in lti_plant_approx_discrete.state_labels and col in lti_plant_approx_discrete.state_labels:
            row_idx = lti_plant_approx_discrete.state_labels.index(row)
            col_idx = lti_plant_approx_discrete.state_labels.index(col)
            newA.loc[row,col] = lti_plant_approx_discrete.A[row_idx,col_idx] # copy over the original entries
newA.loc[raingage_N_shift_names,raingage_N_shift_names] = shift # add the shift matrix
# make sure the row 'raingage_N' is all zeros
newA.loc['raingage_N',:] = 0
# add a "1" in the row "raingage_N" and the column "raingage_N_00H05M"
newA[raingage_N_shift_names[-1]]['raingage_N'] = 1
# fill na's with zero
newA = newA.fillna(0)

# now update B accordingly
newB = pd.DataFrame(index=states,columns=lti_plant_approx_discrete.input_labels) # columns will remain the same
# insert rows of zeros for all the new states. otherwise, copy over the original B matrix
for row in newB.index:
    if row in lti_plant_approx_discrete.state_labels:
        row_idx = lti_plant_approx_discrete.state_labels.index(row)
        newB.loc[row,:] = lti_plant_approx_discrete.B[row_idx,:]
    else:
        newB.loc[row,:] = 0
        
# now update C
newC = pd.DataFrame(index=lti_plant_approx_discrete.output_labels,columns=states) # rows will remain the same
# insert columns of zeros for all the new states. otherwise, copy over the original C matrix
for col in newC.columns:
    if col in lti_plant_approx_discrete.state_labels:
        col_idx = lti_plant_approx_discrete.state_labels.index(col)
        newC.loc[:,col] = lti_plant_approx_discrete.C[:,col_idx]
    else:
        newC.loc[:,col] = 0
        
# after A, B, and C have all been redefined, update the system approximation object

# cast newA, newB, and newC to floats
newA = newA.astype(float)
newB = newB.astype(float)
newC = newC.astype(float)

# reality check, make sure you didn't break anything
# print out the column names which have nonzero entries in A for the row for (10,depthN). include what the coefficients are
print(newA.loc["('10', 'depthN')"].loc[newA.loc["('10', 'depthN')"] != 0])
# do the same for the B matrix
print(newB.loc["('10', 'depthN')"].loc[newB.loc["('10', 'depthN')"] != 0])

lti_plant_approx_discrete = ct.ss(newA, newB, newC, 0, inputs = newB.columns, outputs=newC.index, states=newA.columns,dt = 1)

# do the same for raingage_S
#raingage_S_shift_names = ['raingage_S_' + str(i) + 'H' + str(j) + 'M' for i in range(48) for j in range(0,60,5)]
raingage_S_shift_names = ['raingage_S_' + str(i) + 'H' + str(j) + 'M' for i in range(6) for j in range(0,60,5)]
# get rid of 0H0M and add 48H0M
raingage_S_shift_names = raingage_S_shift_names[1:]
#raingage_S_shift_names.append('raingage_S_48H00M')
raingage_S_shift_names.append('raingage_S_06H00M')

# reverse the names so that the farthest in the future is at the top
raingage_S_shift_names = raingage_S_shift_names[::-1]
# don't need to define shift again, it's the same as before

# for raingage S
before_raingage_S_index = list(lti_plant_approx_discrete.state_labels[:lti_plant_approx_discrete.state_labels.index('raingage_S')])
after_raingage_S_index = list(lti_plant_approx_discrete.state_labels[lti_plant_approx_discrete.state_labels.index('raingage_S')+1:])
states = before_raingage_S_index + raingage_S_shift_names + ['raingage_S'] + after_raingage_S_index # state dimension expands by 72
newA = pd.DataFrame(index=states,columns=states)
for row in newA.index:
    for col in newA.columns:
        if row in lti_plant_approx_discrete.state_labels and col in lti_plant_approx_discrete.state_labels:
            row_idx = lti_plant_approx_discrete.state_labels.index(row)
            col_idx = lti_plant_approx_discrete.state_labels.index(col)
            newA.loc[row,col] = lti_plant_approx_discrete.A[row_idx,col_idx] # copy over the original entries
newA.loc[raingage_S_shift_names,raingage_S_shift_names] = shift # add the shift matrix
# make sure the row 'raingage_S' is all zeros
newA.loc['raingage_S',:] = 0
# add a "1" in the row "raingage_S" and the column "raingage_S_00H05M"
newA[raingage_S_shift_names[-1]]['raingage_S'] = 1
# fill na's with zero
newA = newA.fillna(0)

# now update B accordingly
newB = pd.DataFrame(index=states,columns=lti_plant_approx_discrete.input_labels) # columns will remain the same
# insert rows of zeros for all the new states. otherwise, copy over the original B matrix

for row in newB.index:
    if row in lti_plant_approx_discrete.state_labels:
        row_idx = lti_plant_approx_discrete.state_labels.index(row)
        newB.loc[row,:] = lti_plant_approx_discrete.B[row_idx,:]
    else:
        newB.loc[row,:] = 0

# now update C
newC = pd.DataFrame(index=lti_plant_approx_discrete.output_labels,columns=states) # rows will remain the same
# insert columns of zeros for all the new states. otherwise, copy over the original C matrix
for col in newC.columns:
    if col in lti_plant_approx_discrete.state_labels:
        col_idx = lti_plant_approx_discrete.state_labels.index(col)
        newC.loc[:,col] = lti_plant_approx_discrete.C[:,col_idx]
    else:
        newC.loc[:,col] = 0
        
# after A, B, and C have all been redefined, update the system approximation object

# cast newA, newB, and newC to floats
newA = newA.astype(float)
newB = newB.astype(float)
newC = newC.astype(float)

# reality check, make sure you didn't break anything
# print out the column names which have nonzero entries in A for the row for (1,depthN). include what the coefficients are
print(newA.loc["('6', 'depthN')"].loc[newA.loc["('6', 'depthN')"] != 0])
# do the same for the B matrix
print(newB.loc["('6', 'depthN')"].loc[newB.loc["('6', 'depthN')"] != 0])

lti_plant_approx_discrete = ct.ss(newA, newB, newC, 0, inputs = newB.columns, outputs=newC.index, states=newA.columns,dt = 1)

if max_iter < 5:
        # pickle the discrete time plant approximation to load later
    with open("C:/teamwork_without_talking/plant_approx_discrete_w_predictions_lofi.pickle", 'wb') as handle:
        pickle.dump(lti_plant_approx_discrete, handle)
else:
    # pickle the discrete time plant approximation to load later
    with open("C:/teamwork_without_talking/plant_approx_discrete_w_predictions.pickle", 'wb') as handle:
        pickle.dump(lti_plant_approx_discrete, handle)


print("disturbance prediction matrices updated")

