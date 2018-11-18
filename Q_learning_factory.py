
import simpy
import pandas as pd
import numpy as np 
from collections import namedtuple
from recordtype import recordtype
import random 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from prettytable import PrettyTable
#get_ipython().magic('matplotlib inline')


SIM_TIME = 10000000 #simulation time
# OBS_TIME = 100 # observation time cycle
# BREAK_TIME = random.expovariate(.5) # time after which we send in a new set of cassettes 


WAIT_TIME_IF_FULL = 6


# Create an environment and start the setup process
env = simpy.Environment()

n=0

Machine = recordtype('Machine', 'duration, name, station_name, queue_length, max_q_length, output, t_start')

m1 = Machine(20, 'M1','S1', 0, 3, 0, 0)
m2 = Machine(20, 'M2','S1', 0, 3, 0, 0)
m3 = Machine(20, 'M3','S1', 0, 3, 0, 0)
m4 = Machine(10, 'M4','S2', 0, 3, 0, 0)
m5 = Machine(10, 'M5','S2', 0, 3, 0, 0)
m6 = Machine(10, 'M6','S2', 0, 3, 0, 0)


station1 = [m1, m2, m3]
station2 = [m4, m5, m6]

# Total list of all the stations in the factory 
ls = [station1, station2]

#List to store the state space everytime an action takes place 
state_list = []

# List to store all the possible actions
action_list = []

# List to store the actions taken (choosing a machine) 
acted = []

# List to store the output of all the machines 
output_list = []

# The time at which a wafer enters the factory or makes a transition to a new station
ip_time = []

# The time at which a wafer enters the factory or makes a transition to a new station
op_time = []
wafer_list = []

# List to store the wait time of each wafer
wt = []

# Dictionary for q function values
q_dict = {}

epsilon = 0.0

N_EPISODES = 10**4*3*2  # 10^24

MIN_ALPHA = 0.01

# learning rate 
alphas = np.linspace(1, MIN_ALPHA, N_EPISODES)


# Q function to maintain the Q-value table
def q(state, all_actions, action=None):
    if state not in q_dict:
        q_dict[state] = np.zeros(len(all_actions))

    if action is None:
        return q_dict[state]
    
    return q_dict[state][action]



# Action function to choose the best action given the q-table if not exploring based on epsilon
def choose_action(state, allowed_actions):
    if random.uniform(0, 1) < epsilon:
        return random.choice(allowed_actions) 
    else:
        return allowed_actions[np.argmax(q(state, allowed_actions))]



def wafer(name, env, st):
    global n
    global state_list
    global action_list
    global acted 
    global output_list
    global ip_time
    global op_time
    global wafer_list 
    global wt
    #allowed_actions = []
    l = len(st)
    i = 0
#     print("W%s entered the factory at t = %s" %(name, env.now))
#     print("-----------------------------------")
    total_run_t = 0
    t_arrival = env.now
    while(i!=l):
        state = [[m.queue_length for m in s] for s in st]
        
        state_str = str([state, i])
#        print(state_str)
#         print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
#         print('The Current queue state is:')
#         print(state_str)
#        state_list.append(state)
#         print('The allowable machines for dispatch are:')
        allowed_actions = [m for m in st[i] if m.queue_length < m.max_q_length]
        # for m in st[i]:
        #     if m.queue_length < m.max_q_length:
        #         allowed_actions.append(m)
        #     else:
        #         raise Exception('All the queues are full, cannot take in more wafers!')
        #     yield env.timeout(WAIT_TIME_IF_FULL)

        # print(type(allowed_actions))
#        action_list.append([m.name for m in allowed_actions])
#         print([m.name for m in allowed_actions])
        
        dispatch_machine = choose_action(state_str, allowed_actions)
        dispatch_machine.queue_length = dispatch_machine.queue_length + 1
        
        q_enter_time = env.now
        
#         print("W%s got %s from %s at t = %s"%(name, dispatch_machine.name, dispatch_machine.station_name, env.now))
#         print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        
#        ip_time.append(env.now)
#        acted.append(dispatch_machine.name)
#        wafer_list.append('W'+str(name))

        if dispatch_machine.queue_length <= 1:
            dispatch_machine.t_start = env.now
            
        running_time = env.now - dispatch_machine.t_start            
        
        yield env.timeout(dispatch_machine.queue_length*dispatch_machine.duration-running_time)
        
        total_run_t += dispatch_machine.duration
#        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        dispatch_machine.queue_length = dispatch_machine.queue_length - 1
        
        next_state = [[m.queue_length for m in s] for s in st]
        next_state_str = str([next_state, i+1])
        
        print("state",state)
        print("action taken",dispatch_machine.name)
        print("next_state", next_state)


        if i < l-1:
            next_allowed_actions = [m for m in st[i+1] if m.queue_length < m.max_q_length]
            reward = q_enter_time - env.now
        else:
            reward = q_enter_time - env.now
        
        m_index = allowed_actions.index(dispatch_machine)
        # Update your q-value for that corresponding action taken 
        q(state_str, allowed_actions)[m_index] = q(state_str, allowed_actions, m_index) + alphas[n] * (reward + np.max(q(next_state_str, next_allowed_actions)) - q(state_str, allowed_actions, m_index))
        n+=1
        
        if dispatch_machine.queue_length >=1:
            dispatch_machine.t_start = env.now
        
        dispatch_machine.output = dispatch_machine.output + 1
       
#         print("W%s released %s from %s at t = %s"%(name, dispatch_machine.name, dispatch_machine.station_name, env.now))
#        op_time.append(env.now)
#         print('the total output of each machine is:')
#         print([[m.output for m in station] for station in st])
#        output_list.append([[m.output for m in station] for station in st])
#         print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$') 
        i = i + 1
    t_del = env.now - t_arrival
    w_del = t_del - total_run_t
    print("************")
    wt.append(w_del)
#     print("W%s left the factory at t = %s" %(name, env.now))
#     print("|||||||||||||||||||||||||||||||")

    
    
# Start sending the wafers into the factory 
def run_incoming_cassettes(env, ls, m, n, breaktime):
    l = 0
    # sending in the cassettes 
    while l < m:
        l += 1
        i = 0
        # sending in the wafers
        while i < n:
            env.process(wafer(i, env, ls))
            i += 1
        yield(env.timeout(breaktime))
    
# Start Processing a set of n casset
env.process(run_incoming_cassettes(env, ls, 2*10**3, 3, 24))

# Start the simulation 
env.run(SIM_TIME)


#t = PrettyTable(['Time Step', 'Wafer Name', 'Input Time', 'Queue State Space', 'Action Space', 'Action taken', '                                ', 'Output Time', 'Output State Space'])

# df = pd.DataFrame([wafer_list[i], ip_time[i], state_list[i], action_list[i], acted[i]])
# print(df)

# for i in range(len(state_list)):
#     t.add_row([i, wafer_list[i], ip_time[i], state_list[i], action_list[i], acted[i], '                             ', op_time[i], output_list[i]])
#     t.add_row([" ", " ", " ", " ", " ", " ", " ", " ", " "])


# print(t)

#np.mean(wt[1500:])

print(len(wt))

plt.plot(wt)
plt.show()


print(len(q_dict))
#print(q_dict.values())

