
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
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam 
from collections import deque 


SIM_TIME = 10**3 #simulation time
# OBS_TIME = 100 # observation time cycle
# BREAK_TIME = random.expovariate(.5) # time after which we send in a new set of cassettes 

UPDATE_TARGET_NETWORK = 10

WAIT_TIME_IF_FULL = 6


# Create an environment and start the setup process
env = simpy.Environment()

n=0

Machine = recordtype('Machine', 'duration, name, station_name, queue_length, max_q_length, output, t_start')

m1 = Machine(10, 'M0','S1', 0, 1000, 0, 0)
m2 = Machine(10, 'M1','S1', 0, 1000, 0, 0)
m3 = Machine(10, 'M2','S1', 0, 1000, 0, 0)
m4 = Machine(20, 'M3','S2', 0, 1000, 0, 0)
m5 = Machine(20, 'M4','S2', 0, 1000, 0, 0)
m6 = Machine(20, 'M5','S2', 0, 1000, 0, 0)


station1 = [m1, m2, m3]
station2 = [m4, m5, m6]

# Total list of all the stations in the factory 
ls = [station1, station2]

# List to store the wait time of each wafer
wt = []

# A fixed state space of our simulation
STATE_SPACE = [[m.queue_length for m in s] for s in ls]

# A fixed action space of our simulation
ACTION_SPACE = [[m for m in s] for s in ls]


class DQN:
    def __init__(self):
        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.tau = 0.125
        self.learning_rate = 0.005
        self.memory = deque(maxlen= 2000)
        self.model = self.create_model()
        self.target_model = self.create_model()



    # create the neural network to train the q function 
    def create_model(self):
        model = Sequential()
        st = np.array(STATE_SPACE).reshape(1,len(sum(STATE_SPACE, [])))
        act = np.array(ACTION_SPACE).reshape(1,len(sum(ACTION_SPACE, [])))
        state_shape = st.shape
        action_shape = act.shape
        model.add(Dense(24, input_dim= (state_shape[1] +1), activation= 'relu'))
        model.add(Dense(48, activation= 'relu'))
        model.add(Dense(24, activation= 'relu'))
        model.add(Dense(action_shape[1]))
        model.compile(loss= 'mean_squared_error', optimizer= Adam(lr= self.learning_rate))
        return model 



    # Action function to choose the best action given the q-function if not exploring based on epsilon
    def choose_action(self, state, allowed_actions):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        r = np.random.random()
        if r < self.epsilon:
            #print("**** Choosing a random action ****")
            return random.choice(allowed_actions)
        state = np.array(state).reshape(1,len(state))
        list_of_allowed_machines = []
        
        for i in range(len(allowed_actions)):
            list_of_allowed_machines.append(int(allowed_actions[i].name[-1]))

        pred = self.model.predict(state)
        pred = sum(pred.tolist(), [])
        semi_list = [pred[i] for i in list_of_allowed_machines]
        #print("************************* Choosing a predicted action **********************************")
        return allowed_actions[np.argmax(semi_list)]



    # create replay buffer memory to sample randomly
    def remember(self, state, action, reward, next_state, next_action, done):
        self.memory.append([state, action, reward, next_state, next_action, done])



    # build the replay buffer
    def replay(self):
        batch_size = 32
        list_of_next_allowed_machines = []
        if len(self.memory) < batch_size:
            return 
        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, new_action, done = sample
            state = np.array(state).reshape(1,state.shape[0])
            new_state = np.array(new_state).reshape(1,new_state.shape[0])
            target = self.target_model.predict(state)
            action_id = int(action.name[-1])
            if done:
                target[0][action_id] = reward
            else:
                # take max only from next_allowed_actions
                for i in range(len(new_action)):
                    list_of_next_allowed_machines.append(int(new_action[i].name[-1]))

                next_pred = self.target_model.predict(new_state)[0]
                next_pred = next_pred.tolist()
                semi_next_list = [next_pred[i] for i in list_of_next_allowed_machines]
                Q_future = max(semi_next_list)
                target[0][action_id] = reward + Q_future * self.gamma

            self.model.fit(state, target, epochs= 1, verbose= 0)



    # update our target network 
    def train_target(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)



    # save our model 
    def save_model(self, fn):
        self.model.save(fn)


# Initialize the policy and the target network
dqn_agent = DQN()


    
# Start sending the wafers into the factory 
def run_incoming_cassettes(env, ls, n, breaktime):
    global loop_num
    loop_num = 0 
    # sending in the cassettes
    while True:
        print("Working on Cassette set C%s"%(loop_num))
        loop_num += 1
        k = 0
        # sending in the wafers
        while k < n:
            env.process(wafer(env, ls))
            k += 1
        yield(env.timeout(breaktime))
    print("Completed in {} cassettes".format(m))
    dqn_agent.save_model("policy_network.h5")
    



def wafer(env, st):
    lenst = len(st)
    i = 0
    total_run_t = 0
    t_arrival = env.now

    while(i!=lenst):
        state = [[m.queue_length for m in s] for s in st]
        allowed_actions = [m for m in st[i] if m.queue_length < m.max_q_length]

        # appending the current station number to the state
        state = np.append(state, i)
        print("current state", state)
        print(" ")
        print(" ")
        dispatch_machine = dqn_agent.choose_action(state, allowed_actions)
        #print("selected action", dispatch_machine.name)
        dispatch_machine.queue_length = dispatch_machine.queue_length + 1
        
        q_enter_time = env.now

        if dispatch_machine.queue_length <= 1:
            dispatch_machine.t_start = env.now
    
        running_time = env.now - dispatch_machine.t_start            
        yield env.timeout(dispatch_machine.queue_length*dispatch_machine.duration-running_time)
        
        total_run_t += dispatch_machine.duration
        #print("total_run_t", total_run_t)
        dispatch_machine.queue_length = dispatch_machine.queue_length - 1
        
        next_state = [[m.queue_length for m in s] for s in st]

        # appending the current station number to the state
        next_state = np.append(next_state, i+1)

        # build your reward function
        reward = (q_enter_time - env.now) - np.sum(state)

        if i < lenst-1:
            done = False
            next_allowed_actions = [m for m in st[i+1] if m.queue_length < m.max_q_length]
        else:
            done = True

        cur_state = state 
        action = dispatch_machine
        new_state = next_state 
        new_action = next_allowed_actions
        reward = reward
        done = done

        dqn_agent.remember(cur_state, action, reward, new_state, new_action, done)
        dqn_agent.replay()

        # updating the target network less frequently
        if(loop_num%UPDATE_TARGET_NETWORK == 0):
            dqn_agent.train_target()


        if dispatch_machine.queue_length >=1:
            dispatch_machine.t_start = env.now
        
        dispatch_machine.output = dispatch_machine.output + 1
       
        i = i + 1

    t_del = env.now - t_arrival
    w_del = t_del - total_run_t
    
    wt.append(w_del)




# Start Processing a set of n casset
env.process(run_incoming_cassettes(env, ls, 5, 25))

# Start the simulation 
env.run(SIM_TIME)

print("Total wafers made: "len(wt))
print("Wait times of each wafer:")
print(wt)

plt.plot(wt)
plt.show()















































