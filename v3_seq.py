import simpy 
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_SEED = 0
PT_MEAN = 10.0         # Avg. processing time in minutes
PT_SIGMA = 2.0         # Sigma of processing time
RGW = 1.0 / 5.0		 # Rate of generating wafers (1 wafer per minute)
MTTF = 3.0           # Mean time to failure in minutes
BREAK_MEAN = 1.0 / MTTF  # Param. for expovariate distribution
REPAIR_TIME = 300.0     # Time it takes to repair a machine in minutes
JOB_DURATION = random.random()    # Duration of other jobs in minutes
NUM_MACHINES = 35    # Number of machines in the machine shop
NUM_REPAIRMEN = 4	 # Number of repairmen in the factory 
OBS_TIME = 0.5		 # Observe time after which you want to observe the machines 
NOW = 0

WEEKS = 4              # Simulation time in weeks
SIM_TIME = WEEKS * 7 * 24 * 60	# Simulation time in minutes

# 

def total_time_per_part():
    """Return  processing time for a wafer to get processed on a given machine."""
    return random.normalvariate(PT_MEAN, PT_SIGMA)

def time_to_failure():
    """Return time until next failure for a machine."""
    return random.expovariate(BREAK_MEAN)

def num_of_machines_required():
    """Return the number of machines required to finish the wafer completely"""
    return random.randint(1,NUM_MACHINES)

# def time_per_machine(total_time):
#     """Return time spent by a wafer at each machine it is going into"""
#     return random.uniform(0, total_time)

def generate_intervals():
	"""Return the time after which a wafer enters factory"""
	return random.expovariate(RGW)

# A list to store the wait times of all the wafers
wait_t = []

# A list to store the queue length of the factory 
q_length = []

#Observation time of the factory in minutes 
obs_times = []


def machine(env, machines):
	i = 0
	global NOW
	while True:
		i+= 1
		yield env.timeout(generate_intervals())
		env.process(wafer(env, i, machines))
		NOW = NOW + 1
		



def wafer(env, wafer_name, machines):
	num = num_of_machines_required()
	t_arrival = env.now
	ls = []
	print("Wafer W%s enters the factory at %s and has to go through %s machines to complete it's process" % (wafer_name, t_arrival, num))
	print("-------------------------------------------------------------------------------------")
	while(num!= 0):
		with machines.request() as request:
			yield request
			print('Wafer W%s is being processed at Machine M%s on %s min' % (wafer_name, num, env.now))
			print("------------------------------------------------------------------------------------------------------------")
			temp = total_time_per_part()
			ls.append(temp)
			yield env.timeout(temp)
			print('Wafer W%s finished processing at Machine M%s %s min' % (wafer_name, num, env.now))
			print("------------------------------------------------------------------------------------------------------------")
			num = num - 1
	t_depart = env.now
	print("Wafer W%s leaves the factory at %s" % (wafer_name, t_depart))
	wait_t.append(t_depart - t_arrival - sum(ls))
	print("Wait time for W%s is %s minutes" % (wafer_name, (t_depart - t_arrival - sum(ls))))
	print("------------------------------------------------------------------------------------------------------------")



def observe(env, machines):
	while True:
		obs_times.append(env.now)
		q_length.append(len(machines.queue))
		#print(machines.queue)
		yield env.timeout(OBS_TIME) # Observe these machines after every OBS_TIME minutes




# Setup and start the simulation

# This helps reproducing the results
random.seed(RANDOM_SEED)  

# Create an environment and start the setup process
env = simpy.Environment()

# Create the machine resources 
machines = simpy.Resource(env, capacity = NUM_MACHINES)

# Start the machines in the factory
env.process(machine(env, machines))

#Start the observation cycle simultaneously
env.process(observe(env, machines))

# Execute!
env.run(SIM_TIME)


# The wait time per wafer
#print("Wait time of all the wafers are:- ", wait_t)

#Plot the histogram of wait times of wafers at each machine

plt.hist(wait_t)
plt.xlabel("Waiting time (in min)")
plt.ylabel("Number of wafers")
plt.show()


# Plot the Q-length of each machine in the factory 
plt.step(obs_times, q_length, where = 'post')
plt.xlabel("Observation Time (in minutes)")
plt.ylabel("Queue length")
plt.show()

print("--------------******-------------")
print("--------------******-------------")
print("--------------******-------------")
print("--------------******-------------")
print("--------------******-------------")
print("--------------******-------------")
print("--------------******-------------")
print("--------------******-------------")
print("--------------******-------------")
print("--------------******-------------")
print("--------------******-------------")
print("--------------******-------------")
print("*********************************")
print("*********************************")
print('*****-WD Simulation Factory-*****')
print("*********************************")
print("*********************************")
print("--------------******-------------")
print("--------------******-------------")
print("--------------******-------------")
print("--------------******-------------")
print("--------------******-------------")
print("--------------******-------------")
print("--------------******-------------")
print("--------------******-------------")
print("--------------******-------------")
print("--------------******-------------")
print("--------------******-------------")
print("--------------******-------------")
print("Description:")
print("---------------------------------")
print("The factory simulation model consists of %s machines and currently uses a GREEDY POLICY to send wafers to the machines in the the factory. Right now we have built the model under the assumption that all the machines are unqiue but can process any wafer that accquires that particular machine. All the wafers that enter the factory have to go through a assigned number of step (no. of machines) to finish the process. This number is currently being sampled randomly from a random distribution constrained on being less than the total number of machines in the factory. All the wafers take a variable amount of time to finish being processed on a particular machine which is again sampled from a normal distribution with a given MEAN and SIGMA. We can also vary the rate at which each wafer enters the factory - currently its kept to be random. There are %s number of repairmen in the factory who take care of the wear and tear of the entire system. Whenever a machine breaks, it raises a repair request and a repairman repairs the machine on a priority basis. These requests preempt other regular tasks which reparimen are handling on a regular basis. Currently, we are randomly sampling the time after which a machine breaks but can be varied as per the need."% (NUM_MACHINES, NUM_REPAIRMEN))
print("--------------******-------------")
print("--------------******-------------")
print("--------------******-------------")
print("--------------******-------------")
print("--------------******-------------")
print("--------------******-------------")

print("Statistics:")
print("---------------------------------")
print("The simulation ran was %s minutes" % (SIM_TIME))
print("The wafers enter the factory at an average rate of %s wafer per minute"% (1/RGW))
print("The average processing time of a wafer on a single machine in the factory is %s minutes" %(PT_MEAN))
print("The number of machines a wafer has to pass through to complete the process is sampled randomly.")
print("Total machines in the factory %s." %(NUM_MACHINES))
print("Total repairmen in the factory %s." %(NUM_REPAIRMEN))
print("Total number of wafers produced by the factory during the given simulation time was %s." %(NOW))
print("Throughput of the factory in the given simulation was %s wafers/minute." %(NOW/SIM_TIME))
if (len(q_length)!=0):
	print("The average queue length throughout the simulation was %s" %(sum(q_length)/len(q_length)))
else:
	print("The average queue length was Zero.")
if(len(wait_t)!=0):
	print("The average wait time per wafer in the simulation was %s minutes." %(sum(wait_t)/len(wait_t)))
else:
	print("Average wait time per wafer in the simulation was Zero.")
print("--------------******-------------")
print("--------------******-------------")
print("--------------******-------------")
print("--------------******-------------")







