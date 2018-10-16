import random
import simpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_SEED = 0
PT_MEAN = 10.0         # Avg. processing time in minutes
PT_SIGMA = 2.0         # Sigma of processing time
MTTF = 30.0           # Mean time to failure in minutes
BREAK_MEAN = 1 / MTTF  # Param. for expovariate distribution
REPAIR_TIME = 300.0     # Time it takes to repair a machine in minutes
JOB_DURATION = random.random()    # Duration of other jobs in minutes
NUM_MACHINES = 3     # Number of machines in the machine shop
WEEKS = 1              # Simulation time in weeks
SIM_TIME = WEEKS * 7 * 24 * 60  # Simulation time in minutes


def total_time_per_part():
    """Return total processing time for a wafer to get processed."""
    return random.normalvariate(PT_MEAN, PT_SIGMA)

def time_to_failure():
    """Return time until next failure for a machine."""
    return random.expovariate(BREAK_MEAN)

def num_of_machines_required():
    """Return the number of machines required to finish the wafer completely"""
    return random.randint(1,NUM_MACHINES)

def time_per_machine(k):
    """Return time spent by a wafer at each machine it is going into"""
    c = range(0, k)
    return random.sample(c, num_of_machines_required())


class Machine(object):
    """A machine produces wafers and may get broken every now and then.

    If it breaks, it requests a *repairman* and continues the production
    after the it is repaired.

    A machine has a *name* and a numberof *wafers_made* thus far.

    """
    def __init__(self, env, name, repairman):
        self.env = env
        self.name = name
        self.wafers_made = 0
        self.broke_num = 0 
        self.broken = False

        # Start "working" and "break_machine" processes for this machine.
        self.process = env.process(self.working(repairman))
        env.process(self.break_machine())

    def working(self, repairman):
        """Produce wafers as long as the simulation runs.
        While making a part, the machine may break multiple times.
        Request a repairman when this happens.
        """
        while True:
            # Start making a new part
            done_in = total_time_per_part()
            while done_in:
                try:
                    # Working on the part
                    start = self.env.now
                    #print("Started working machine %s" % self.name)
                    #print("----------------------")
                    yield self.env.timeout(done_in)
                    done_in = 0  # Set to 0 to exit while loop.

                except simpy.Interrupt:
                    self.broken = True
                    self.broke_num += 1
                    done_in -= self.env.now - start  # How much time left?
                    print('%s broke down and is in need for assistance' % self.name)
                    print("--------------------------------------------")
                    # Request a repairman. This will preempt its "other_job".
                    with repairman.request(priority=1) as req:
                        print('%s requesting at %s' % (self.name, self.env.now))
                        print("----------------------")
                        yield req
                        print('%s got resource at %s'%(self.name, self.env.now))
                        print("----------------------")
                        yield self.env.timeout(REPAIR_TIME)

                    self.broken = False

            # Part is done.
            self.wafers_made += 1

    def break_machine(self):
        """Break the machine every now and then."""
        while True:
            yield self.env.timeout(time_to_failure())
            if not self.broken:
                # Only break the machine if it is currently working.
                self.process.interrupt()


def other_jobs(env, repairman):
    """The repairman's other (unimportant) job."""
    while True:
        # Start a new job
        done_in = JOB_DURATION
        while done_in:
            # Retry the job until it is done.
            # It's priority is lower than that of machine repairs.
            with repairman.request(priority=2) as req:
                print('other jobs requesting assistance at %s' % env.now)
                print("--------------------------------------")
                yield req
                print('other jobs got assistance at %s'% env.now)
                print("--------------------------------------")
                try:
                    start = env.now
                    yield env.timeout(done_in)
                    done_in = 0
                except simpy.Interrupt:
                    done_in -= env.now - start


# Setup and start the simulation
print('Machine shop')
random.seed(RANDOM_SEED)  # This helps reproducing the results

# Create an environment and start the setup process
env = simpy.Environment()
repairman = simpy.PreemptiveResource(env, capacity=1)
machines = [Machine(env, 'Machine M%d' % i, repairman) for i in range(NUM_MACHINES)]
env.process(other_jobs(env, repairman))

# Execute!
env.run(SIM_TIME)
# for i in range(1,SIM_TIME):
#     env.run(until=i)
#     env.progressbar.update(i)
# print(progressbar)
# Analyis/results
print('Machine shop results after %s weeks' % WEEKS)
dis = []
for machine in machines:
    print('%s made %d wafers and broke %d times.' % (machine.name, machine.wafers_made, machine.broke_num))
    dis.append(machine.wafers_made)

# print(type(dis)
#print(dis)
#print(len(dis))
# plt.plot(progressbar)
# plt.show()
mac = list(range(0,NUM_MACHINES))
sns.barplot(mac, dis)
plt.show()









