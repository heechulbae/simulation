import random

import simpy


RANDOM_SEED = 42
NUM_MACHINES = 10  # Number of machines in the factory
PROCESSTIME = 5      # Minutes it takes to produce a wafer
T_INTER = 7       # Create a wafer every ~7 minutes
WEEKS = 1              # Simulation time in weeks
SIM_TIME = WEEKS * 7 * 24 * 60  # Simulation time in minutes



class factory(object):
    """A factory has a limited number of machines (``NUM_MACHINES``) to
    create wafers in parallel.

    Wafers have to request one of the machines. When they get one, they
    can start the building processes and wait for it to finish (which
    takes ``PROCESSTIME`` minutes).

    """
    def __init__(self, env, num_machines, proc_time):
        self.env = env
        self.machine = simpy.Resource(env, num_machines)
        self.proc_time = proc_time

    def build(self, wafer):
        """The making processes. It takes a ``wafer`` processes and tries
        to make it."""
        yield self.env.timeout(PROCESSTIME)
        


def wafer(env, name, fac):
    """The wafer process (each wafer has a ``type``) arrives at the factory
    (``fac``) and requests a building machine.

    It then starts the making process, waits for it to finish and
    leaves to never come back ...

    """
    print('%s arrives at the factory at %.2f.' % (name, env.now))
    with fac.machine.request() as request:
        yield request

        print('%s enters the factory at %.2f.' % (name, env.now))
        yield env.process(fac.build(name))

        print('%s leaves the factory at %.2f.' % (name, env.now))


def setup(env, num_machines, buildtime, t_inter):
    """Create a factory, a number of initial wafers and keep creating wafers
    approx. every ``t_inter`` minutes."""
    # Create the factory
    Factory = factory(env, num_machines, buildtime)

    # Create 4 initial wafers
    for i in range(100):
        env.process(wafer(env, 'Wafer %d' % i, Factory))

    # Create more wafers while the simulation is running
    while True:
        yield env.timeout(random.randint(t_inter - 2, t_inter + 2))
        i += 1
        env.process(wafer(env, 'Wafer %d' % i, Factory))


# Setup and start the simulation
print('Factory')
random.seed(RANDOM_SEED)  # This helps reproducing the results

# Create an environment and start the setup process
env = simpy.Environment()
env.process(setup(env, NUM_MACHINES, PROCESSTIME, T_INTER))

# Execute!
env.run(until=SIM_TIME)



