import numpy as np


class Simulation:
    def __init__(self):
        self.num_in_system = 0
        
        self.clock = 0.0
        self.t_arrival = self.generate_interarrival()
        self.t_depart = float('inf')
        
        self.num_arrivals = 0
        self.num_departs = 0
        self.total_wait = 0.0
        
    def advance_time(self):
        t_event = min(self.t_arrival, self.t_depart)
        
        self.total_wait += self.num_in_system*(t_event - self.clock)
        
        self.clock = t_event
        
        if self.t_arrival <= self.t_depart:
            self.handle_arrival_event()
        else:
            self.handle_depart_event()
            
    def handle_arrival_event(self):
        self.num_in_system += 1
        self.num_arrivals += 1
        if self.num_in_system <= 1:
            self.t_depart = self.clock + self.generate_service()
        self.t_arrival = self.clock + self.generate_interarrival()
    
    def handle_depart_event(self):
        self.num_in_system -= 1
        self.num_departs += 1
        if self.num_in_system > 0:
            self.t_depart = self.clock + self.generate_service()
        else:
            self.t_depart = float('inf')
            
    def generate_interarrival(self):
        return np.random.exponential(1./3)
    
    def generate_service(self):
        return np.random.exponential(1./4)
    
s = Simulation()


for i in range(1,15):
    hours = int(s.clock)
    minutes = (s.clock*60) % 60
    seconds = (s.clock*3600) % 60
    s.advance_time()
    print("Clock time:",("%d:%02d.%02d" % (hours, minutes, seconds))) 
    print("Number of wafers in system:",s.num_in_system) 
    print("No of arrivals:",s.num_arrivals)
    print("No. of departures:",s.num_departs)
print("efficiency", s.total_wait / s.num_departs)


