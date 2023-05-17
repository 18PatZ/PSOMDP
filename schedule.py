import numpy as np

class Schedule:
    
    def __init__(self, strides, pi_exec_data, pi_checkin_data, pi_mid_data, is_multi_layer=False):
        self.strides = strides
        self.pi_exec_data = pi_exec_data
        self.pi_checkin_data = pi_checkin_data
        self.pi_mid_data = pi_mid_data
        self.is_multi_layer = is_multi_layer

        self.set_bounds()

    def to_str(self):
        name = ""
        for checkin in self.strides:
            name += str(checkin)

        if self.is_multi_layer:
            name = f"({name})*"
        else:
            name += "*"
        return name

    def set_upper_bound(self):
        self.upper_bound = [self.pi_exec_data]

        if self.pi_checkin_data is not None: # if no pi_checkin_data, that means schedule is single point (pi^exec) not upper/lower bounds
            self.upper_bound += self.pi_mid_data
            self.upper_bound.append(self.pi_checkin_data)

    def set_lower_bound(self): # needs upper bound set first!
        # lower bound is bottom left corner of each pair of upper bound points
        # |_
        # |_|_
        #   |_|_
        self.lower_bound = []

        if self.pi_checkin_data is not None:
            for i in range(len(self.upper_bound)-1):
                p1 = self.upper_bound[i]
                p2 = self.upper_bound[i+1]

                self.lower_bound.append((p1[0], p2[1]))

    def set_bounds(self):
        self.set_upper_bound()
        self.set_lower_bound()

    def project_bounds(self, func):
        self.proj_lower_bound = [func(point) for point in self.lower_bound]
        self.proj_upper_bound = [func(point) for point in self.upper_bound]

    def not_dominated_by(self, upper_bound):
        for lower_point in self.proj_lower_bound:
            not_dominated = bool(np.all(np.any(upper_bound > lower_point,axis=1)))
            if not_dominated: # one of my lower bound points is not dominated by any of the other schedule's upper bound points
                return True
        return False

    def get_proj_bounds(self):
        return ScheduleBounds([self.to_str(), self.proj_lower_bound, self.proj_upper_bound])


class ScheduleBounds:
    def __init__(self, data):
        self.name = data[0]
        self.lower_bound = data[1]
        self.upper_bound = data[2]

    def to_arr(self):
        return [self.name, self.lower_bound, self.upper_bound]