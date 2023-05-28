import sys
 
# adding above folder to the system path
sys.path.insert(0, '../')

from figure import *

import numpy as np
import math

def has(suffix, fronts):
    for i in range(len(fronts)-1):
        for f in fronts[i]:
            other_name = f[0]
            if suffix == other_name:
                    return True
    return False




fronts = []
for i in range(8):
    name = f"pareto-c3-l8-truth_no-alpha_-step{i+1}"
    schedules, is_efficient, optimistic_front, realizable_front = loadDataChains(name, outputDir="../output")

    fronts.append(realizable_front)


last = fronts[-1]
print(len(last),"on last front")
for front in last:
    name = front[0]
    for i in range(1, len(name)-2):
        suffix = name[i:]
        if not has(suffix, fronts):
             print("Doesn't:", name, suffix)
        