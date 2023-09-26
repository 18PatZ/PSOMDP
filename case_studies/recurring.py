import sys
 
# adding above folder to the system path
sys.path.insert(0, '../')

from mdp import *
import numpy as np
import math



def corridorTwoCadence(n1, n2, cadence1, cadence2, discount = math.sqrt(0.99)):
    goalActionReward = 10000
    noopReward = 0#-1
    wallPenalty = -300000
    movePenalty = -1

    moveProb = 0.9

    grid = [
        [0],
        [0],
        [0],
        [0],
        [0]
    ]

    for i in range(n1):
        for j in range(cadence1-1):
            for k in range(len(grid)):
                grid[k] += [0]
        grid[0] += [0]
        grid[1] += [1]
        grid[2] += [0]
        grid[3] += [1]
        grid[4] += [0]

    for i in range(n2):
        for j in range(cadence2-1):
            for k in range(len(grid)):
                grid[k] += [0]
        grid[0] += [0]
        grid[1] += [1]
        grid[2] += [0]
        grid[3] += [1]
        grid[4] += [0]
    
    grid[0] += [0, 0]
    grid[1] += [0, 0]
    grid[2] += [0, 2]
    grid[3] += [0, 0]
    grid[4] += [0, 0]

    start_state = (1, 2)

    mdp = createMDP(grid, goalActionReward, noopReward, wallPenalty, movePenalty, moveProb)
    return grid, mdp, discount, start_state



def extendHybrid(sched, k):
    strides = list(sched.strides)
    strides.insert(0, k)

    newSched = Schedule(strides = strides, recc_strides = sched.recc_strides, pi_exec_data=None,pi_checkin_data=None,pi_mid_data=None)
    return newSched

def extendRecurring(sched, k):
    recc_strides = list(sched.recc_strides)
    recc_strides.insert(0, k)

    newSched = Schedule(strides = sched.strides, recc_strides = recc_strides, pi_exec_data=None,pi_checkin_data=None,pi_mid_data=None)
    return newSched

def makeBase(k):
    return Schedule(strides = [], recc_strides = [k], pi_exec_data=None,pi_checkin_data=None,pi_mid_data=None)
    

def printHybrid(sched):

    leading = ''.join(list(map(str, sched.strides)))
    recc = ''.join(list(map(str, sched.recc_strides)))
    
    print(f"{leading}({recc})*")


def runComparison(grid, mdp, start_state, discount, discount_checkin):
    additional_schedules = []

    base_strides = [1, 2, 3, 4]
    stride_list = [[k] for k in base_strides]
    l = 3#4

    hybrid_scheds = []
    recc_scheds = [makeBase(k) for k in base_strides]

    print(f"\nSTAGE 0 RECURRING:")
    for sched in recc_scheds:
        printHybrid(sched)

    for stage in range(l-1):
        new_hybrids = []
        new_recc = []

        for k in base_strides:
            for sched in (hybrid_scheds + recc_scheds):
                if stage == 0 and k == sched.recc_strides[0]:
                    continue # don't duplicate recurring tail value (e.g. 23* and 233*)
                
                newSched = extendHybrid(sched, k)
                new_hybrids.append(newSched)

            for sched in recc_scheds:
                if stage == 0 and k == sched.recc_strides[0]:
                    continue # don't duplicate recurring tail value (e.g. 23* and 233*)
                
                newSched = extendRecurring(sched, k)
                new_recc.append(newSched)

        # hybrid_scheds.extend(new_hybrids)
        # recc_scheds.extend(new_recc)
        hybrid_scheds = new_hybrids
        recc_scheds = new_recc

        print(f"\n\nSTAGE {stage+1} HYBRIDS:")
        for sched in new_hybrids:
            printHybrid(sched)

        print(f"\nSTAGE {stage+1} RECURRING:")
        for sched in new_recc:
            printHybrid(sched)


    # all_compMDPs = createCompositeMDPs(mdp, discount, np.max(base_strides))    

    # for strides in stride_list:
    #     _, policy_layers, value_layers, _, _ = runMultiLayer(grid, mdp, discount, start_state, strides=strides, all_compMDPs=all_compMDPs, drawPolicy=False, outputDir="../output")
    #     values = value_layers[0]
    #     eval_normal = gliderScheduleCheckinCostFunctionMulti(strides, discount_checkin)
    #     sched = Schedule(strides=strides, pi_exec_data=(values, eval_normal), pi_checkin_data=None, pi_mid_data=None, is_multi_layer=True)
    #     additional_schedules.append(sched)


    # if True:
    #     runMultiLayer(grid, mdp, discount, start_state, strides=[1, 2, 3], drawPolicy=True, outputDir="../output")
    #     exit()

    # if True:
    #     print(gliderScheduleCheckinCostFunction([1], discount_checkin))
    #     exit()


if __name__ == '__main__':

    # start = time.time()

    grid, mdp, discount, start_state = corridorTwoCadence(n1=3, n2=6, cadence1=2, cadence2=3)
    
    discount_checkin = discount

    runComparison(grid, mdp, start_state, discount, discount_checkin)