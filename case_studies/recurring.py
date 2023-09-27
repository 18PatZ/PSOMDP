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



def extendHybrid(discount, compMDPs, sched, k):
    strides = list(sched.strides)
    strides.insert(0, k)

    # newSched = Schedule(strides = strides, recc_strides = sched.recc_strides, pi_exec_data=None,pi_checkin_data=None,pi_mid_data=None)

    compMDP = compMDPs[k]

    discount_t = pow(discount, k)
    
    old_values = sched.pi_exec_data[0]
    new_values = runOneValueIterationPass(old_values, discount_t, compMDP)
    policy = policyFromValues(compMDP, new_values, discount_t)

    opt_policies = list(sched.opt_policies)
    opt_policies.insert(0, policy)

    opt_values = list(sched.opt_values)
    opt_values.insert(0, new_values)
    
    new_sched = Schedule(
        strides=strides, 
        recc_strides = sched.recc_strides,
        is_multi_layer=True,
        pi_exec_data=(new_values, None), 
        pi_checkin_data=None, pi_mid_data=None,
        opt_policies=opt_policies,
        opt_values=opt_values)
    
    return new_sched


def extendRecurring(grid, mdp, discount, start_state, all_compMDPs, sched, k):
    recc_strides = list(sched.recc_strides)
    recc_strides.insert(0, k)

    _, policy_layers, value_layers, _, _ = runMultiLayer(grid, mdp, discount, start_state, strides=recc_strides, all_compMDPs=all_compMDPs, drawPolicy=False, outputDir="../output")

    newSched = Schedule(
        strides = [], 
        recc_strides = recc_strides, 
        is_multi_layer=True,
        pi_exec_data=(value_layers[0], None),
        pi_checkin_data=None,pi_mid_data=None, 
        opt_policies=[policy_layers], 
        opt_values=[value_layers])
    
    return newSched

def makeBase(discount, compMDPs, k):
    discount_t = pow(discount, k)
    compMDP = compMDPs[k]

    policy, values = linearProgrammingSolve(compMDP, discount_t)
    return Schedule(strides = [], recc_strides = [k], is_multi_layer=True, pi_exec_data=(values, None), pi_checkin_data=None, pi_mid_data=None, opt_policies=[policy], opt_values=[values])
    # return Schedule(strides = [], recc_strides = [k], pi_exec_data=None,pi_checkin_data=None,pi_mid_data=None)
    

def printHybrid(sched):
    print(sched.to_str())


def runComparison(grid, mdp, start_state, discount, discount_checkin):
    additional_schedules = []

    base_strides = [1, 2, 3]#, 4]
    l = 6

    all_compMDPs = createCompositeMDPs(mdp, discount, np.max(base_strides)) 
    compMDPs = {k: all_compMDPs[k - 1] for k in base_strides}

    stages = []

    l1 = time.time()
    hybrid_scheds = []
    recc_scheds = [makeBase(discount, compMDPs, k) for k in base_strides]
    print("Base took", time.time() - l1, "seconds")

    stages.append((hybrid_scheds, recc_scheds))

    print(f"\nSTAGE 0 RECURRING:")
    for sched in recc_scheds:
        printHybrid(sched)

    total_hybrid = 0
    total_recc = 0

    for stage in range(l-1):
        new_hybrids = []
        new_recc = []

        recc_time = 0
        hybrid_time = 0

        for k in base_strides:
            for sched in (hybrid_scheds + recc_scheds):
                if stage == 0 and k == sched.recc_strides[0]:
                    continue # don't duplicate recurring tail value (e.g. 23* and 233*)
                
                l1 = time.time()
                newSched = extendHybrid(discount, compMDPs, sched, k)
                new_hybrids.append(newSched)
                print(f"    HYB {sched.to_str()} -> {newSched.to_str()} took {time.time() - l1} seconds")
                hybrid_time += time.time() - l1
                total_hybrid += time.time() - l1

            for sched in recc_scheds:
                if stage == 0 and k == sched.recc_strides[0]:
                    continue # don't duplicate recurring tail value (e.g. 23* and 233*)
                
                l1 = time.time()
                newSched = extendRecurring(grid, mdp, discount, start_state, all_compMDPs, sched, k)
                new_recc.append(newSched)
                print(f"    REC {sched.to_str()} -> {newSched.to_str()} took {time.time() - l1} seconds")
                recc_time += time.time() - l1
                total_recc += time.time() - l1

        # hybrid_scheds.extend(new_hybrids)
        # recc_scheds.extend(new_recc)
        hybrid_scheds = new_hybrids
        recc_scheds = new_recc

        print(f"\n\nSTAGE {stage+1} HYBRID: ({len(new_hybrids)}) TOOK {hybrid_time}")
        for sched in new_hybrids:
            printHybrid(sched)

        print(f"\nSTAGE {stage+1} RECURR: ({len(new_recc)}) TOOK {recc_time}")
        for sched in new_recc:
            printHybrid(sched)

        stages.append((hybrid_scheds, recc_scheds))

    print(f"TOTAL HYBRID TIME: {total_hybrid} SECONDS")
    print(f"TOTAL RECURR TIME: {total_recc} SECONDS")

    # start = time.time()

    # print("\n\nRunning")
    # sched = new_recc[0]
    # printHybrid(sched)
    # _, policy_layers, value_layers, _, _ = runMultiLayer(grid, mdp, discount, start_state, strides=sched.recc_strides, all_compMDPs=all_compMDPs, drawPolicy=False, outputDir="../output")
    
    # elapsed = time.time() - start

    # print("elapsed",elapsed)
    

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