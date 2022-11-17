from docplex.mp.model import Model
import time

def linearProgrammingSolve(grid, mdp, discount, restricted_action_set = None):

    time_start = time.time()

    lp = Model()
    v = lp.continuous_var_dict(keys = mdp.states, name = "v")

    t1 = time.time()
    print("T1: "+str(t1 - time_start))

    lp.v_sum = lp.sum(v[state] for state in mdp.states)
    objective = lp.minimize(lp.v_sum)

    t2 = time.time()
    print("T2: "+str(t2 - t1))

    timeA = 0
    timeB = 0
    timeC = 0

    for state in mdp.states:
        action_set = mdp.actions if restricted_action_set is None else restricted_action_set[state]
        for action in action_set:
            tA = time.time()
            
            leList = [mdp.transitions[state][action][end_state] * v[end_state] for end_state in mdp.transitions[state][action].keys()]
            tB = time.time()
            timeA += tB - tA
            
            leSum = lp.sum(leList)
            tC = time.time()
            timeB += tC - tB
            
            constraint = lp.add_constraint(v[state] >= (mdp.rewards[state][action] + discount * leSum))
            tD = time.time()
            timeC += tD - tC

    print("  TA: "+str(timeA))
    print("  TB: "+str(timeB))
    print("  TC: "+str(timeC))

    t3 = time.time()
    print("T3: "+str(t3 - t2))

    time_elapsed = (time.time() - time_start)
         
    print("time to create the model: "+str(time_elapsed))
        
    # print("ILP was made") 
        
    # print("------Solve the model--------------")
        
    # print("Solve the ILP\n")

    time_start = time.time()
    lp.solve()

    time_elapsed = (time.time() - time_start)
         
    print("time to solve the model: "+str(time_elapsed))

    # print(lp.solution) 
    # print(lp.solution.get_value_dict(v))

    values = lp.solution.get_value_dict(v)

    policy = {}
    for state in mdp.states:
        best_action = None
        max_expected = None
        
        action_set = mdp.actions if restricted_action_set is None else restricted_action_set[state]
        for action in action_set:
            expected_value = mdp.rewards[state][action]
            for end_state in mdp.transitions[state][action].keys():
                prob = mdp.transitions[state][action][end_state]
                expected_value += discount * prob * values[end_state]

            if max_expected is None or expected_value > max_expected:
                best_action = action
                max_expected = expected_value

        if max_expected is None:
            max_expected = 0
        
        policy[state] = best_action

    return policy, values