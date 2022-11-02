from docplex.mp.model import Model
import time

def linearProgrammingSolve(grid, mdp, discount):

    time_start = time.time()

    lp = Model()
    v = lp.continuous_var_dict(keys = mdp.states, name = "v")

    lp.v_sum = lp.sum(v[state] for state in mdp.states)
    objective = lp.minimize(lp.v_sum)

    for state in mdp.states:
        for action in mdp.actions:
            constraint = lp.add_constraint(v[state] >= (mdp.rewards[state][action] + discount * lp.sum(mdp.transitions[state][action][end_state] * v[end_state] for end_state in mdp.transitions[state][action].keys())))

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
        max_expected = -1e20
        for action in mdp.actions:
            expected_value = mdp.rewards[state][action]
            for end_state in mdp.transitions[state][action].keys():
                prob = mdp.transitions[state][action][end_state]
                expected_value += discount * prob * values[end_state]

            if expected_value > max_expected:
                best_action = action
                max_expected = expected_value
        policy[state] = best_action

    return policy, values