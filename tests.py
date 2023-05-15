from mdp import *

# start = time.time()

# grid, mdp, discount, start_state = paper2An(3)#splitterGrid(rows = 50, discount=0.99)#paper2An(3)#, 0.9999)

grid, mdp, discount, start_state = corridorTwoCadence(n1=3, n2=6, cadence1=2, cadence2=3)
# grid, mdp, discount, start_state = splitterGrid2(rows = 12)
discount_checkin = discount

if False:

    scaling_factor = 9.69/1.47e6 # y / x
    midpoints = [getAdjustedAlphaValue(i, scaling_factor) for i in np.arange(0.25, 1, 0.25)]

    print(getAdjustedAlphaValue(i, scaling_factor))

    exit()

    k = 4
    # alpha = 0.000005
    # beta = 1 - alpha
    compMDP = createCompositeMDP(mdp, discount, k)
    checkinMDP = convertCompToCheckinMDP(grid, compMDP, k, discount_checkin)

    discount_t = pow(discount, k)
    discount_c_t = pow(discount_checkin, k)

    policy, values = linearProgrammingSolve(grid, compMDP, discount_t)
    policy_greedy, values_greedy = linearProgrammingSolve(grid, checkinMDP, discount_c_t, restricted_action_set=None, is_negative=True)
    # policy_blend, values_blend = linearProgrammingSolve(grid, blendedMDP, discount_t)

    eval_normal = policyEvaluation(checkinMDP, policy, discount_c_t)
    eval_greedy = policyEvaluation(compMDP, policy_greedy, discount_t)


    initialDistribution = dirac(mdp, start_state)
    point1 = getStateDistributionParetoValues(mdp, (values, eval_normal), [initialDistribution])
    point2 = getStateDistributionParetoValues(mdp, (eval_greedy, values_greedy), [initialDistribution])
    print(point1, point2)

    #scaling_factor = (8.476030558294275 - 6.868081239897704) / (1410952.6446555236 - 1076057.2978729124)
    #scaling_factor = abs((point2[1] - point1[1]) / (point2[0] - point1[0]))
    #scaling_factor = (abs(point2[1] / point2[0]) + abs(point1[1] / point1[0])) / 2
    #print(scaling_factor)
    
    midpoints = [i for i in np.arange(0, 1e-5, 1e-7)]
    #midpoints = [i for i in np.arange(0, 1, 0.25)]
    #midpoints = [i for i in np.arange(0, 1, 0.01)]
    midpoints.append(1)


    file = open(f'output/alpha9.csv', "w")
    file.write("alpha,scaled_alpha,execution,checkin\n")
    for alpha in midpoints:
        desired_value = alpha * point1[0] + (1-alpha) * point2[0]
        desired_checkin = alpha * point1[1] + (1-alpha) * point2[1]

        #scaling_factor = abs(desired_checkin / desired_value)

        #midpoint_alpha = getAdjustedAlphaValue(alpha, scaling_factor)
        midpoint_alpha = alpha
        policy_blend = mixedPolicy(values, values_greedy, compMDP, checkinMDP, midpoint_alpha)

        eval_blend_exec = policyEvaluation(compMDP, policy_blend, discount_t)
        eval_blend_check = policyEvaluation(checkinMDP, policy_blend, discount_c_t)

        point = (eval_blend_exec, eval_blend_check)
        
        vals = getStateDistributionParetoValues(mdp, point, [initialDistribution])
        file.write(f"{alpha},{midpoint_alpha},{vals[0]},{vals[1]}\n")

    file.close()

    exit()


#     v1 = np.array([values[s] for s in mdp.states])
#     v2 = np.array([values_greedy[s] for s in mdp.states])
#     v3 = np.array([values_blend[s] for s in mdp.states])
#     print("Diff A", np.linalg.norm(v1 - v2))

#     v4 = alpha * v1 + beta * v2
#     print("Diff B", np.linalg.norm(v3 - v4))
#     print("Max diff", np.max(np.absolute(v3 - v4)))
#     print(v3 - v4)

#     m = 0
#     for state in mdp.states:
#         if policy_blend[state] != policy[state]:
#             print(state, policy_blend[state], 'vs', policy[state])
#             m += 1
#     print(m,"different policy actions")


#     s_ind = mdp.states.index(start_state)
#     print(v1[s_ind], v2[s_ind], v3[s_ind], v4[s_ind])

#     draw(grid, compMDP, values, policy, True, False, "output/leblend")

#     exit()

# end = time.time()
# print("MDP creation time:", end - start)

# checkin_period = 2
# mdp = convertToGreedyMDP(grid, mdp)

#run(grid, mdp, discount, start_state, checkin_period, doBranchAndBound=False, doLinearProg=False) # VI
# run(grid, mdp, discount, start_state, checkin_period, doBranchAndBound=True, doLinearProg=False) # BNB
# _, policy, _, compMDP = run(grid, mdp, discount, start_state, checkin_period, doBranchAndBound=False, doLinearProg=True) # LP

# run(grid, mdp, discount, start_state, checkin_period, doBranchAndBound=False, doLinearProg=True, 
#     doSimilarityCluster=True, simClusterParams=(7, 1e-5)) # LP w/ similarity clustering

# run(grid, mdp, discount, start_state, checkin_period, doBranchAndBound=True, doLinearProg=True) # BNB w/ LP w/ greedy
# run(grid, mdp, discount, start_state, checkin_period, doBranchAndBound=True, doLinearProg=True, bnbGreedy=800) # BNB w/ LP w/ greedy

# runTwoCadence(2, 3)
target_state = None
for y in range(len(grid)):
    for x in range(len(grid[y])):
        state = (x, y)
        state_type = grid[y][x]

        if state_type == TYPE_GOAL:
            target_state = state
            break

# markov = markovProbsFromPolicy(compMDP, policy)
# hitting_checkins = expectedMarkovHittingTime(mdp, markov, target_state, 1)
# print("Hitting time:", hitting_checkins[mdp.states.index(start_state)])

if False:
    # start_state = (8, 11)
    # drawChainPolicy(grid, mdp, discount, start_state, target_state, 
    #     checkin_periods=[1, 2, 3, 4], 
    #     chain_checkins=[2,2,1,3], 
    #     name="output/policy-chain-splitter2-2213*")

    # front = ["21*", "221*", "2221*", "22221*", "22112*", "2212*", "11213*", "2213*", "23*", "23334*", "33334*", "43334*", "3334*", "4334*", "434*", "4434*", "44434*"]
    # front = ["3*", "23*", "223*", "2223*", "22223*", "222223*"]
    front = ["223*"]
    for c in front:
        checkins = []
        for l in c[:-1]:
            checkins.append(int(l))
        drawChainPolicy(grid, mdp, discount, discount_checkin, start_state, target_state, 
            checkin_periods=[1, 2, 3, 4], 
            chain_checkins=checkins, 
            # name="output/policy-chain-splitter3-" + c[:-1])
            name="output/policy-chain-corridor-" + c[:-1])
    
    exit()
    
# bounding_box = np.array([[-1.5e6, -1.38e6], [0.0001, 30]])
#bounding_box = np.array([[-1.5e6, -1.39e6], [0.0001, 30]])
bounding_box = np.array([[-1.5e6, -1e6], [0.0001, 30]])
# bounding_box = np.array([[-1.56e6, -1.32e6], [0, 30]])
# bounding_box = np.array([[-1.53e6, -1.38e6], [0.0001, 30]])
# bounding_box = np.array([[-32000, -17000], [0, 30]])
# bounding_box = np.array([[-57000, -38000], [0, 30]])

compares = [('Truth', [74.15495896339417, 0]), ('Dirac 0', [24.367187023162842, 0.3009274397414072]), ('Dirac 0.1', [25.151644229888916, 0.26517950151114317]), ('Dirac 0.2', [49.12092685699463, 0.16426632211119754]), ('Gaussian 0', [38.717007875442505, 0.14972084714714468]), ('Gaussian 0.05', [44.45709490776062, 0.0006497689800423529]), ('Gaussian 0.1', [59.4831109046936, 0]), ('Uniform 0', [31.752576112747192, 0.15037592784433365]), ('Uniform 0.05', [46.60546684265137, 0.15037592784433365]), ('Uniform 0.1', [39.49303913116455, 0.0])]


if False:
    x_offset = 1.56e6
    x_scale = 1/1000

    # names = [
    #     "pareto-c4-l32-uniform-filtered-margin0.000-step1",
    #     "pareto-c4-l32-uniform-filtered-margin0.000-step2",
    #     # "pareto-c4-l32-uniform-filtered-margin0.000-step4",
    #     # "pareto-c4-l32-uniform-filtered-margin0.000-step8",
    #     # "pareto-c4-l32-uniform-filtered-margin0.000-step16",
    #     # "pareto-c4-l32-uniform-filtered-margin0.000-step32"
    #     ]

    names = [
        "pareto-c4-l4-uniform-sOriginal-filtered-margin0.040",
    ]

    loffsets = {
        # "2321*": (-5, 0),
        # "43334*": (-10, 0),
        # "2232121*": (-10, 0),

        "1121*": (-20, 0),

        "221*": (5, 5),
        "2221*": (5, 5),
        "2231*": (25, 15),
        "2331*": (30, 20),
        "2341*": (5, 5),
        "2113*": (-10, 0),
        "2234*": (-15, 5),
        "2334*": (-10, 0),
    }

    for name in names:
        chains, indices, is_efficient = loadDataChains(name)
        drawParetoFront(chains, indices, is_efficient,
            true_front = None, 
            true_costs = None, 
            name=name, title="", bounding_box=bounding_box, prints=False, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)

    # names = [
    #     "pareto-c4-l4-uniform-filtered-margin0.040-step1",
    #     "pareto-c4-l4-uniform-filtered-margin0.040-step2",
    #     "pareto-c4-l4-uniform-filtered-margin0.040-step3",
    #     "pareto-c4-l4-uniform-filtered-margin0.040-step4"
    #     ]

    # colors = ["red", "orange", "green", "blue"]
    # stuffs = []
    # outputName = "pareto-c4-l4-uniform-filtered-margin0.040-steps"

    # loffsets = {
    #     "1121*": (0, 10),
    #     "2113*": (0, 10)
    # }

    # for i in range(len(names)):
    #     name = names[i]
    #     chains, is_efficient = loadDataChains(name)
    #     stuffs.append((chains, is_efficient, colors[i]))
    # drawChainsParetoFrontSuperimposed(stuffs,
    #     true_front = TRUTH_C4L4, 
    #     true_costs = TRUTH_COSTS_C4L4, 
    #     name=outputName, bounding_box=bounding_box, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)

    # names = [
    #     "pareto-c4-l4-uniform-filtered-margin0.040-step1",
    #     "pareto-c4-l4-uniform-filtered-margin0.040-step2",
    #     "pareto-c4-l4-uniform-filtered-margin0.040-step3",
    #     "pareto-c4-l4-uniform-filtered-margin0.040-step4"
    #     ]

    # colors = ["blue", "red"]
    # og = None
    # names = [
    #     "pareto-c4-l4-uniform-sOriginal-filtered-margin0.040",
        
    #     # "pareto-c4-l4-uniform-sLeft-filtered-margin0.040",
    #     # "pareto-c4-l4-uniform-sRight-filtered-margin0.040",
    #     # "pareto-c4-l4-uniform-sUp-filtered-margin0.040",
    #     # "pareto-c4-l4-uniform-sDown-filtered-margin0.040",
    #     "pareto-c4-l4-uniform-sCombo-filtered-margin0.040",
    #     ]

    # loffsets = {
    #     # "121*": (-20, -10),
    #     # "4*": (-40, -0)

    #     # "121*": (80, 90),
    #     # "1221*": (80, 90),
    #     # "1231*": (80, 90),
    #     # "1331*": (80, 90),
    #     # "1113*": (-15, 10),
    #     # "1234*": (-5, 0),

    #     # "21*": (-35, -5),
    #     # "2121*": (-25, 0),
    #     # "3234*": (-10, 0),
    #     # "3334*": (-5, 0),
    #     # "31*": (0, 5),
    #     # "321*": (5, 5),
    #     # "3221*": (5, 5),
    #     # "3231*": (25, 15),
    #     # "3331*": (5, 5),
    #     # "3341*": (5, 5),

    #     "1*": (-20, -10),
    #     "1123*": (-20, 10),
    #     "1223*": (-15, 0),
    #     "223*": (-10, 0),
    #     "2234*": (-5, 20),
    #     "3234*": (-5, 0),

    # }

    # for i in range(len(names)):
    #     name = names[i]
    #     chains, is_efficient = loadDataChains(name)
    #     s = (chains, is_efficient, colors[0 if i == 0 else 1])

    #     if i == 0:
    #         og = s

    #     else:
    #         stuffs = [og, s]

    #         drawChainsParetoFrontSuperimposed(stuffs,
    #             true_front = None, 
    #             true_costs = None, 
    #             name=name, bounding_box=bounding_box, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)

    exit()

# drawCompares(compares)

# if True:
#     exit()
if True:
    start_state_index = mdp.states.index(start_state)

    distributions = []
        
    distStart = []
    for i in range(len(mdp.states)):
        distStart.append(1 if i == start_state_index else 0)
    # distributions.append(distStart)

    distributions.append(uniform(mdp))
    # distributions.append(gaussian(mdp, center_state=start_state, sigma=4))
    # distributions.append(gaussian(mdp, center_state=start_state, sigma=10))
    # distributions.append(gaussian(mdp, center_state=target_state, sigma=4))

    initialDistribution = dirac(mdp, start_state)
    # initialDistribution = dirac(mdp, (start_state[0]-1, start_state[1]))
    # initialDistribution = dirac(mdp, (start_state[0]+1, start_state[1]))
    # initialDistribution = dirac(mdp, (start_state[0], start_state[1]-1))
    # initialDistribution = dirac(mdp, (start_state[0], start_state[1]+1))
    initialDistributionCombo = \
        0.5 * np.array(dirac(mdp, start_state)) + \
        0.125 * np.array(dirac(mdp, (start_state[0]-1, start_state[1]))) + \
        0.125 * np.array(dirac(mdp, (start_state[0]+1, start_state[1]))) + \
        0.125 * np.array(dirac(mdp, (start_state[0], start_state[1]-1))) + \
        0.125 * np.array(dirac(mdp, (start_state[0], start_state[1]+1)))

    # distributions.append(initialDistributionCombo)
    # distributions.append(initialDistribution)
    # initialDistribution = initialDistributionCombo

    # margins = [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    # margins = np.arange(0, 0.1001, 0.005)
    # margins = np.arange(0, 0.0501, 0.005)
    # margins = np.arange(0.055, 0.1001, 0.005)
    # margins = np.arange(0.01, 0.0251, 0.005)
    # margins = [0.04]
    margins = [0]
    # margins = [0.015]

    lengths = [4]#[1, 2, 3, 4, 5, 6, 7]

    repeats = 1
    results = []

    # truth_name = "pareto-c4-l4-truth"
    # true_fronts, truth_schedules = loadTruth(truth_name)

    scaling_factor = 9.69/1.47e6 # y / x
    # scaling_factor = (8.476030558294275 - 6.868081239897704) / (1410952.6446555236 - 1076057.2978729124)
    
    #alpha = 0.000006591793283#0.000005
    #beta = 1-alpha
    # midpoints = [
    #     getAdjustedAlphaValue(0.75, scaling_factor),
    #     getAdjustedAlphaValue(0.5, scaling_factor),
    #     getAdjustedAlphaValue(0.25, scaling_factor)
    # ]
    #midpoints = [getAdjustedAlphaValue(i, scaling_factor) for i in np.arange(0.1, 1, 0.1)]
    #midpoints = [getAdjustedAlphaValue(i, scaling_factor) for i in [0.25, 0.375, 0.5, 0.75]]
    # midpoints = [0.25, 0.375, 0.5, 0.75]
    midpoints = []
    # midpoints = [0.2, 0.4, 0.6, 0.8]
    n = 10
    # midpoints = [1.0/(2**x) for x in range(n-1,0,-1)]
    # midpoints = list(np.arange(0.1, 1, 0.1))
    midpoints = [getAdjustedAlphaValue(m, scaling_factor) for m in midpoints]

    alphas_name = "_no-alpha_"
    # alphas_name = "_4alpha_"
    # alphas_name = "_4e-alpha_"
    # alphas_name = "_10alpha_"

    print(midpoints)

    for length in lengths:
        print("\n\n  Running length",length,"\n\n")

        truth_name = f"pareto-c4-l{length}-truth_no-alpha_"#"pareto-c4-l4-truth"
        # truth_name = f"pareto-c4-l32-initial_10alpha_-filtered-margin0.000-step17"
        true_fronts, truth_schedules = loadTruth(truth_name)

        for margin in margins:
            print("\n\n  Running margin",margin,"\n\n")

            running_time_avg = 0
            error = -1
            trimmed = 0

            for i in range(repeats):
                running_time, error, trimmed = runChains(
                    grid, mdp, discount, discount_checkin, start_state, target_state,
                    checkin_periods=[1, 2, 3, 4],
                    chain_length=length,
                    do_filter = True,
                    margin = margin,
                    distName = 'uniform' + alphas_name,
                    startName = '',
                    distributions = distributions, 
                    initialDistribution = initialDistribution,
                    bounding_box = bounding_box, 
                    TRUTH = true_fronts,#TRUTH_C4L4, 
                    TRUTH_COSTS = truth_schedules,#TRUTH_COSTS_C4L4,
                    drawIntermediate=True,
                    midpoints = midpoints)

                running_time_avg += running_time
            running_time_avg /= repeats

            quality = (1 - error) * 100
            quality_upper = (1 - error) * 100
            results.append((margin, running_time_avg, quality, trimmed, quality_upper))
            print("\nRESULTS:\n")
            for r in results:
                print(str(r[0])+","+str(r[1])+","+str(r[2])+","+str(r[3]))


# runCheckinSteps(1, 20)

# runFig2Ratio(210, 300, 10, _discount=0.9999)



# compMDP = createCompositeMDP(mdp, discount, checkin_period)
# print("Actions:",len(mdp.actions),"->",len(compMDP.actions))

# end1 = time.time()
# print("MDP composite time:", end1 - end)

# thresh = 1e-5
# # diffs = checkActionSimilarity(compMDP)
# count, counts = countActionSimilarity(compMDP, thresh)

# end2 = time.time()
# print("Similarity time:", end2 - end1)

# # count, counts = countSimilarity(compMDP, diffs, 2, thresh)
# percTotal = "{:.2f}".format(count / (len(compMDP.states) * len(compMDP.actions)) * 100)
# percStart = "{:.2f}".format(counts[start_state] / (len(compMDP.actions)) * 100)
# print(start_state)
# print(f"{len(compMDP.states)} states {len(compMDP.actions)} actions per state")
# print(f"Pairs under {thresh} total: {count} / {len(compMDP.states) * len(compMDP.actions)} ({percTotal}%)")
# print(f"Pairs under {thresh} in start state: {counts[start_state]} / {len(compMDP.actions)} ({percStart}%)")

# # visualizeActionSimilarity(compMDP, diffs, start_state, f"-{checkin_period}")