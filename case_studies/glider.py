import sys
 
# adding above folder to the system path
sys.path.insert(0, '../')

from mdp import *

# start = time.time()

grid, mdp, discount, start_state = corridorTwoCadence(n1=3, n2=6, cadence1=2, cadence2=3)
discount_checkin = discount


target_state = None
for y in range(len(grid)):
    for x in range(len(grid[y])):
        state = (x, y)
        state_type = grid[y][x]

        if state_type == TYPE_GOAL:
            target_state = state
            break

bounding_box = np.array([[-1.5e6, -1e6], [0.0001, 30]])

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

# margins = np.arange(0.01, 0.0251, 0.005)
# margins = [0.04]
margins = [0]
# margins = [0.015]

lengths = [4]#[1, 2, 3, 4, 5, 6, 7]

repeats = 1
results = []

scaling_factor = 9.69/1.47e6 # y / x
# scaling_factor = (8.476030558294275 - 6.868081239897704) / (1410952.6446555236 - 1076057.2978729124)

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
    true_fronts, truth_schedules = loadTruth(truth_name, outputDir="../output")

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
                TRUTH = true_fronts,
                TRUTH_COSTS = truth_schedules,
                drawIntermediate=True,
                midpoints = midpoints, 
                outputDir = "../output")

            running_time_avg += running_time
        running_time_avg /= repeats

        quality = (1 - error) * 100
        quality_upper = (1 - error) * 100
        results.append((margin, running_time_avg, quality, trimmed, quality_upper))
        print("\nRESULTS:\n")
        for r in results:
            print(str(r[0])+","+str(r[1])+","+str(r[2])+","+str(r[3]))