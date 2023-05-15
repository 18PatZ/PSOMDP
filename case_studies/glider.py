import sys
 
# adding above folder to the system path
sys.path.insert(0, '../')

from mdp import *
import numpy as np
import math

def clamp(state, grid):
    x = state[0]
    y = state[1]

    clamped = False

    if x < 0:
        x = 0
        clamped = True
    if x > len(grid[0]) - 1:
        x = len(grid[0]) - 1
        clamped = True
    if y < 0:
        y = 0
        clamped = True
    if y > len(grid) - 1:
        y = len(grid) - 1
        clamped = True

    return (x, y), clamped

def attemptMove(grid, state, displace, driftDir):
    moveto_state = (state[0] + displace[0], state[1] + displace[1] + driftDir)
    new_state, hit_wall = clamp(moveto_state, grid)
    # if grid[new_state[1]][new_state[0]] == TYPE_WALL:
    #     new_state = state
    #     hit_wall = True
    return new_state, hit_wall

def addOrSet(dictionary, key, val):
    if key in dictionary:
        dictionary[key] += val
    else:
        dictionary[key] = val

def multTup(tup, mult):
    return (tup[0] * mult, tup[1] * mult)

def score(old_state, new_state, hit_wall, center_state, wallPenalty, movePenalty, collidePenalty, state_reward_function):
    reward = movePenalty
    
    if hit_wall:
        reward = wallPenalty
    
    if new_state == center_state:
        reward += collidePenalty
        displace = (new_state[0] - old_state[0], new_state[1] - old_state[1])
        if displace[1] != 0:
            new_state = (center_state[0], center_state[1] - np.sign(displace[1])) # side collision
        else:
            new_state = (center_state[0] - np.sign(displace[0]), center_state[1]) # front/rear collision

    reward += state_reward_function(new_state)

    return new_state, reward

def gaussian1D(max, x, sigma):
    gaussian = max * math.exp(- pow(x, 2) / (2 * pow(sigma, 2)))
    return gaussian


def gliderStateRewardFunction(max_reward, state, center_state, desiredSeparation, maxSeparation):
    sigma = maxSeparation * 0.3 # most lies within 3 std devs
    peak = max_reward / 2

    x_diff = state[0] - center_state[0]
    x_reward = gaussian1D(peak, x_diff, sigma)

    y_diff = abs(state[1] - center_state[1]) - desiredSeparation
    y_reward = gaussian1D(peak, y_diff, sigma)

    return x_reward + y_reward



def gliderMDP(maxSeparation = 4, desiredSeparation = 2, moveProb = 0.9, wallPenalty = -10, movePenalty = 0, collidePenalty = -100, desiredStateReward=5):
    gridSize = maxSeparation * 2 + 1

    # grid = [
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 2, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0],
    # ]

    centerInd = int(gridSize / 2)

    grid = [[0 for x in range(gridSize)] for y in range(gridSize)]

    center_state = (centerInd, centerInd)

    start_state = (centerInd, centerInd - desiredSeparation - 1)

    # we have the glider always moving to the right. "Left" means it goes diagonally. TODO variable speeds
    glider1_actions = {
        "LEFT": (1, -1), 
        "FORWARD": (1, 0), 
        "DOUBLE": (2, 0), 
        "RIGHT": (1, 1)
    }

    # the world is centered around glider 2: states represent relative position of g1 to g2. thus g2 actions inverse g1 actions
    glider2_actions = {key: multTup(glider1_actions[key], -1) for key in glider1_actions}

    actions = {}
    for action1 in glider1_actions:
        for action2 in glider2_actions:
            combined_name = f"{action1}-{action2}"
            a1 = glider1_actions[action1]
            a2 = glider2_actions[action2]
            combined_action = (a1[0] + a2[0], a1[1] + a2[1])

            actions[combined_name] = combined_action

    mdp = MDP([], list(actions.keys()), {}, {}, [])

    state_reward_function = lambda state: gliderStateRewardFunction(desiredStateReward, state, center_state, desiredSeparation, maxSeparation)

    for y in range(len(grid)):
        for x in range(len(grid[y])):
            state = (x, y)
            # state_type = grid[y][x]

            # if state_type == TYPE_WALL:
            #     continue

            mdp.states.append(state)

            mdp.transitions[state] = {action: {} for action in mdp.actions}
            mdp.rewards[state] = {}

            for action in actions:
                displace = actions[action]
                new_state, hit_wall = attemptMove(grid, state, displace, driftDir=0)
                new_state_left, hit_wall_left = attemptMove(grid, state, displace, driftDir=-1)
                new_state_right, hit_wall_right = attemptMove(grid, state, displace, driftDir=1)

                hit_wall_left = hit_wall_left or hit_wall
                hit_wall_right = hit_wall_right or hit_wall

                new_state, reward_main = score(state, new_state, hit_wall, 
                                               center_state, wallPenalty, movePenalty, collidePenalty, state_reward_function)
                new_state_left, reward_left = score(state, new_state_left, hit_wall_left, 
                                               center_state, wallPenalty, movePenalty, collidePenalty, state_reward_function)
                new_state_right, reward_right = score(state, new_state_right, hit_wall_right, 
                                               center_state, wallPenalty, movePenalty, collidePenalty, state_reward_function)

                addOrSet(mdp.transitions[state][action], new_state, moveProb)
                addOrSet(mdp.transitions[state][action], new_state_left, (1 - moveProb)/2)
                addOrSet(mdp.transitions[state][action], new_state_right, (1 - moveProb)/2)

                reward = (
                    moveProb * reward_main +
                    (1 - moveProb)/2 * reward_left +
                    (1 - moveProb)/2 * reward_right
                )

                mdp.rewards[state][action] = reward

    return grid, mdp, start_state



# start = time.time()

grid, mdp, start_state = gliderMDP(
    maxSeparation = 4, 
    desiredSeparation = 2, 
    moveProb = 0.9, 
    wallPenalty = -10, 
    movePenalty = 0, 
    collidePenalty = -100, 
    desiredStateReward=5)

discount = math.sqrt(0.99)
discount_checkin = discount


# run(grid, mdp, discount, start_state, checkin_period=2, doBranchAndBound=False, 
#         drawPolicy=True, drawIterations=True, outputPrefix="", doLinearProg=True, 
#         bnbGreedy=-1, doSimilarityCluster=False, simClusterParams=None, outputDir="../output")

# if True:
#     exit()

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
                grid, mdp, discount, discount_checkin, start_state,
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