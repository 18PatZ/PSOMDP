import sys
 
# adding above folder to the system path
sys.path.insert(0, '../')
from mdp import *


#######################
###  GRID EXAMPLES  ###
#######################

def mediumGrid():
    goalActionReward = 10000
    noopReward = 0#-1
    wallPenalty = -50000
    movePenalty = -1

    moveProb = 0.4
    discount = 0.707106781#0.5

    grid = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    start_state = (0, 0)

    mdp = createMDP(grid, goalActionReward, noopReward, wallPenalty, movePenalty, moveProb)
    return grid, mdp, discount, start_state


def largeGrid():
    goalActionReward = 10000
    noopReward = 0#-1
    wallPenalty = -50000
    movePenalty = -1

    moveProb = 0.4
    discount = 0.707106781#0.5

    grid = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    start_state = (0, 0)

    mdp = createMDP(grid, goalActionReward, noopReward, wallPenalty, movePenalty, moveProb)
    return grid, mdp, discount, start_state


def paper2An(n, discount = math.sqrt(0.99)):
    goalActionReward = 10000
    noopReward = 0#-1
    wallPenalty = -300000
    movePenalty = -1

    moveProb = 0.9
    # discount = math.sqrt(0.99)

    # grid = [
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    #     [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # ]
    grid = [
        [0],
        [0],
        [0],
        [0],
        [0]
    ]

    for i in range(n):
        grid[0] += [0, 0, 0]
        grid[1] += [0, 0, 1]
        grid[2] += [0, 0, 0]
        grid[3] += [0, 0, 1]
        grid[4] += [0, 0, 0]
    
    grid[0] += [0, 0, 0]
    grid[1] += [0, 0, 0]
    grid[2] += [0, 0, 2]
    grid[3] += [0, 0, 0]
    grid[4] += [0, 0, 0]

    start_state = (1, 2)

    mdp = createMDP(grid, goalActionReward, noopReward, wallPenalty, movePenalty, moveProb)
    return grid, mdp, discount, start_state


def paper2A():
    return paper2An(3)


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
    
    # grid[0] += [0, 0, 0]
    # grid[1] += [0, 0, 0]
    # grid[2] += [0, 0, 2]
    # grid[3] += [0, 0, 0]
    # grid[4] += [0, 0, 0]
    grid[0] += [0, 0]
    grid[1] += [0, 0]
    grid[2] += [0, 2]
    grid[3] += [0, 0]
    grid[4] += [0, 0]

    start_state = (1, 2)

    mdp = createMDP(grid, goalActionReward, noopReward, wallPenalty, movePenalty, moveProb)
    return grid, mdp, discount, start_state


def splitterGrid(rows = 8, discount = math.sqrt(0.9)):
    goalActionReward = 10000
    noopReward = 0#-1
    wallPenalty = -50000
    movePenalty = -1

    moveProb = 0.9
    # discount = math.sqrt(0.9)

    grid = []
    
    # rows = 8
    p1 = 2
    p2 = 3

    maxN = math.floor(rows / 3)#3
    nL = 0
    nR = 0

    for i in range(rows):
        row = None
        if nL < maxN and i % p1 == 1:#(rows-i) % p1 == 1:
            nL += 1
            row = [0, 1, 0, 1, 0, 1]
        else:
            row = [0, 0, 0, 0, 0, 0]

        row.append(2 if i == 0 else 1)

        if nR < maxN and i % p2 == 1:#(rows-i) % p2 == 1:
            nR += 1
            row += [1, 0, 1, 0, 1, 0]
        else:
            row += [0, 0, 0, 0, 0, 0]
        grid.append(row)
    
    grid.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    grid.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    grid.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    grid.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    start_state = (6, rows+4-1)

    mdp = createMDP(grid, goalActionReward, noopReward, wallPenalty, movePenalty, moveProb)
    return grid, mdp, discount, start_state


def splitterGrid2(rows = 8, discount = math.sqrt(0.9)):
    goalActionReward = 10000
    noopReward = 0#-1
    wallPenalty = -50000
    movePenalty = -1

    moveProb = 0.9
    # discount = math.sqrt(0.9)

    grid = []
    
    # rows = 8
    p1 = 2
    p2 = 3

    maxN = math.floor(rows / 3)#3
    nL = 0
    nR = 0

    grid.append([0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0])

    j = 1
    for i in range(1, rows):
        row = None
        if nL < maxN and j % p1 == 1:#(rows-i) % p1 == 1:
            nL += 1
            row = [0, 1, 0, 1, 0, 1]

            if nL == 2:
                p1 = p2
                j = 1
        else:
            row = [0, 0, 0, 0, 0, 0]

        j += 1

        # row.append(2 if i == 0 else 1)
        row.append(1)

        if nR < maxN and i % p2 == 1:#(rows-i) % p2 == 1:
            nR += 1
            row += [1, 0, 1, 0, 1, 0]
        else:
            row += [0, 0, 0, 0, 0, 0]
        # grid.append(row)
        grid.insert(1, row)
    
    grid.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    grid.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    grid.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    grid.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    start_state = (6, rows+4-1)

    mdp = createMDP(grid, goalActionReward, noopReward, wallPenalty, movePenalty, moveProb)
    return grid, mdp, discount, start_state



#########################
###  PARETO EXAMPLES  ###
#########################




# start = time.time()

# grid, mdp, discount, start_state = paper2An(3)#splitterGrid(rows = 50, discount=0.99)#paper2An(3)#, 0.9999)

grid, mdp, discount, start_state = corridorTwoCadence(n1=3, n2=6, cadence1=2, cadence2=3)
# grid, mdp, discount, start_state = splitterGrid2(rows = 12)
discount_checkin = discount

target_state = findTargetState(grid)

drawChainPolicy(grid, mdp, discount, discount_checkin, start_state, target_state, 
    checkin_periods=[1, 2, 3], 
    chain_checkins=[2,2,3], 
    name="../output/policy-chain-corridor-223*")

if False:
    # start_state = (8, 11)
    # drawChainPolicy(grid, mdp, discount, start_state, target_state, 
    #     checkin_periods=[1, 2, 3, 4], 
    #     chain_checkins=[2,2,1,3], 
    #     name="output/policy-chain-splitter2-2213*")

    # front = ["21*", "221*", "2221*", "22221*", "22112*", "2212*", "11213*", "2213*", "23*", "23334*", "33334*", "43334*", "3334*", "4334*", "434*", "4434*", "44434*"]
    # front = ["3*", "23*", "223*", "2223*", "22223*", "222223*"]
    front = ["2*", "3*"]
    for i in range(11):
        s = ""
        for j in range(i+1):
            s += "2"
        front.append(s + "3*")
    print(front)
    for c in front:
        checkins = []
        for l in c[:-1]:
            checkins.append(int(l))
        drawChainPolicy(grid, mdp, discount, discount_checkin, start_state, target_state, 
            checkin_periods=[1, 2, 3],#, 4], 
            chain_checkins=checkins, 
            # name="../output/policy-chain-splitter-" + c[:-1])
            name="output/policy-chain-corridor-" + c[:-1])
    
    exit()
