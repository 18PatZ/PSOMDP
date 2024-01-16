
class MDP:
    def __init__(self, states, actions, transitions, rewards, terminals):
        self.states = states
        self.actions = actions
        self.transitions = transitions
        self.rewards = rewards
        self.terminals = terminals

    def T(self, state, action, new_state):
        if state not in self.transitions or action not in self.transitions[state] or new_state not in self.transitions[state][action]:
            return 0
        return self.transitions[state][action][new_state]

    def T(self, state, action):
        if state not in self.transitions or action not in self.transitions[state]:
            return []
        return [(new_state, self.transitions[state][action][new_state]) for new_state in self.transitions[state][action].keys()]

    def R(self, state, action):
        if state not in self.rewards or action not in self.rewards[state]:
            return 0
        return self.rewards[state][action]

#goalReward = 100
#stateReward = 0
# goalActionReward = 10000
# noopReward = 0#-1
# wallPenalty = -50000
# movePenalty = -1

TYPE_STATE = 0
TYPE_WALL = 1
TYPE_GOAL = 2

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

cardinals = ["NORTH", "EAST", "SOUTH", "WEST"]
dpos = [(0, -1), (1, 0), (0, 1), (-1, 0)]
def driftAction(actionInd, direction):
    #ind = cardinals.index(action)
    #return cardinals[actionInd + direction]
    return (actionInd + direction + len(cardinals)) % len(cardinals)

def attemptMove(grid, state, dirInd):
    moveto_state = (state[0] + dpos[dirInd][0], state[1] + dpos[dirInd][1])
    new_state, hit_wall = clamp(moveto_state, grid)
    if grid[new_state[1]][new_state[0]] == TYPE_WALL:
        new_state = state
        hit_wall = True
    return new_state, hit_wall

def addOrSet(dictionary, key, val):
    if key in dictionary:
        dictionary[key] += val
    else:
        dictionary[key] = val

def createMDP(grid, goalActionReward, noopReward, wallPenalty, movePenalty, moveProb):

    mdp = MDP([], ["NORTH", "EAST", "SOUTH", "WEST", "NO-OP"], {}, {}, [])

    for y in range(len(grid)):
        for x in range(len(grid[y])):
            state = (x, y)
            state_type = grid[y][x]

            if state_type == TYPE_WALL:
                continue

            mdp.states.append(state)

            mdp.transitions[state] = {}
            for action in mdp.actions:
                mdp.transitions[state][action] = {}

            mdp.rewards[state] = {}

            if state_type == TYPE_GOAL:
                mdp.terminals.append(state)

                for action in mdp.actions:
                    mdp.transitions[state][action][state] = 1
                    mdp.rewards[state][action] = goalActionReward # goal infinitely loops onto itself

            else:
                mdp.transitions[state]["NO-OP"][state] = 1 # no-op loops back to itself
                mdp.rewards[state]["NO-OP"] = noopReward

                for dirInd in range(len(cardinals)):
                    direction = cardinals[dirInd]

                    new_state, hit_wall = attemptMove(grid, state, dirInd)
                    new_state_left, hit_wall_left = attemptMove(grid, new_state, driftAction(dirInd, -1))
                    new_state_right, hit_wall_right = attemptMove(grid, new_state, driftAction(dirInd, 1))

                    hit_wall_left = hit_wall_left or hit_wall
                    hit_wall_right = hit_wall_right or hit_wall

                    #prob = 0.4

                    addOrSet(mdp.transitions[state][direction], new_state, moveProb)
                    addOrSet(mdp.transitions[state][direction], new_state_left, (1 - moveProb)/2)
                    addOrSet(mdp.transitions[state][direction], new_state_right, (1 - moveProb)/2)

                    reward = (
                        moveProb * ((wallPenalty if hit_wall else movePenalty)) +
                        (1 - moveProb)/2 * ((wallPenalty if hit_wall_left else movePenalty)) +
                        (1 - moveProb)/2 * ((wallPenalty if hit_wall_right else movePenalty))
                    )

                    # if y == 5 and x == 2:
                    #     print("\n",state, direction,new_state,hit_wall)
                    #     print("DRIFT LEFT", new_state_left, hit_wall_left)
                    #     print("DRIFT RIGHT", new_state_right, hit_wall_right)
                    #     print("REWARD", reward)

                    mdp.rewards[state][direction] = reward

    return mdp



def stateToStr(state):
    return f"{state[0]}-{state[1]}"


def fourColor(state):
    color = ""
    if state[0] % 2 == 0:
        color = "#880000" if state[1] % 2 == 0 else "#008800"
    else:
        color = "#000088" if state[1] % 2 == 0 else "#888800"
    return color