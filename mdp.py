
import numpy as np

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
        if state not in self.transitions or action not in self.transitions[state]
            return []
        return [(new_state, self.transitions[state][action][new_state]) for new_state in self.transitions[state][action].keys()]

    def R(self, state, action):
        if state not in self.rewards or action not in self.rewards[state]
            return 0
        return self.rewards[state][action]

goalReward = 100
stateReward = 0
noopReward = 0
wallPenalty = -5
movePenalty = -1

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
def driftAction(action, direction):
    ind = cardinals.index(action)
    return cardinals[ind + direction]

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

def createMDP(grid):

    mdp = MDP([], ["NORTH", "EAST", "SOUTH", "WEST", "NO-OP"], {}, {}, [])

    for y in range(len(grid)):
        for x in range(len(grid[y])):
            state = (x, y)
            state_type = grid[y][x]

            if state_type != TYPE_WALL:
                mdp.states.append(state)

            self.transitions[state] = {}
            for action in mdp.actions:
                self.transitions[state][action] = {}

            self.rewards[state] = {}
            
            if state_type == TYPE_GOAL:
                mdp.terminals.append(state)
                
                for action in mdp.actions:
                    self.transitions[state][action][state] = 1
                    self.rewards[state][action] = goalReward # goal infinitely loops onto itself

            elif:
                self.transitions[state]["NO-OP"][state] = 1 # no-op loops back to itself
                self.rewards[state]["NO-OP"] = noopReward

                for dirInd in range(len(cardinals)):
                    direction = cardinals[dirInd]
                    
                    new_state, hit_wall = attemptMove(grid, state, dirInd)
                    new_state_left, hit_wall_left = attemptMove(grid, new_state, driftAction(dirInd, -1))
                    new_state_right, hit_wall_right = attemptMove(grid, new_state, driftAction(dirInd, 1))

                    hit_wall_left = hit_wall_left || hit_wall
                    hit_wall_right = hit_wall_right || hit_wall

                    addOrSet(self.transitions[state][direction], new_state, 0.9)
                    addOrSet(self.transitions[state][direction], new_state_left, 0.05)
                    addOrSet(self.transitions[state][direction], new_state_right, 0.05)

                    reward = (
                        0.9 * ((wallPenalty if hit_wall else movePenalty) + stateReward) + 
                        0.05 * ((wallPenalty if hit_wall_left else movePenalty) + stateReward) + 
                        0.05 * ((wallPenalty if hit_wall_right else movePenalty) + stateReward)
                    )

                    self.rewards[state][direction] = reward
                    




mdp = createMDP([
    [0, 0, 0, 0, 1, 0, 2],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
])
