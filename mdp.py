
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation

from networkx.drawing.nx_agraph import graphviz_layout, to_agraph

import colorsys
import math

import time

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

def createMDP(grid):

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
                    mdp.rewards[state][action] = noopReward#goalReward # goal infinitely loops onto itself

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

                    addOrSet(mdp.transitions[state][direction], new_state, 0.9)
                    addOrSet(mdp.transitions[state][direction], new_state_left, 0.05)
                    addOrSet(mdp.transitions[state][direction], new_state_right, 0.05)

                    reward = (
                        0.9 * ((wallPenalty if hit_wall else movePenalty)) + 
                        0.05 * ((wallPenalty if hit_wall_left else movePenalty)) + 
                        0.05 * ((wallPenalty if hit_wall_right else movePenalty))
                    )

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


def draw(grid, mdp, policy, policyOnly, name):
    G = nx.MultiDiGraph()

    #G.add_node("A")
    #G.add_node("B")
    #G.add_edge("A", "B")
    for state in mdp.states:
        G.add_node(state)

    #'''
    for y in range(len(grid)):
        for x in range(len(grid[y])):
            state = (x, y)
            state_type = grid[y][x]

            if state_type == TYPE_WALL:
                G.add_node(state)
    #'''

    for begin in mdp.transitions.keys():
        for action in mdp.transitions[begin].keys():
            for end in mdp.transitions[begin][action].keys():
                probability = mdp.transitions[begin][action][end]

                color = fourColor(begin)

                isPolicy = begin in policy and policy[begin] == action
                if isPolicy:
                    if policyOnly and probability >= 0.9:
                        color = "blue"
                    else:
                        color = "black"
                if not policyOnly or isPolicy:
                    G.add_edge(begin, end, prob=probability, label=f"{action}: " + "{:.2f}".format(probability), color=color, fontcolor=color)

    # Build plot
    fig, ax = plt.subplots(figsize=(8, 8))
    #fig.canvas.mpl_connect('key_press_event', on_press)
    #fig.canvas.mpl_connect('button_press_event', onClick)

    # layout = nx.spring_layout(G)
    kDist = dict(nx.shortest_path_length(G))
    #kDist['C']['D'] = 1
    #kDist['D']['C'] = 1
    #kDist['C']['E'] = 1.5
    #layout = nx.kamada_kawai_layout(G, dist=kDist)
    layout = {}

    ax.clear()
    labels = {}
    edge_labels = {}
    color_map = []

    G.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved', 'fontsize':'10'}
    G.graph['graph'] = {'scale': '3'}

    A = to_agraph(G) 

    A.node_attr['style']='filled'

    for node in G.nodes():
        #mass = "{:.2f}".format(G.nodes[node]['mass'])
        labels[node] = f"{stateToStr(node)}"#f"{node}\n{mass}"

        layout[node] = (node[0], -node[1])

        state_type = grid[node[1]][node[0]]

        color = "#FFA500" if state_type == TYPE_STATE else ("#6a0dad" if state_type == TYPE_WALL else "#00FFFF")
        n = A.get_node(node)
        n.attr['color'] = fourColor(node)
        n.attr['fillcolor']=color

        #frac = G.nodes[node]['mass'] / 400
        # col = (0, 0, int(frac * 255))
        #if frac > 1:
        #    frac = 1
        #if frac < 0:
        #    frac = 0
        #col = colorsys.hsv_to_rgb(0.68, frac, 1)
        #col = (int(col[0] * 255), int(col[1] * 255), int(col[2] * 255))
        #col = '#%02x%02x%02x' % col
        color_map.append(color)

    for s, e, d in G.edges(data=True):
        edge_labels[(s, e)] = "{:.2f}".format(d['prob'])

    #nx.draw(G, pos=layout, node_color=color_map, labels=labels, node_size=2500)
    #nx.draw_networkx_edge_labels(G, pos=layout, edge_labels=edge_labels)

    # Set the title
    #ax.set_title("MDP")

    #plt.show()
    m = 1.5
    for k,v in layout.items():
        A.get_node(k).attr['pos']='{},{}!'.format(v[0]*m,v[1]*m)

    #A.layout('dot')                                                                 
    A.layout(prog='neato')
    A.draw(name + '.png')#, prog="neato") 


def valueIteration(grid, mdp, discount, threshold, max_iterations):

    values = {state: (goalReward if grid[state[1]][state[0]] == TYPE_GOAL else stateReward) for state in mdp.states}

    for iteration in range(max_iterations):
        prev_values = values.copy()
        
        for state in mdp.states:
            if grid[state[1]][state[0]] == TYPE_GOAL:
                continue # todo fix this duct tape thing
            max_expected = 0
            for action in mdp.actions:
                expected_value = mdp.rewards[state][action]
                for end_state in mdp.transitions[state][action].keys():
                    prob = mdp.transitions[state][action][end_state]
                    expected_value += discount * prob * prev_values[end_state]

                max_expected = max(max_expected, expected_value)
            values[state] = max_expected
        
        new_values = np.array(list(values.values()))
        old_values = np.array(list(prev_values.values()))
        relative_value_difference = np.linalg.norm(new_values-old_values) / np.linalg.norm(new_values)

        print(f"Iteration {iteration}: {relative_value_difference}")

        if relative_value_difference <= threshold:
            break

    policy = {}
    for state in mdp.states:
        best_action = None
        max_expected = -1
        for action in mdp.actions:
            expected_value = mdp.rewards[state][action]
            for end_state in mdp.transitions[state][action].keys():
                prob = mdp.transitions[state][action][end_state]
                expected_value += discount * prob * values[end_state]

            if expected_value > max_expected:
                best_action = action
                max_expected = expected_value
        policy[state] = best_action

    return policy


grid = [
    [0, 0, 0, 0, 1, 0, 2],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
]

mdp = createMDP(grid)

policy = valueIteration(grid, mdp, 0.9, 1e-20, int(1e4))
print(policy)

draw(grid, mdp, policy, False, "multi")
draw(grid, mdp, policy, True, "policy")

