
from tabnanny import check
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation

from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

import pandas as pd
import json

from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
from matplotlib.font_manager import FontProperties
from matplotlib import rc


import colorsys
import math

import os

import time
import math

from lp import linearProgrammingSolve
from figure import drawParetoFront, loadDataChains

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


def convertSingleStepMDP(mdp):
    compMDP = MDP([], [], {}, {}, [])

    compMDP.states = mdp.states.copy()
    compMDP.terminals = mdp.terminals.copy()

    for action in mdp.actions:
        compMDP.actions.append((action,)) # 1-tuple

    for state in mdp.transitions.keys():
        compMDP.transitions[state] = {}
        for action in mdp.transitions[state].keys():
            compMDP.transitions[state][(action,)] = {}
            for end_state in mdp.transitions[state][action].keys():
                prob = mdp.transitions[state][action][end_state]
                compMDP.transitions[state][(action,)][end_state] = prob

    for state in mdp.rewards.keys():
        compMDP.rewards[state] = {}
        for action in mdp.rewards[state].keys():
            reward = mdp.rewards[state][action]
            compMDP.rewards[state][(action,)] = reward

    return compMDP

def createCompositeMDP(mdp, discount, checkin_period):
    if checkin_period == 1:
        return convertSingleStepMDP(mdp)

    prevPeriodMDP = createCompositeMDP(mdp, discount, checkin_period-1)
    return extendCompositeMDP(mdp, discount, prevPeriodMDP)

def createCompositeMDPs(mdp, discount, checkin_period):
    mdps = []
    prevPeriodMDP = None
    for c in range(1, checkin_period + 1):
        if c == 1:
            prevPeriodMDP = convertSingleStepMDP(mdp)
        else:
            prevPeriodMDP = extendCompositeMDP(mdp, discount, prevPeriodMDP)
        mdps.append(prevPeriodMDP)
    return mdps

def extendCompositeMDP(mdp, discount, prevPeriodMDP, restricted_action_set = None):
    compMDP = MDP([], [], {}, {}, [])

    compMDP.states = mdp.states.copy()
    compMDP.terminals = mdp.terminals.copy()

    for action_sequence in prevPeriodMDP.actions:
        for action in mdp.actions:
            action_tuple = action if type(action) is tuple else (action,)
            extended_action_sequence = action_sequence + action_tuple # extend tuple
            compMDP.actions.append(extended_action_sequence)

    for state in prevPeriodMDP.transitions.keys():
        compMDP.transitions[state] = {}
        for prev_action_sequence in prevPeriodMDP.transitions[state].keys():
            if restricted_action_set is not None and prev_action_sequence not in restricted_action_set[state]:
                continue
            
            for end_state in prevPeriodMDP.transitions[state][prev_action_sequence].keys():
                # looping through every state-actionsequence-state chain in the previous step MDP
                # now extend chain by one action by multiplying transition probability of previous chain end state to new end state through action

                for action in mdp.actions:
                    action_tuple = action if type(action) is tuple else (action,)
                    prob_chain = prevPeriodMDP.transitions[state][prev_action_sequence][end_state]

                    if end_state in mdp.transitions and action in mdp.transitions[end_state]:
                        for new_end_state in mdp.transitions[end_state][action].keys():
                            prob_additional = mdp.transitions[end_state][action][new_end_state]

                            extended_action_sequence = prev_action_sequence + action_tuple

                            extended_prob = prob_chain * prob_additional

                            if extended_action_sequence not in compMDP.transitions[state]:
                                compMDP.transitions[state][extended_action_sequence] = {}
                            if new_end_state not in compMDP.transitions[state][extended_action_sequence]:
                                compMDP.transitions[state][extended_action_sequence][new_end_state] = 0

                            # the same action sequence might diverge to two different states then converge again, so sum probabilities
                            compMDP.transitions[state][extended_action_sequence][new_end_state] += extended_prob

    for state in prevPeriodMDP.rewards.keys():
        compMDP.rewards[state] = {}
        for prev_action_sequence in prevPeriodMDP.rewards[state].keys():
            if restricted_action_set is not None and prev_action_sequence not in restricted_action_set[state]:
                continue

            prev_reward = prevPeriodMDP.rewards[state][prev_action_sequence]

            for action in mdp.actions:
                if action in mdp.rewards[end_state]:
                    # extend chain by one action
                    action_tuple = action if type(action) is tuple else (action,)
                    extended_action_sequence = prev_action_sequence + action_tuple

                    extension_reward = 0

                    for end_state in prevPeriodMDP.transitions[state][prev_action_sequence].keys():
                        if end_state in mdp.rewards:
                            # possible end states of the chain
                            prob_end_state = prevPeriodMDP.transitions[state][prev_action_sequence][end_state] # probability that chain ends in this state
                            extension_reward += prob_end_state * mdp.rewards[end_state][action]

                    step = len(prev_action_sequence)
                    discount_factor = pow(discount, step)
                    extended_reward = prev_reward + discount_factor * extension_reward
                    compMDP.rewards[state][extended_action_sequence] = extended_reward

    return compMDP


def draw(grid, mdp, values, policy, policyOnly, drawMinorPolicyEdges, name):

    max_value = None
    min_value = None

    if len(values) > 0:
        min_value = min(values.values())
        max_value = max(values.values())

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

            maxProb = -1
            maxProbEnd = None

            isPolicy = begin in policy and policy[begin] == action

            if not policyOnly or isPolicy:
                for end in mdp.transitions[begin][action].keys():
                    probability = mdp.transitions[begin][action][end]

                    if probability > maxProb:
                        maxProb = probability
                        maxProbEnd = end

                for end in mdp.transitions[begin][action].keys():
                    probability = mdp.transitions[begin][action][end]

                    color = fourColor(begin)

                    if isPolicy:
                        color = "grey"
                        if maxProbEnd is not None and end == maxProbEnd:
                            color = "blue"
                        #if policyOnly and probability >= 0.3:#0.9:
                        #    color = "blue"
                        #else:
                        #    color = "black"
                    if not policyOnly or drawMinorPolicyEdges or (maxProbEnd is None or end == maxProbEnd):
                        G.add_edge(begin, end, prob=probability, label=f"{action}: " + "{:.2f}".format(probability), color=color, fontcolor=color)

            # if policyOnly and maxProbEnd is not None:
            #     color = "blue"
            #     G.remove_edge(begin, maxProbEnd)
            #     G.add_edge(begin, maxProbEnd, prob=maxProb, label=f"{action}: " + "{:.2f}".format(maxProb), color=color, fontcolor=color)

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

    G.graph['edge'] = {'arrowsize': '0.6', 'fontsize':'10'}
    G.graph['graph'] = {'scale': '3', 'splines': 'true'}

    A = to_agraph(G)

    A.node_attr['style']='filled'

    for node in G.nodes():
        #mass = "{:.2f}".format(G.nodes[node]['mass'])
        labels[node] = f"{stateToStr(node)}"#f"{node}\n{mass}"

        layout[node] = (node[0], -node[1])

        state_type = grid[node[1]][node[0]]

        n = A.get_node(node)
        n.attr['color'] = fourColor(node)

        if state_type != TYPE_WALL:
            n.attr['xlabel'] = "{:.4f}".format(values[node])

        color = None
        if state_type == TYPE_WALL:
            color = "#6a0dad"
        elif min_value is None and state_type == TYPE_GOAL:
            color = "#00FFFF"
        elif min_value is None:
            color = "#FFA500"
        else:
            value = values[node]
            frac = (value - min_value) / (max_value - min_value)
            hue = frac * 250.0 / 360.0 # red 0, blue 1

            col = colorsys.hsv_to_rgb(hue, 1, 1)
            col = (int(col[0] * 255), int(col[1] * 255), int(col[2] * 255))
            color = '#%02x%02x%02x' % col

            # if node == (2, 5) or state_type == TYPE_GOAL:
            #     print(value)

        n.attr['fillcolor'] = color

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




def drawBNBIteration(grid, mdp, ratios, upperBounds, lowerBounds, pruned, iteration, name):
    G = nx.MultiDiGraph()

    for state in mdp.states:
        G.add_node(state)

    for y in range(len(grid)):
        for x in range(len(grid[y])):
            state = (x, y)
            state_type = grid[y][x]

            if state_type == TYPE_WALL:
                G.add_node(state)

    upper_policy = None
    lower_policy = None
    pruned_q = None

    if iteration < len(upperBounds):
        upper_policy, upper_state_values = extractPolicyFromQ(mdp, upperBounds[iteration], mdp.states, {state: upperBounds[iteration][state].keys() for state in mdp.states})
    if iteration < len(lowerBounds):
        lower_policy, lower_state_values = extractPolicyFromQ(mdp, lowerBounds[iteration], mdp.states, {state: lowerBounds[iteration][state].keys() for state in mdp.states})
    # if iteration < len(pruned):
    #     pruned_q = pruned[iteration]


    for begin in mdp.transitions.keys():
        for action in mdp.transitions[begin].keys():

            maxProb = -1
            maxProbEnd = None

            action_prefix = action[:(iteration+1)]

            isUpperPolicy = upper_policy is not None and begin in upper_policy and upper_policy[begin] == action_prefix
            isLowerPolicy = lower_policy is not None and begin in lower_policy and lower_policy[begin] == action_prefix
            isPruned = pruned_q is not None and begin in pruned_q and action_prefix in pruned_q[begin]

            if isUpperPolicy or isLowerPolicy or isPruned:
                for end in mdp.transitions[begin][action].keys():
                    probability = mdp.transitions[begin][action][end]

                    if probability > maxProb:
                        maxProb = probability
                        maxProbEnd = end

                for end in mdp.transitions[begin][action].keys():
                    probability = mdp.transitions[begin][action][end]

                    color = fourColor(begin)

                    if isUpperPolicy:
                        color = "blue"
                    if isLowerPolicy:
                        color = "green"
                    if isPruned:
                        color = "red"
                    if maxProbEnd is None or end == maxProbEnd:
                        G.add_edge(begin, end, prob=probability, label=f"{action}: " + "{:.2f}".format(probability), color=color, fontcolor=color)

    # Build plot
    fig, ax = plt.subplots(figsize=(8, 8))

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
        labels[node] = f"{stateToStr(node)}"

        layout[node] = (node[0], -node[1])

        state_type = grid[node[1]][node[0]]

        n = A.get_node(node)
        n.attr['color'] = fourColor(node)

        if node in ratios[iteration]:
            n.attr['xlabel'] = "{:.2f}".format(ratios[iteration][node])

        color = None
        if state_type == TYPE_WALL:
            color = "#6a0dad"
        elif state_type == TYPE_GOAL:
            color = "#00FFFF"
        else:
            value = ratios[iteration][node]
            frac = value
            hue = frac * 250.0 / 360.0 # red 0, blue 1

            col = colorsys.hsv_to_rgb(hue, 1, 1)
            col = (int(col[0] * 255), int(col[1] * 255), int(col[2] * 255))
            color = '#%02x%02x%02x' % col

        n.attr['fillcolor'] = color

        color_map.append(color)

    for s, e, d in G.edges(data=True):
        edge_labels[(s, e)] = "{:.2f}".format(d['prob'])

    m = 1.5
    for k,v in layout.items():
        A.get_node(k).attr['pos']='{},{}!'.format(v[0]*m,v[1]*m)

    #A.layout('dot')
    A.layout(prog='neato')
    A.draw(name + '.png')#, prog="neato")


def valueIteration(grid, mdp, discount, threshold, max_iterations):

    #values = {state: (goalReward if grid[state[1]][state[0]] == TYPE_GOAL else stateReward) for state in mdp.states}
    values = {state: 0 for state in mdp.states}

    statesToIterate = []
    # order starting from goal nodes
    for state in mdp.states:
        if grid[state[1]][state[0]] == TYPE_GOAL:
            statesToIterate.append(state)

    # add rest
    for state in mdp.states:
        if grid[state[1]][state[0]] != TYPE_GOAL:
            statesToIterate.append(state)

    # print("states to iterate", len(statesToIterate), "vs",len(mdp.states))

    for iteration in range(max_iterations):
        prev_values = values.copy()

        for state in statesToIterate:
            max_expected = -1e20
            for action in mdp.actions:
                expected_value = mdp.rewards[state][action]
                future_value = 0

                for end_state in mdp.transitions[state][action].keys():
                    prob = mdp.transitions[state][action][end_state]
                    future_value += discount * prob * prev_values[end_state]

                # if state == (2,5):
                #     print(action,"action reward",expected_value)
                #     print(action,"future reward",future_value)
                #     print(action,"total value",expected_value)

                expected_value += future_value

                max_expected = max(max_expected, expected_value)
            values[state] = max_expected

        new_values = np.array(list(values.values()))
        old_values = np.array(list(prev_values.values()))
        relative_value_difference = np.linalg.norm(new_values-old_values) / np.linalg.norm(new_values)

        print(f"Iteration {iteration}: {relative_value_difference}")

        if relative_value_difference <= threshold:
            break

    policy = {}
    for state in statesToIterate:
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


def extractPolicyFromQ(mdp, values, statesToIterate, restricted_action_set):
    policy = {}
    state_values = {}
    for state in statesToIterate:
        best_action = None
        max_expected = None
        action_set = mdp.actions if restricted_action_set is None else restricted_action_set[state]
        for action in action_set:
            expected_value = values[state][action]

            if max_expected is None or expected_value > max_expected:
                best_action = action
                max_expected = expected_value

        if max_expected is None:
            max_expected = 0

        policy[state] = best_action
        state_values[state] = max_expected

    return policy, state_values


def qValueIteration(grid, mdp, discount, threshold, max_iterations, restricted_action_set = None):

    values = {state: {action: 0 for action in mdp.transitions[state].keys()} for state in mdp.states}
    state_values = {state: None for state in mdp.states}

    statesToIterate = []
    # order starting from goal nodes
    for state in mdp.states:
        if grid[state[1]][state[0]] == TYPE_GOAL:
            statesToIterate.append(state)

    # add rest
    for state in mdp.states:
        if grid[state[1]][state[0]] != TYPE_GOAL:
            statesToIterate.append(state)

    # print("states to iterate", len(statesToIterate), "vs",len(mdp.states))

    for iteration in range(max_iterations):
        start = time.time()
        prev_state_values = state_values.copy() # this is only a shallow copy
        # old_values = np.array(list([np.max(list(values[state].values())) for state in mdp.states]))

        for state in statesToIterate:
            action_set = mdp.actions if restricted_action_set is None else restricted_action_set[state]
            for action in action_set:
                expected_value = mdp.rewards[state][action]
                future_value = 0

                for end_state in mdp.transitions[state][action].keys():
                    prob = mdp.transitions[state][action][end_state]

                    # maxQ = None
                    # for action2 in mdp.actions:
                    #     q = values[end_state][action2] # supposed to use previous values?
                    #     if maxQ is None or q > maxQ:
                    #         maxQ = q

                    maxQ = prev_state_values[end_state]
                    if maxQ is None:
                        maxQ = 0

                    future_value += discount * prob * maxQ

                expected_value += future_value

                values[state][action] = expected_value

                prevMaxQ = state_values[state]

                # if state == (1,2):
                #     print("STATE",state,"ACTION",action,"REWARD",mdp.rewards[state][action],"FUTURE",future_value,"Q",expected_value,"PREVMAX",prevMaxQ)

                if prevMaxQ is None or expected_value > prevMaxQ:
                    state_values[state] = expected_value

        # new_values = np.array(list([np.max(list(values[state].values())) for state in mdp.states]))
        new_values = np.array([0 if v is None else v for v in state_values.values()])
        old_values = np.array([0 if v is None else v for v in prev_state_values.values()])
        relative_value_difference = np.linalg.norm(new_values-old_values) / np.linalg.norm(new_values)

        end = time.time()
        print(f"Iteration {iteration}: {relative_value_difference}. Took",end-start)

        if relative_value_difference <= threshold:
            break

    # policy = {}
    # state_values = {}
    # for state in statesToIterate:
    #     best_action = None
    #     max_expected = None
    #     action_set = mdp.actions if restricted_action_set is None else restricted_action_set[state]
    #     for action in action_set:
    #         expected_value = values[state][action]

    #         if max_expected is None or expected_value > max_expected:
    #             best_action = action
    #             max_expected = expected_value

    #     if max_expected is None:
    #         max_expected = 0

    #     policy[state] = best_action
    #     state_values[state] = max_expected

    policy, state_values = extractPolicyFromQ(mdp, values, statesToIterate, restricted_action_set)
    return policy, state_values, values

def qValuesFromR(mdp, discount, state_values, restricted_action_set = None):
    q_values = {state: {action: 0 for action in mdp.transitions[state].keys()} for state in mdp.states}

    for state in mdp.states:
        action_set = mdp.actions if restricted_action_set is None else restricted_action_set[state]
        for action in action_set:
            expected_value = mdp.rewards[state][action]
            future_value = 0

            for end_state in mdp.transitions[state][action].keys():
                prob = mdp.transitions[state][action][end_state]

                maxQ = state_values[end_state]
                if maxQ is None:
                    maxQ = 0

                future_value += discount * prob * maxQ

            expected_value += future_value

            q_values[state][action] = expected_value

    return q_values


def branchAndBound(grid, base_mdp, discount, checkin_period, threshold, max_iterations, doLinearProg=False, greedy=-1):

    compMDP = convertSingleStepMDP(base_mdp)
    pruned_action_set = {state: set([action for action in compMDP.actions]) for state in base_mdp.states}

    upperBound = None
    lowerBound = None

    ratios = []
    upperBounds = []
    lowerBounds = []
    pruned = []
    compMDPs = []

    for t in range(1, checkin_period+1):
        start = time.time()
        if t > 1:
            # compMDP.actions = pruned_action_set
            compMDP = extendCompositeMDP(base_mdp, discount, compMDP, pruned_action_set)
            # pruned_action_set = compMDP.actions

            for state in base_mdp.states:
                extended_action_set = set()
                for prev_action_sequence in pruned_action_set[state]:
                    for action in base_mdp.actions:
                        extended_action_set.add(prev_action_sequence + (action,))
                pruned_action_set[state] = extended_action_set

        if t >= checkin_period:
            break

        if checkin_period % t == 0: # is divisor
            # restricted_action_set = [action[:t] for action in compMDP.actions]
            # og_action_set = compMDP.actions
            # compMDP.actions = restricted_action_set

            # policy, values, q_values = qValueIteration(grid, compMDP, discount, threshold, max_iterations)

            # upperBound = {state: {} for state in mdp.states}
            # for state in compMDP.states:
            #     for action in compMDP.actions:
            #         prefix = action[:t]
            #         q_value = q_values[state][action]
            #         if prefix not in upperBound[state] or q_value > upperBound[state][prefix]: # max
            #             upperBound[state][prefix] = q_value

            discount_input = pow(discount, t)
            if doLinearProg:
                policy, values = linearProgrammingSolve(grid, compMDP, discount_input, pruned_action_set)
                q_values = qValuesFromR(compMDP, discount_input, values, pruned_action_set)
            else:
                policy, values, q_values = qValueIteration(grid, compMDP, discount_input, threshold, max_iterations, pruned_action_set)
            upperBound = q_values

        else: # extend q-values?
            newUpper = {state: {} for state in base_mdp.states}
            for state in compMDP.states:
                for action in compMDP.actions:
                    if action not in pruned_action_set[state]:
                        continue
                    prefix = action[:t]
                    prev_prefix = action[:(t-1)]

                    if prev_prefix in upperBound[state]:
                        newUpper[state][prefix] = upperBound[state][prev_prefix]
            upperBound = newUpper

        discount_input = pow(discount, checkin_period)
        if doLinearProg:
            policy, state_values = linearProgrammingSolve(grid, compMDP, discount_input, pruned_action_set)
            q_values = qValuesFromR(compMDP, discount_input, state_values, pruned_action_set)
        else:
            policy, state_values, q_values = qValueIteration(grid, compMDP, discount_input, threshold, max_iterations, pruned_action_set)
        lowerBound = state_values

        upperBounds.append(upperBound)
        lowerBounds.append(q_values)

        pr = {}

        tot = 0
        for state in base_mdp.states:
            toPrune = []
            action_vals = {}
            
            for action in pruned_action_set[state]:
                prefix = action[:t]
                # print(prefix, upperBound[state][prefix], lowerBound[state])
                if upperBound[state][prefix] < lowerBound[state]:
                    toPrune.append(prefix)
                else:
                    action_vals[action] = upperBound[state][prefix]

            if greedy > -1 and len(action_vals) > greedy:
                sorted_vals = sorted(action_vals.items(), key=lambda item: item[1], reverse=True)
                for i in range(greedy, len(sorted_vals)):
                    action = sorted_vals[i][0]
                    toPrune.append(action[:t])

            # print("BnB pruning",len(toPrune),"/",len(pruned_action_set[state]),"actions")
            pruned_action_set[state] = [action for action in pruned_action_set[state] if action[:t] not in toPrune] # remove all actions with prefix

            tot += len(pruned_action_set[state])

            pr[state] = toPrune

        pruned.append(pr)

        ratios.append({state: (len(pruned_action_set[state]) / len(compMDP.actions)) for state in base_mdp.states})
        compMDPs.append(compMDP)

        # print("BnB Iteration",t,"/",checkin_period,":",tot / len(base_mdp.states),"avg action prefixes")
        end = time.time()
        print("BnB Iteration",t,"/",checkin_period,":", tot,"/",(len(base_mdp.states) * len(compMDP.actions)),"action prefixes. Took",end-start)

    # compMDP.actions = pruned_action_set
    # compMDP = extendCompositeMDP(base_mdp, discount, compMDP)

    tot = 0
    for state in base_mdp.states:
        tot += len(pruned_action_set[state])

    discount_input = pow(discount, checkin_period)
    # print("final",checkin_period,len(compMDP.actions),discount_input,threshold,max_iterations, tot,"/",(len(base_mdp.states) * len(compMDP.actions)))

    start = time.time()
    if doLinearProg:
        policy, values = linearProgrammingSolve(grid, compMDP, discount_input, pruned_action_set)
        q_values = qValuesFromR(compMDP, discount_input, values, pruned_action_set)
    else:
        policy, values, q_values = qValueIteration(grid, compMDP, discount_input, threshold, max_iterations, pruned_action_set)
    end = time.time()

    # print(len(compMDP.actions),"actions vs",pow(len(base_mdp.actions), checkin_period))
    print("BnB Iteration",t,"/",checkin_period,":", tot,"/",(len(base_mdp.states) * len(compMDP.actions)),"action prefixes. Took",end-start)

    return compMDP, policy, values, q_values, ratios, upperBounds, lowerBounds, pruned, compMDPs


def smallGrid():
    goalActionReward = 10000
    noopReward = 0#-1
    wallPenalty = -50000
    movePenalty = -1

    moveProb = 0.4
    discount = 0.707106781#0.5

    grid = [
        [0, 0, 0, 0, 1, 0, 2],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ]

    start_state = (0, 0)

    mdp = createMDP(grid, goalActionReward, noopReward, wallPenalty, movePenalty, moveProb)
    return grid, mdp, discount, start_state


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


def run(grid, mdp, discount, start_state, checkin_period, doBranchAndBound, drawPolicy=True, drawIterations=True, outputPrefix="", doLinearProg=False, bnbGreedy=-1, doSimilarityCluster=False, simClusterParams=None):
    policy = None
    values = None
    q_values = None

    start = time.time()
    elapsed = None

    if not doBranchAndBound:
        compMDPs = createCompositeMDPs(mdp, discount, checkin_period)
        compMDP = compMDPs[-1]
        print("Actions:",len(mdp.actions),"->",len(compMDP.actions))

        end1 = time.time()
        print("MDP composite time:", end1 - start)

        # policy, values = valueIteration(grid, compMDP, discount, 1e-20, int(1e4))#1e-20, int(1e4))
        discount_t = pow(discount, checkin_period)
        # print("final",checkin_period,len(compMDP.actions),discount_t,1e-20, int(1e4), (len(mdp.states) * len(compMDP.actions)))

        restricted_action_set = None

        if doSimilarityCluster:
            sc1 = time.time()
            
            checkinPeriodLimit = simClusterParams[0]
            thresh = simClusterParams[1]

            if checkinPeriodLimit < 0:
                checkinPeriodLimit = checkin_period

            mdpToCluster = compMDPs[checkinPeriodLimit-1]

            clusters = getActionClusters(mdpToCluster, thresh)

            count = 0
            count_s = 0

            restricted_action_set = {}

            for state in compMDP.states:
                restricted_action_set[state] = [action for action in compMDP.actions if action[:checkinPeriodLimit] not in clusters[state]]

                num_removed = len(compMDP.actions) - len(restricted_action_set[state])
                count += num_removed

                if state == start_state:
                    count_s = num_removed

            sc2 = time.time()
            print("Similarity time:", sc2 - sc1)

            percTotal = "{:.2f}".format(count / (len(compMDP.states) * len(compMDP.actions)) * 100)
            percStart = "{:.2f}".format(count_s / (len(compMDP.actions)) * 100)
            print(f"Actions under {thresh} total: {count} / {len(compMDP.states) * len(compMDP.actions)} ({percTotal}%)")
            print(f"Actions under {thresh} in start state: {count_s} / {len(compMDP.actions)} ({percStart}%)")

        if doLinearProg:
            l1 = time.time()
            policy, values = linearProgrammingSolve(grid, compMDP, discount_t, restricted_action_set = restricted_action_set)
            
            end2 = time.time()
            print("MDP linear programming time:", end2 - l1)
        else:
            q1 = time.time()
            policy, values, q_values = qValueIteration(grid, compMDP, discount_t, 1e-20, int(1e4), restricted_action_set=restricted_action_set)#1e-20, int(1e4))
            print(policy)

            end2 = time.time()
            print("MDP value iteration time:", end2 - q1)
        
        print("MDP total time:", end2 - start)
        elapsed = end2 - start

        print("Start state value:",values[start_state])

        if drawPolicy:
            draw(grid, compMDP, values, policy, True, False, "output/policy-"+outputPrefix+str(checkin_period)+("-vi" if not doLinearProg else "-lp"))
    else:
        compMDP, policy, values, q_values, ratios, upperBounds, lowerBounds, pruned, compMDPs = branchAndBound(grid, mdp, discount, checkin_period, 1e-20, int(1e4), doLinearProg=doLinearProg, greedy=bnbGreedy)
        print(policy)
        
        end = time.time()
        print("MDP branch and bound with " + ("linear programming" if doLinearProg else "q value iteration") + " time:", end - start)
        print("MDP total time:", end - start)
        elapsed = end - start

        print("Start state", start_state, "value:",values[start_state])

        suffix = "bnb-lp" if doLinearProg else "bnb-q"

        if bnbGreedy <= 0:
            suffix += "-nG"
        else:
            suffix += "-G" + str(bnbGreedy)

        if drawIterations:
            for i in range(0, checkin_period-1):
                drawBNBIteration(grid, compMDPs[i], ratios, upperBounds, lowerBounds, pruned, i, "output/policy-"+outputPrefix+str(checkin_period)+"-"+suffix+"-"+str(i+1))

        if drawPolicy:
            draw(grid, compMDP, values, policy, True, False, "output/policy-"+outputPrefix+str(checkin_period)+"-"+suffix+"-f")

    # if not os.path.exists("output/"):
    #     os.makedirs("output/")

    # draw(grid, compMDP, values, {}, False, True, "output/multi"+str(checkin_period))
    # draw(grid, compMDP, values, policy, True, False, "output/policy"+str(checkin_period))


    # s = compMDP.states[0]
    # for action in compMDP.transitions[s].keys():
    #     for end_state in compMDP.transitions[s][action].keys():
    #         print(s,action,"->",end_state,"is",compMDP.transitions[s][action][end_state])

    return values[start_state], policy, elapsed, compMDP


def runFig2Ratio(wallMin, wallMax, increment = 1, _discount = math.sqrt(0.99)):
    results = []
    for numWalls in range(wallMin, wallMax+increment, increment):
        grid, mdp, discount, start_state = paper2An(numWalls, _discount)

        pref = "paperFig2-" + str(numWalls) + "w-"
        value2, _, elapsed2, _ = run(grid, mdp, discount, start_state, checkin_period=2, doBranchAndBound=False, doLinearProg=True, drawPolicy=False, drawIterations=False, outputPrefix=pref)
        value3, _, elapsed3, _ = run(grid, mdp, discount, start_state, checkin_period=3, doBranchAndBound=False, doLinearProg=True, drawPolicy=False, drawIterations=False, outputPrefix=pref)

        r = (numWalls, value2, value3)
        results.append(r)

        print("\nLength " + str(3 + r[0] * 3 + 1) + " (" + str(r[0]) + " walls):")
        print("\tValue k=2:", r[1])
        print("\tValue k=3:", r[2])
        print("\tRatio k3/k2:", r[2]/r[1])
        print("")

    print("\n\n ===== RESULTS ===== \n\n")
    for r in results:
        print("Length " + str(3 + r[0] * 3 + 1) + " (" + str(r[0]) + " walls) k3/k2:", r[2]/r[1])
        # print("")


def runCheckinSteps(checkinMin, checkinMax, increment = 1):
    grid, mdp, discount, start_state = paper2An(3)
    times = []

    for checkin_period in range(checkinMin, checkinMax+increment, increment):
        print("\n\n ==== CHECKIN PERIOD " + str(checkin_period)  + " ==== \n\n")
        time = 0
        for i in range(0, 1):
            value, _, elapsed, _ = run(grid, mdp, discount, start_state, checkin_period, doBranchAndBound=False, doLinearProg=True) # LP
            time += elapsed
        time /= 1

        times.append(time)
        print("")
        print("Average",time)
        print(times)


def countActionSimilarity(mdp, thresh):

    count = 0
    counts = {}
    
    clusters = getActionClusters(mdp, thresh)

    for state in mdp.states:
        num_removed = len(clusters[state])
        count += num_removed
        counts[state] = num_removed 

    return count, counts

def getActionClusters(mdp, thresh):

    tA = 0
    tB = 0
    tC = 0
    tD = 0
    tE = 0

    clusters = {state: {} for state in mdp.states}

    ati = {}
    for i in range(len(mdp.actions)):
        ati[mdp.actions[i]] = i
    sti = {}
    for i in range(len(mdp.states)):
        sti[mdp.states[i]] = i

    for state in mdp.states:
        # s1 = time.time()

        actions = np.zeros((len(mdp.actions), len(mdp.states)))
        for action in mdp.transitions[state]:
            for end_state in mdp.transitions[state][action]:
                actions[ati[action]][sti[end_state]] = mdp.transitions[state][action][end_state]

        # actions = np.array([([(mdp.transitions[state][action][end_state] if end_state in mdp.transitions[state][action] else 0) for end_state in mdp.states]) for action in mdp.actions])

        # s2 = time.time()
        # tA += s2 - s1

        rewards = np.array([mdp.rewards[state][mdp.actions[i]] for i in range(len(mdp.actions))])
        rewards_transpose = rewards[:,np.newaxis]
        reward_diffs = np.abs(rewards_transpose - rewards)
        # reward_diffs = np.array([([abs(mdp.rewards[state][mdp.actions[i]] - mdp.rewards[state][mdp.actions[j]]) for j in range(len(mdp.actions))]) for i in range(len(mdp.actions))])

        # s2b = time.time()
        # tB += s2b - s2

        A_sparse = sparse.csr_matrix(actions)

        # s3 = time.time()
        # tC += s3 - s2b

        differences = 1 - cosine_similarity(A_sparse)

        total_diffs = reward_diffs + differences

        # s4 = time.time()
        # tD += s4 - s3

        indices = np.where(total_diffs <= thresh) # 1st array in tuple is row indices, 2nd is column
        filtered = np.where(indices[0] > indices[1])[0] # ignore diagonal, ignore duplicate

        indices_filtered = [(indices[0][i], indices[1][i]) for i in filtered] # array of pairs of indices

        G = nx.Graph()
        G.add_edges_from(indices_filtered)

        for connected_comp in nx.connected_components(G):
            cluster = [mdp.actions[ind] for ind in connected_comp]
            
            for i in range(1, len(cluster)): # skip first one in cluster (leader)
                action = cluster[i]
                clusters[state][action] = cluster
                
        # s5 = time.time()
        # tE += s5 - s4

    # print(tA)
    # print(tB)
    # print(tC)
    # print(tD)
    # print(tE)

    return clusters



def checkActionSimilarity(mdp):

    nA = len(mdp.actions)
    diffs = {}

    for state in mdp.states:
        actions = {}
  
        cost_diffs = np.zeros((nA, nA))
        transition_diffs = np.zeros((nA, nA))
        total_diffs = np.zeros((nA, nA))

        for action in mdp.actions:
            reward = mdp.rewards[state][action]
            transitions = mdp.transitions[state][action]
            probability_dist = np.array([(transitions[end_state] if end_state in transitions else 0) for end_state in mdp.states])

            actions[action] = (reward, probability_dist)

        for i in range(len(mdp.actions) - 1):
            actionA = mdp.actions[i]
            transitionsA = actions[actionA][1]

            for j in range(i+1, len(mdp.actions)):
                actionB = mdp.actions[j]
                transitionsB = actions[actionB][1]

                cost_difference = abs(actions[actionA][0] - actions[actionB][0])

                # cosine similarity, 1 is same, 0 is orthogonal, and -1 is opposite
                transition_similarity = np.dot(transitionsA, transitionsB) / np.linalg.norm(transitionsA) / np.linalg.norm(transitionsB)
                # difference, 0 is same, 1 is orthogonal, 2 is opposite
                transition_difference = 1 - transition_similarity

                total_difference = 1 * cost_difference + 1 * transition_difference

                cost_diffs[i][j] = cost_difference
                cost_diffs[j][i] = cost_difference

                transition_diffs[i][j] = transition_difference
                transition_diffs[j][i] = transition_difference

                total_diffs[i][j] = total_difference
                total_diffs[j][i] = total_difference

        diffs[state] = (cost_diffs, transition_diffs, total_diffs)

    return diffs

def makeTable(short_names, diffs):
    idx = pd.Index(short_names)
    df = pd.DataFrame(diffs, index=idx, columns=short_names)

    vals = np.around(df.values, 3) # round to 2 digits
    # norm = plt.Normalize(vals.min()-1, vals.max()+1)
    norm = plt.Normalize(vals.min(), vals.max()+0.2)
    colours = plt.cm.plasma_r(norm(vals))

    colours[np.where(diffs < 1e-5)] = [1, 1, 1, 1]

    fig = plt.figure(figsize=(15,8), dpi=300)
    ax = fig.add_subplot(111, frameon=True, xticks=[], yticks=[])

    the_table=plt.table(cellText=vals, rowLabels=df.index, colLabels=df.columns, 
                        loc='center', 
                        cellColours=colours)

def visualizeActionSimilarity(mdp, diffs, state, midfix=""):
    print("State:", state)
    cost_diffs, transition_diffs, total_diffs = diffs[state]

    short_names = []
    for action in mdp.actions:
        short_name = ""
        for a in action:
            short_name += ("0" if a == "NO-OP" else a[0])
        short_names.append(short_name)

    makeTable(short_names, cost_diffs)
    plt.savefig(f'output/diff{midfix}-cost.png', bbox_inches='tight')

    makeTable(short_names, transition_diffs)
    plt.savefig(f'output/diff{midfix}-transition.png', bbox_inches='tight')

    makeTable(short_names, total_diffs)
    plt.savefig(f'output/diff{midfix}-total.png', bbox_inches='tight')

    # plt.show()

def countSimilarity(mdp, diffs, diffType, thresh):
    count = 0
    counts = {}
    
    for state in mdp.states:
        d = diffs[state][diffType]
        indices = np.where(d <= thresh) # 1st array in tuple is row indices, 2nd is column
        filter = np.where(indices[0] > indices[1])[0] # ignore diagonal, ignore duplicate
        indices_filtered = [(indices[0][i], indices[1][i]) for i in filter] # array of pairs of indices

        c = len(indices_filtered)
        
        count += c
        counts[state] = c

    return count, counts


def blendMDP(mdp1, mdp2, stepsFromState, stateReference):

    mdp = MDP([], [], {}, {}, [])
    mdp.states = mdp1.states
    mdp.terminals = mdp1.terminals

    for state in mdp1.states:
        manhattanDist = abs(state[0] - stateReference[0]) + abs(state[1] - stateReference[1])
        mdpToUse = mdp1 if manhattanDist < stepsFromState else mdp2

        mdp.transitions[state] = mdpToUse.transitions[state]
        mdp.rewards[state] = {}
        for action in mdpToUse.transitions[state].keys():
            if action not in mdp.actions:
                mdp.actions.append(action)
            if action in mdpToUse.rewards[state]:
                mdp.rewards[state][action] = mdpToUse.rewards[state][action]

    return mdp
        

def runTwoCadence(checkin1, checkin2):

    n1 = 3
    grid, mdp, discount, start_state = corridorTwoCadence(n1=n1, n2=n1, cadence1=checkin1, cadence2=checkin2)
    policy = None
    values = None
    
    start = time.time()
    elapsed = None

    compMDP1 = createCompositeMDP(mdp, discount, checkin1)
    compMDP2 = createCompositeMDP(mdp, discount, checkin2)
    print("Actions:",len(mdp.actions),"->",str(len(compMDP1.actions)) + ", " + str(len(compMDP2.actions)))

    end1 = time.time()
    print("MDP composite time:", end1 - start)

    tB = 0
    tL = 0

    bestK = -1
    bestStartVal = -1
    bestBlendedMDP = None
    bestPolicy = None
    bestValues = None
    vals = []

    for k in range(4,5):#0, 14):
        b1 = time.time()
        blendedMDP = blendMDP(mdp1 = compMDP1, mdp2 = compMDP2, stepsFromState = k, stateReference=start_state)
        tB += time.time() - b1

        discount_t = discount#pow(discount, checkin_period)

        l1 = time.time()
        policy, values = linearProgrammingSolve(grid, blendedMDP, discount_t)
        tL += time.time() - l1

        vals.append(values[start_state])
        
        if values[start_state] > bestStartVal:
            bestStartVal = values[start_state] 
            bestK = k
            bestBlendedMDP = blendedMDP
            bestPolicy = policy
            bestValues = values

    print("Best K:", bestK)
    print("Values", vals)
    # print("Val diff:", vals[(n1-1)*2] - vals[0])

    print("MDP blend time:", tB)
    print("MDP linear programming time:", tL)
    
    end = time.time()
    print("MDP total time:", end - start)
    elapsed = end - start

    print("Start state value:",bestValues[start_state])

    draw(grid, bestBlendedMDP, bestValues, bestPolicy, True, False, "output/policy-comp-"+str(checkin1)+"v"+str(checkin2)+"-lp")

    return bestValues[start_state], elapsed

def runOneValueIterationPass(prev_values, discount, mdp):
    new_values = {}

    for state in mdp.states:
        max_expected = -1e20
        for action in mdp.actions:
            expected_value = mdp.rewards[state][action]
            future_value = 0

            for end_state in mdp.transitions[state][action].keys():
                prob = mdp.transitions[state][action][end_state]
                future_value += discount * prob * prev_values[end_state]

            expected_value += future_value

            max_expected = max(max_expected, expected_value)
        new_values[state] = max_expected

    return new_values

def policyFromValues(mdp, values, restricted_action_set = None):
    policy = {}
    for state in mdp.states:
        best_action = None
        max_expected = None
        
        action_set = mdp.actions if restricted_action_set is None else restricted_action_set[state]
        for action in action_set:
            if action in mdp.transitions[state]:
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
    return policy


def extendMarkovHittingTime(mdp, transition_matrix, target_state, checkin_period, prev_hitting_times):
    H = []
    for i in range(len(mdp.states)):
        h_i = 0
        if mdp.states[i] != target_state:
            h_i = checkin_period
            for j in range(len(mdp.states)):
                h_i += transition_matrix[i][j] * prev_hitting_times[j]
        H.append(h_i)
    return H


def expectedMarkovHittingTime(mdp, transition_matrix, target_state, checkin_period):
    # H_i = hitting time from state i to target state
    # H_F = hitting time from target state to itself (0)
    # H_i = 1 + sum (p_ij * H_j) over all states j (replace 1 with checkin period)
    
    # (I - P) H = [1, 1, ..., 1] 
    #   where row in P corresponding to target state is zero'd 
    #   and element in right vector corresponding to target state is zero'd

    n = len(mdp.states)

    target_index = mdp.states.index(target_state)
    
    I = np.identity(n)
    P = np.matrix.copy(transition_matrix)
    C = np.full(n, checkin_period)#np.ones(n)
    
    C[target_index] = 0
    P[target_index] = 0

    A = I - P

    H = np.linalg.solve(A, C) # Ax = C

    return H


def markovProbsFromPolicy(mdp, policy):
    transition_matrix = []
    for start_state in mdp.states:
        action = policy[start_state]
        row = [(mdp.transitions[start_state][action][end_state] if action is not None and end_state in mdp.transitions[start_state][action] else 0) for end_state in mdp.states]
        transition_matrix.append(row)
    return np.array(transition_matrix)


def policyEvaluation(mdp, policy, discount):
    # U(s) = C(s, pi(s)) + sum over s' {T'(s', pi(s), s) U(s')}
    # (I - P) U = C
    
    transition_matrix = markovProbsFromPolicy(mdp, policy)

    n = len(mdp.states)

    I = np.identity(n)
    P = discount * np.matrix.copy(transition_matrix)
    C = np.array([mdp.rewards[state][policy[state]] for state in mdp.states])

    A = I - P

    U = np.linalg.solve(A, C) # Ax = C

    return {mdp.states[i]: U[i] for i in range(len(U))}

def extendPolicyEvaluation(mdp, policy, oldEval, discount):
    U = {}
    for state in mdp.states:
        action = policy[state]
        u_i = mdp.rewards[state][action]
        
        for end_state in mdp.states:
            if end_state in mdp.transitions[state][action]:
                u_i += discount * mdp.transitions[state][action][end_state] * oldEval[end_state]
        U[state] = u_i
    return U

# def getAllStateParetoValues(mdp, chain):
#     pareto_values = []
#     for i in range(len(mdp.states)):
#         state = mdp.states[i]

#         values = chain[1]
#         hitting = chain[3]
    
#         hitting_time = hitting[0][i]
#         hitting_checkins = hitting[1][i]

#         checkin_cost = hitting_checkins
#         execution_cost = - values[state]

#         pareto_values.append(checkin_cost)
#         pareto_values.append(execution_cost)
#     return pareto_values

def getStateDistributionParetoValues(mdp, chain, distributions):
    pareto_values = []
    for distribution in distributions:
        dist_checkin_cost = 0
        dist_execution_cost = 0

        for i in range(len(mdp.states)):
            state = mdp.states[i]

            # values = chain[1]
            # hitting = chain[3]
        
            # hitting_time = hitting[0][i]
            # hitting_checkins = hitting[1][i]
            values = chain[0]
            
            execution_cost = - values[state]
            checkin_cost = - chain[1][state]

            dist_execution_cost += distribution[i] * execution_cost
            dist_checkin_cost += distribution[i] * checkin_cost

        pareto_values.append(dist_execution_cost)
        pareto_values.append(dist_checkin_cost)
    return pareto_values


def getStartParetoValues(mdp, chains, initialDistribution):
    dists = [initialDistribution]

    costs = []
    indices = []
    for chain in chains:
        name = ""
        for checkin in chain[0]:
            name += str(checkin)
        name += "*"

        points = chainPoints(chain)
        idx = []
        #nameSuff = [' $\pi^\\ast$', ' $\pi^c$']
        for p in range(len(points)):
            point = points[p]
            
            idx.append(len(costs))
            costs.append([name, getStateDistributionParetoValues(mdp, point, dists)])

        indices.append([name, idx])
    return costs, indices

def dirac(mdp, state):
    dist = []
    for s in mdp.states:
        dist.append(1 if s == state else 0)
    return dist


def gaussian(mdp, center_state, sigma):
    dist = []
    total = 0

    for i in range(len(mdp.states)):
        state = mdp.states[i]
        x_dist = abs(state[0] - center_state[0])
        y_dist = abs(state[1] - center_state[1])

        gaussian = 1 / (2 * math.pi * pow(sigma, 2)) * math.exp(- (pow(x_dist, 2) + pow(y_dist, 2)) / (2 * pow(sigma, 2)))
        dist.append(gaussian)

        total += gaussian

    # normalize
    if total > 0:
        for i in range(len(mdp.states)):
            dist[i] /= total
    
    return dist

def uniform(mdp):
    each_value = 1.0 / len(mdp.states)
    dist = [each_value for i in range(len(mdp.states))]
    
    return dist

def chainPoints(chain):
    execution_pi_star = chain[1][0][0]
    checkins_pi_star = chain[1][0][1]
    
    checkins_pi_c = chain[1][1][0]
    execution_pi_c = chain[1][1][1]
    
    # return [[values, hitting_checkins]]
    return [[execution_pi_star, checkins_pi_star], [execution_pi_star, checkins_pi_c], [execution_pi_c, checkins_pi_c]]
    # return [[values, hitting_checkins], [values, hitting_checkins_greedy], [values_greedy, hitting_checkins_greedy]]

def step_filter(new_chains, all_chains, distributions, margin, bounding_box):

    costs = []
    indices = []

    for i in range(len(all_chains)):
        chain = all_chains[i]

        idx = []

        points = chainPoints(chain)

        for j in range(len(points)):
            point = points[j]
            cost = getStateDistributionParetoValues(mdp, point, distributions)
            idx.append(len(costs))
            costs.append(cost)

        indices.append(idx)
        
    #costs = [getStateDistributionParetoValues(mdp, chain, distributions) for chain in all_chains]
    is_efficient = calculateParetoFrontC(costs)
    is_efficient_chains = []

    front = np.array([costs[i] for i in range(len(costs)) if is_efficient[i]])

    #filtered_all_chains = [all_chains[i] for i in range(len(all_chains)) if is_efficient[i]]
    filtered_all_chains = []
    for i in range(len(all_chains)):
        chain = all_chains[i]

        efficient = False

        for idx in indices[i]:
            if is_efficient[idx]:
                efficient = True
                filtered_all_chains.append(chain)
                break
        is_efficient_chains.append(efficient)

        if not efficient and margin > 0:
            for idx in indices[i]:
                cost = np.array(costs[idx])
                dist = calculateDistance(cost, front, bounding_box)
                if dist <= margin:
                    filtered_all_chains.append(chain)
                    break

    # front = np.array([costs[i] for i in range(len(all_chains)) if is_efficient[i]])
    
    # if margin > 0 and len(front) >= 1:
    #     for i in range(len(all_chains)):
    #         if not is_efficient[i]:
    #             chain = all_chains[i]
    #             cost = np.array(costs[i])
    #             dist = calculateDistance(cost, front, bounding_box)
    #             if dist <= margin:
    #                 filtered_all_chains.append(chain)

    filtered_new_chains = [chain for chain in new_chains if chain in filtered_all_chains] # can do this faster with index math

    return filtered_new_chains, filtered_all_chains


def chain_to_str(chain):
    name = ""
    for checkin in chain[0]:
        name += str(checkin)
    name += "*"
    return name

def chains_to_str(chains):
    text = "["
    for chain in chains:
        name = ""
        for checkin in chain[0]:
            name += str(checkin)
        name += "*"

        if text != "[":
            text += ", "
        text += name
    text += "]"
    return text
    

def drawParetoStep(mdp, chains, initialDistribution, TRUTH, TRUTH_COSTS, plotName, title, stepLen, bounding_box):

    plotName += "-step" + str(stepLen)
    title += " Length " + str(stepLen)

    start_state_costs, indices = getStartParetoValues(mdp, chains, initialDistribution)
        
    is_efficient = calculateParetoFront(start_state_costs)
    saveDataChains(start_state_costs, indices, is_efficient, TRUTH, TRUTH_COSTS, "pareto-" + plotName)
    drawParetoFront(start_state_costs, indices, is_efficient, TRUTH, TRUTH_COSTS, "pareto-" + plotName, title, bounding_box, prints=False)

def createChainTail(grid, mdp, discount, discount_checkin, target_state, compMDPs, greedyCompMDPs, k):
    discount_t = pow(discount, k)
    discount_c_t = pow(discount_checkin, k)
    compMDP = compMDPs[k]
    greedyMDP = greedyCompMDPs[k]

    policy, values = linearProgrammingSolve(grid, compMDP, discount_t)
    policy_greedy, values_greedy = linearProgrammingSolve(grid, greedyMDP, discount_c_t, restricted_action_set=None, is_negative=True) # we know values are negative, LP & simplex method doesn't work with negative decision variables so we flip 
    
    #hitting_time = expectedMarkovHittingTime(mdp, markov, target_state, k)
    #hitting_checkins = expectedMarkovHittingTime(mdp, markovProbsFromPolicy(compMDP, policy), target_state, 1)

    #eval_greedy = policyEvaluation(greedyMDP, policy_greedy, discount_c_t)
    eval_normal = policyEvaluation(greedyMDP, policy, discount_c_t)
    eval_greedy = policyEvaluation(compMDP, policy_greedy, discount_t)

    #hitting_checkins_greedy = expectedMarkovHittingTime(mdp, markovProbsFromPolicy(greedyMDP, policy_greedy), target_state, 1)

    # print(k)
    # # print("valu1",values)
    # print("valu",[values_greedy[mdp.states[i]] for i in range(len(mdp.states))])
    # print("eval",eval_greedy)
    # print("hitt", hitting_checkins_greedy)

    # draw(grid, greedyMDP, values_greedy, policy_greedy, True, False, "output/policy-chain"+str(k)+"-lp")

    #chain = [[k], values, [policy], (hitting_time, hitting_checkins)]
    #chain = [[k], [[values, hitting_checkins], [values_greedy, eval_greedy]]]
    chain = [[k], [[values, eval_normal], [values_greedy, eval_greedy]]]
    return chain

def extendChain(discount, discount_checkin, compMDPs, greedyCompMDPs, chain, k):
    compMDP = compMDPs[k]
    greedyMDP = greedyCompMDPs[k]

    chain_checkins = list(chain[0])
    chain_checkins.insert(0, k)

    tail_values = chain[1]
    discount_t = pow(discount, k)
    discount_c_t = pow(discount_checkin, k)

    #new_values = runOneValueIterationPass(tail_values, discount_t, compMDP)

    #policies = list(chain[2])
    #policy = policyFromValues(compMDP, tail_values) # !!!! should be using new values not old yes?
    #policies.insert(0, policy)

    # markov = markovProbsFromPolicy(compMDP, policy)
    # prev_hitting_time = chain[3][0]
    # prev_hitting_checkins = chain[3][1]
    # hitting_time = extendMarkovHittingTime(mdp, markov, target_state, k, prev_hitting_time)
    # hitting_checkins = extendMarkovHittingTime(mdp, markov, target_state, 1, prev_hitting_checkins)
    
    # new_chain = [chain_checkins, new_values, policies, (hitting_time, hitting_checkins)]

    new_values = runOneValueIterationPass(tail_values[0][0], discount_t, compMDP)
    policy = policyFromValues(compMDP, new_values)
    
    # prev_hitting = tail_values[0][1]
    # new_hitting = extendMarkovHittingTime(mdp, markovProbsFromPolicy(compMDP, policy), target_state, 1, prev_hitting)
    new_eval = extendPolicyEvaluation(greedyMDP, policy, tail_values[0][1], discount_c_t)


    new_values_greedy = runOneValueIterationPass(tail_values[1][0], discount_c_t, greedyMDP)
    policy_greedy = policyFromValues(greedyMDP, new_values_greedy)

    #new_eval_greedy = extendPolicyEvaluation(greedyMDP, policy_greedy, tail_values[1][1], discount_c_t)
    new_eval_greedy = extendPolicyEvaluation(compMDP, policy_greedy, tail_values[1][1], discount_t)
    
    # prev_hitting_greedy = tail_values[1][2]
    # new_hitting_greedy = extendMarkovHittingTime(mdp, markovProbsFromPolicy(greedyMDP, policy_greedy), target_state, 1, prev_hitting_greedy)
    
    # new_chain = [chain_checkins, [[new_values, new_hitting], [new_values_greedy, new_eval_greedy, new_hitting_greedy]]]
    new_chain = [chain_checkins, [[new_values, new_eval], [new_values_greedy, new_eval_greedy]]]
    return new_chain

def calculateChainValues(grid, mdp, discount, discount_checkin, start_state, target_state, checkin_periods, chain_length, do_filter, distributions, initialDistribution, margin, bounding_box, drawIntermediate, TRUTH, TRUTH_COSTS, name, title):
    all_compMDPs = createCompositeMDPs(mdp, discount, checkin_periods[-1])
    compMDPs = {k: all_compMDPs[k - 1] for k in checkin_periods}

    # greedy_mdp = convertToGreedyMDP(grid, mdp)
    # all_greedy_compMDPs = createCompositeMDPs(greedy_mdp, discount_checkin, checkin_periods[-1])
    # greedyCompMDPs = {k: all_greedy_compMDPs[k-1] for k in checkin_periods}
    greedyCompMDPs = {k: convertCompToCheckinMDP(grid, compMDPs[k], k, discount_checkin) for k in checkin_periods}

    for k in checkin_periods:
        print(k,greedyCompMDPs[k].rewards[greedyCompMDPs[k].states[0]][greedyCompMDPs[k].actions[0]])

    chains_list = []
    all_chains = []

    chains = []
    l = 1
    for k in checkin_periods:
        chain = createChainTail(grid, mdp, discount, discount_checkin, target_state, compMDPs, greedyCompMDPs, k)
        chains.append(chain)
        all_chains.append(chain)

    chains_list.append(chains)

    if drawIntermediate:
        drawParetoStep(mdp, all_chains, initialDistribution, TRUTH, TRUTH_COSTS, name, title, l, bounding_box)

    print("--------")
    print(len(all_chains),"current chains")
    # print("Current chains: " + chains_to_str(all_chains))

    for i in range(1, chain_length):
        l += 1
        previous_chains = chains_list[i - 1]
        chains = []

        for tail in previous_chains:
            for k in checkin_periods:
                if i == 1 and k == tail[0][0]:
                    continue # don't duplicate recurring tail value (e.g. 23* and 233*)
                
                new_chain = extendChain(discount, discount_checkin, compMDPs, greedyCompMDPs, tail, k)
                chains.append(new_chain)
                all_chains.append(new_chain)
        
        if do_filter:
            filtered_chains, filtered_all_chains = step_filter(chains, all_chains, distributions, margin, bounding_box)
            #print("Filtered from",len(chains),"to",len(filtered_chains),"new chains and",len(all_chains),"to",len(filtered_all_chains),"total.")
            og_len = len(all_chains) - len(chains)
            new_len_min_add = len(filtered_all_chains) - len(filtered_chains)
            removed = og_len - new_len_min_add
            
            # print("Considering new chains: " + chains_to_str(chains))
            print("Added",len(filtered_chains),"out of",len(chains),"new chains and removed",removed,"out of",og_len,"previous chains.")
            all_chains = filtered_all_chains

            chains_list.append(filtered_chains)
        else:
            chains_list.append(chains)

        if drawIntermediate:
            drawParetoStep(mdp, all_chains, initialDistribution, TRUTH, TRUTH_COSTS, name, title, l, bounding_box)

        print("--------")
        print(len(all_chains),"current chains")
        # print("Current chains: " + chains_to_str(all_chains))

    start_state_index = mdp.states.index(start_state)

    chains = all_chains
    # chains = sorted(chains, key=lambda chain: chain[1][start_state], reverse=True)

    start_state_costs, indices = getStartParetoValues(mdp, chains, initialDistribution)
    return start_state_costs, indices

    # costs = []
    # start_state_costs = []

    # for chain in chains:
    #     name = ""
    #     for checkin in chain[0]:
    #         name += str(checkin)
    #     name += "*"

    #     values = chain[1]
    #     hitting = chain[3]

    #     hitting_time = hitting[0][start_state_index]
    #     hitting_checkins = hitting[1][start_state_index]

    #     checkin_cost = hitting_checkins
    #     execution_cost = - values[start_state]

    #     # pareto_values = getAllStateParetoValues(mdp, chain)
    #     pareto_values = getStateDistributionParetoValues(mdp, chain, distributions)

    #     # print(name + ":", values[start_state], "| Hitting time:", hitting_time, "| Hitting checkins:", hitting_checkins, "| Execution cost:", execution_cost, "| Checkin cost:", checkin_cost)
    #     print(name + ":", values[start_state], "| Execution cost:", execution_cost, "| Checkin cost:", checkin_cost)
    #     # costs.append((name, execution_cost, checkin_cost))
    #     costs.append((name, pareto_values))
    #     start_state_costs.append((name, [execution_cost, checkin_cost]))
        
    # return costs, start_state_costs

    # best_chain = full_chains[0]
    # name = ""
    # for checkin in best_chain[0]:
    #     name += str(checkin)
    # name += "*"

    # for i in range(0, len(best_chain[0])):
    #     k = best_chain[0][i]
    #     compMDP = compMDPs[k]
        
    #     tail = tuple(best_chain[0][i:])
        
    #     values = all_values[tail]
    #     policy = all_policies[tail]
        
    #     draw(grid, compMDP, values, policy, True, False, "output/policy-comp-"+name+"-"+str(i))

def translateLabel(label):
    label = label[:-2] + "$\overline{" + label[-2] + "}$"
    # label = label[:-2] + "$\dot{" + label[-2] + "}$"
    
    return label

def scatter(ax, chains, doLabel, color, lcolor, arrows=False, x_offset = 0, x_scale=1, loffsets={}):
    # x = [chain[1][start_state_index * 2 + 1] for chain in chains]
    # y = [chain[1][start_state_index * 2] for chain in chains]
    x = [(chain[1][0] + x_offset) * x_scale for chain in chains]
    y = [chain[1][1] for chain in chains]
    labels = [chain[0] for chain in chains]
    
    ax.scatter(x, y, c=color)

    if doLabel:
        for i in range(len(labels)):
            l = labels[i]
            if not arrows:
                ax.annotate(translateLabel(l),
                    xy=(x[i], y[i]), xycoords='data',
                    xytext=(5, 5), textcoords='offset points',
                    color=lcolor)
            else:
                # offset = (-40, -40)
                offset = (40, 40)
                # if len(l) > 4:
                #     offset = (-40 - (min(len(l), 9)-4)*5, -40)
                #     if len(l) >= 20:
                #         offset = (offset[0], -20)

                if l in loffsets:
                    offset = (offset[0] + loffsets[l][0], offset[1] + loffsets[l][1])

                ax.annotate(translateLabel(l), 
                    xy=(x[i], y[i]), xycoords='data',
                    # xytext=((-30, -30) if color == "orange" else (-40, -40)), textcoords='offset points',
                    xytext=offset, textcoords='offset points',
                    arrowprops=dict(arrowstyle="->", color=lcolor), 
                    color=lcolor,fontsize=9)
            # ax.annotate(labels[i], (x[i], y[i]), color=lcolor)

def lines(ax, chains, color):
    x = [chain[1][0] for chain in chains]
    y = [chain[1][1] for chain in chains]
    
    ax.plot(x, y, c=color)

def manhattan_lines(ax, chains, color, bounding_box, x_offset=0, x_scale=1, linestyle=None):
    x = []
    y = []

    xmax = bounding_box[0][1]
    ymax = bounding_box[1][1]
    
    if len(chains) > 0:
        point = chains[0][1]
        x.append((point[0] + x_offset) * x_scale)
        y.append(ymax)

    for i in range(len(chains)):
        point = chains[i][1]
        
        x.append((point[0] + x_offset) * x_scale)
        y.append(point[1])

        if i < len(chains) - 1:
            next_point = chains[i+1][1]

            x.append((next_point[0] + x_offset) * x_scale)
            y.append(point[1])

    if len(chains) > 0:
        point = chains[-1][1]
        x.append((xmax + x_offset) * x_scale)
        y.append(point[1])
    
    if linestyle is None:
        ax.plot(x, y, c=color)
    else:
        ax.plot(x, y, c=color, linestyle=linestyle)

def addXY(point, x, y, x_offset=0, x_scale=1):
    x.append((point[0] + x_offset) * x_scale)
    y.append(point[1])

def box(ax, chains, color, bounding_box, x_offset=0, x_scale=1):
    x = []
    y = []

    # topLeft = chains[0][1]
    # middle = chains[1][1]
    # bottomRight = chains[2][1]
    
    # addXY(topLeft, x, y, x_offset, x_scale)
    # addXY((topLeft[0], bottomRight[1]), x, y, x_offset, x_scale)
    # addXY(bottomRight, x, y, x_offset, x_scale)
    # addXY((bottomRight[0], topLeft[1]), x, y, x_offset, x_scale)

    # addXY(topLeft, x, y, x_offset, x_scale)
    # addXY(middle, x, y, x_offset, x_scale)
    # addXY(bottomRight, x, y, x_offset, x_scale)

    for chain in chains:
        addXY(chain[1], x, y, x_offset, x_scale)
    
    ax.plot(x, y, c=color, linestyle="dashed")

def calculateParetoFront(chains):
    return calculateParetoFrontC([chain[1] for chain in chains])

def calculateParetoFrontC(costs):
    costs = np.array(costs)

    is_efficient = [True for i in range(len(costs))]#list(np.ones(len(costs), dtype = bool))
    for i, c in enumerate(costs):
        is_efficient[i] = bool(np.all(np.any(costs[:i]>c, axis=1)) and np.all(np.any(costs[i+1:]>c, axis=1)))

    return is_efficient

def areaUnderPareto(pareto_front):

    area = 0

    if len(pareto_front) == 1:
        return pareto_front[0][1]

    for i in range(len(pareto_front) - 1):
        x_i = pareto_front[i][0]
        y_i = pareto_front[i][1]

        x_j = pareto_front[i+1][0]
        y_j = pareto_front[i+1][1]

        rectangle = y_i * (x_j - x_i)  # since smaller is good, left hand Riemann
        triangle = (y_j - y_i) * (x_j - x_i) / 2.0

        area += rectangle 
        # area += triangle

    return area

def lineseg_dists(p, a, b):
    # Handle case where p is a single point, i.e. 1d array.
    p = np.atleast_2d(p)

    # TODO for you: consider implementing @Eskapp's suggestions
    if np.all(a == b):
        return np.linalg.norm(p - a, axis=1)

    # normalized tangent vector
    d = np.divide(b - a, np.linalg.norm(b - a))

    # signed parallel distance components
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(p))])

    # perpendicular distance component, as before
    # note that for the 3D case these will be vectors
    c = np.cross(p - a, d)

    # use hypot for Pythagoras to improve accuracy
    return np.hypot(h, c)

def calculateDistance(point, pareto_front, bounding_box):
    # chains_filtered.sort(key = lambda chain: chain[1][0])

    # x_range = (pareto_front[-1][1][0] - pareto_front[0][1][0])
    # min_x = pareto_front[0][1][0]

    # min_y = None
    # max_y = None
    # for c in pareto_front:
    #     y = c[1][1]
    #     if min_y is None or y < min_y:
    #         min_y = y
    #     if max_y is None or y > max_y:
    #         max_y = y
        
    # y_range = (max_y - min_y)

    # mins = np.min(pareto_front, axis=0)
    # ranges = np.ptp(pareto_front, axis=0)
    mins = bounding_box[:,0]
    ranges = bounding_box[:,1] - mins

    times_to_tile = int(len(point) / len(mins)) # since this is higher dimensional space, each point is (execution cost, checkin cost, execution cost, checkin cost, etc.)
    mins = np.tile(mins, times_to_tile)
    ranges = np.tile(ranges, times_to_tile)

    # x = (chain[1][0] - min_x) / x_range
    # y = (chain[1][1] - min_y) / y_range

    # front_normalized = [[(c[1][0] - min_x) / x_range, (c[1][1] - min_y) / y_range] for c in pareto_front]

    point_normalized = np.divide(point - mins, ranges)
    front_normalized = np.divide(pareto_front - mins, ranges)

    min_dist = None

    # p = np.array([x, y])

    # for i in range(len(pareto_front) - 1):
    #     x1 = pareto_front[i][0]
    #     y1 = pareto_front[i][1]

    #     x2 = pareto_front[i+1][0]
    #     y2 = pareto_front[i+1][1]

    #     # dist_to_line = abs((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1)) / math.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))
    #     dist_to_line = lineseg_dists(p, np.array([x1, y1]), np.array([x2, y2]))[0]
        
    #     if min_dist is None or dist_to_line < min_dist:
    #         min_dist = dist_to_line

    # for i in range(len(pareto_front)):
    #     x1 = pareto_front[i][0]
    #     y1 = pareto_front[i][1]
        
    #     dist = math.sqrt(pow(x - x1, 2) + pow(y - y1, 2))
        
    #     if min_dist is None or dist < min_dist:
    #         min_dist = dist

    min_dist = np.min(np.linalg.norm(front_normalized - point_normalized, axis=1))

    return min_dist

def calculateError(chains, is_efficient, true_front, bounding_box):
    chains_filtered = [chains[i][1] for i in range(len(chains)) if is_efficient[i]]
    chains_filtered.sort(key = lambda chain: chain[0])

    true = [t[1] for t in true_front]

    # x_range = (true_front[-1][1][0] - true_front[0][1][0])
    # min_x = true_front[0][1][0]

    # min_y = None
    # max_y = None
    # for c in true_front:
    #     y = c[1][1]
    #     if min_y is None or y < min_y:
    #         min_y = y
    #     if max_y is None or y > max_y:
    #         max_y = y
        
    # y_range = (max_y - min_y)

    # chains_normalized = [[(c[1][0] - min_x) / x_range, (c[1][1] - min_y) / y_range] for c in chains_filtered]
    # true_normalized = [[(c[1][0] - min_x) / x_range, (c[1][1] - min_y) / y_range] for c in true_front]

    mins = bounding_box[:,0]
    ranges = bounding_box[:,1] - bounding_box[:,0]

    chains_normalized = np.divide(np.array(chains_filtered) - mins, ranges)
    true_normalized = np.divide(np.array(true) - mins, ranges)

    area = areaUnderPareto(chains_normalized)
    area_true = areaUnderPareto(true_normalized)

    return abs(area - area_true) / area_true

def saveDataChains(chains, indices, is_efficient, TRUTH, TRUTH_COSTS, name):
    data = {'Points': chains, 'Indices': indices, 'Efficient': is_efficient}
    if TRUTH is not None:
        data['Truth'] = TRUTH
    if TRUTH_COSTS is not None:
        data['Truth Costs'] = TRUTH_COSTS
    jsonStr = json.dumps(data, indent=4)
    
    with open(f'output/data/{name}.json', "w") as file:
        file.write(jsonStr)

# def loadDataChains(filename):
#     with open(f'output/data/{filename}.json', "r") as file:
#         jsonStr = file.read()
#         obj = json.loads(jsonStr)
#         return (obj['Points'], obj['Indices'], obj['Efficient'])

def drawChainsParetoFront(chains, indices, is_efficient, true_front, true_costs, name, title, bounding_box, prints, x_offset=0, x_scale=1, loffsets={}):
    plt.style.use('seaborn-whitegrid')

    arrows = True

    font = FontProperties()
    font.set_family('serif')
    font.set_name('Times New Roman')
    font.set_size(20)
    # rc('font',**{'family':'serif','serif':['Times'],'size':20})
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times']
    plt.rcParams['font.size'] = 20
    plt.rcParams["text.usetex"] = True
    # plt.rcParams['font.weight'] = 'bold'
    
    chains_filtered = []
    chains_dominated = []
    for i in range(len(chains)):
        if is_efficient[i]:
            chains_filtered.append(chains[i])
        else:
            chains_dominated.append(chains[i])

    n = 0
    is_efficient_chains = []
    for i in range(len(indices)):
        idx = indices[i][1]

        efficient = False
        for j in idx:
            if is_efficient[j]:
                efficient = True
                n += 1
                break
        is_efficient_chains.append(efficient)

    print(n,"vs",len(indices))

    chains_filtered.sort(key = lambda chain: chain[1][0])

    if prints:
        print("Non-dominated chains:")
        for chain in chains_filtered:
            print("  ", chain[0])
    # x_f = [chain[1] for chain in chains_filtered]
    # y_f = [chain[2] for chain in chains_filtered]
    # labels_f = [chain[0] for chain in chains_filtered]

    if prints:
        print(len(chains_dominated),"dominated chains out of",len(chains),"|",len(chains_filtered),"non-dominated")

    # costs = [chain[1] for chain in chains_filtered]
    if prints:
        print("Pareto front:",chains_filtered)
    
    fig, ax = plt.subplots()
    # ax.scatter(x, y, c=["red" if is_efficient[i] else "black" for i in range(len(chains))])
    # ax.scatter(x_f, y_f, c="red")

    if true_costs is not None:
        scatter(ax, true_costs, doLabel=False, color="gainsboro", lcolor="gray", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)

    if true_front is not None:
        manhattan_lines(ax, true_front, color="green", bounding_box=bounding_box, x_offset=x_offset, x_scale=x_scale)
        scatter(ax, true_front, doLabel=False, color="green", lcolor="green", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)

    
    # scatter(ax, chains_dominated, doLabel=True, color="orange", lcolor="gray", arrows=arrows, x_offset=x_offset, x_scale=x_scale)
    
    # scatter(ax, chains_dominated, doLabel=False, color="orange", lcolor="gray", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)

    for i in range(len(is_efficient_chains)):
        points = []
        for j in indices[i][1]:
            points.append(chains[j])
        if is_efficient_chains[i]:
            box(ax, points, color="red", bounding_box=bounding_box, x_offset=x_offset, x_scale=x_scale)
            scatter(ax, points, doLabel=True, color="red", lcolor="gray", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)
        else:
            print("bad",indices[i][0], points[0])
            box(ax, points, color="orange", bounding_box=bounding_box, x_offset=x_offset, x_scale=x_scale)
            scatter(ax, points, doLabel=False, color="orange", lcolor="gray", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)

    

    manhattan_lines(ax, chains_filtered, color="red", bounding_box=bounding_box, x_offset=x_offset, x_scale=x_scale)
    scatter(ax, chains_filtered, doLabel=True, color="red", lcolor="black", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)
    
    # for i in range(len(chains)):
    #     plt.plot(x[i], y[i])

    # plt.xlabel("Execution Cost", fontproperties=font, fontweight='bold')
    # plt.ylabel("Checkin Cost", fontproperties=font, fontweight='bold')
    plt.xlabel(r"\textbf{Execution Cost}", fontproperties=font, fontweight='bold')
    plt.ylabel(r"\textbf{Checkin Cost}", fontproperties=font, fontweight='bold')
    #plt.title(title)

    plt.xlim((bounding_box[0] + x_offset) * x_scale)
    plt.ylim(bounding_box[1])

    plt.gcf().set_size_inches(10, 7)
    plt.savefig(f'output/{name}.pdf', format="pdf", bbox_inches='tight', pad_inches=0.2, dpi=300)
    # plt.savefig(f'output/{name}.png', bbox_inches='tight', pad_inches=0.5, dpi=300)
    # plt.savefig(f'output/pareto-{name}.svg', bbox_inches='tight', pad_inches=0.5, dpi=300, format="svg")
    # plt.show()



def drawChainsParetoFrontSuperimposed(stuffs, true_front, true_costs, name, bounding_box, x_offset=0, x_scale=1, loffsets={}):
    plt.style.use('seaborn-whitegrid')

    arrows = True

    font = FontProperties()
    font.set_family('serif')
    font.set_name('Times New Roman')
    font.set_size(20)
    # rc('font',**{'family':'serif','serif':['Times'],'size':20})
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times']
    plt.rcParams['font.size'] = 20
    plt.rcParams["text.usetex"] = True
    # plt.rcParams['font.weight'] = 'bold'
    
    fig, ax = plt.subplots()
    
    if true_costs is not None:
        scatter(ax, true_costs, doLabel=False, color="gainsboro", lcolor="gray", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)

    if true_front is not None:
        manhattan_lines(ax, true_front, color="green", bounding_box=bounding_box, x_offset=x_offset, x_scale=x_scale)
        scatter(ax, true_front, doLabel=False, color="green", lcolor="green", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)

    for i in range(len(stuffs)):
        (chains, is_efficient, color) = stuffs[i]
    
        chains_filtered = []
        chains_dominated = []
        for j in range(len(chains)):
            if is_efficient[j]:
                chains_filtered.append(chains[j])
            else:
                chains_dominated.append(chains[j])

        chains_filtered.sort(key = lambda chain: chain[1][0])
        
        if i == len(stuffs)-1:
            scatter(ax, chains_dominated, doLabel=False, color="orange", lcolor="gray", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)
        
        manhattan_lines(ax, chains_filtered, color=color, bounding_box=bounding_box, x_offset=x_offset, x_scale=x_scale)
        scatter(ax, chains_filtered, doLabel=(i == len(stuffs)-1), color=color, lcolor="black", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)
    
    # plt.xlabel("Execution Cost", fontproperties=font, fontweight='bold')
    # plt.ylabel("Checkin Cost", fontproperties=font, fontweight='bold')
    plt.xlabel(r"\textbf{Execution Cost}", fontproperties=font, fontweight='bold')
    plt.ylabel(r"\textbf{Checkin Cost}", fontproperties=font, fontweight='bold')
    
    #plt.title(title)

    plt.xlim((bounding_box[0] + x_offset) * x_scale)
    plt.ylim(bounding_box[1])

    plt.gcf().set_size_inches(10, 7)
    plt.savefig(f'output/{name}.pdf', format="pdf", bbox_inches='tight', pad_inches=0.2, dpi=300)
    # plt.savefig(f'output/{name}.png', bbox_inches='tight', pad_inches=0.5, dpi=300)
    # plt.savefig(f'output/pareto-{name}.svg', bbox_inches='tight', pad_inches=0.5, dpi=300, format="svg")
    # plt.show()


def drawCompares(data):
    plt.style.use('seaborn-whitegrid')

    fig, ax = plt.subplots()
    
    scatter(ax, data, doLabel=True, color="red", lcolor="black")

    plt.xlabel("Evaluation Time (s)")
    plt.ylabel("Error (%)")

    plt.gcf().set_size_inches(10, 7)
    plt.savefig(f'output/pareto-compare.png', bbox_inches='tight', pad_inches=0.5, dpi=300)
    plt.show()




def drawChainPolicy(grid, mdp, discount, discount_checkin, start_state, target_state, checkin_periods, chain_checkins, name):
    all_compMDPs = createCompositeMDPs(mdp, discount, checkin_periods[-1])
    compMDPs = {k: all_compMDPs[k - 1] for k in checkin_periods}

    # greedy_mdp = convertToGreedyMDP(grid, mdp)
    # all_greedy_compMDPs = createCompositeMDPs(greedy_mdp, discount_checkin, checkin_periods[-1])
    # greedyCompMDPs = {k: all_greedy_compMDPs[k - 1] for k in checkin_periods}
    greedyCompMDPs = {k: convertCompToCheckinMDP(grid, compMDPs[k], k, discount_checkin) for k in checkin_periods}

    i = len(chain_checkins) - 1
    chain = createChainTail(grid, mdp, discount, discount_checkin, target_state, compMDPs, greedyCompMDPs, chain_checkins[i])
    while i >= 0:
        chain = extendChain(discount, discount_checkin, compMDPs, greedyCompMDPs, chain, chain_checkins[i])
        i -= 1
    
    #chain = ([k], values, [policy], (hitting_time, hitting_checkins))
    policies = chain[2]
    values = chain[1]
    sequence = chain[0]

    max_value = None
    min_value = None

    if len(values) > 0:
        min_value = min(values.values())
        max_value = max(values.values())

    G = nx.MultiDiGraph()

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

    current_state = start_state
    stage = 0
    while True:
        if current_state == target_state:
            break

        policy = policies[stage]
        action = policy[current_state]
        k = sequence[stage]
        compMDP = compMDPs[k]

        maxProb = -1
        maxProbEnd = None
        
        for end in compMDP.transitions[current_state][action].keys():
            probability = compMDP.transitions[current_state][action][end]

            if probability > maxProb:
                maxProb = probability
                maxProbEnd = end
        
        if maxProbEnd is not None:
            end = maxProbEnd
            probability = maxProb
            color = "blue"
            G.add_edge(current_state, end, prob=probability, label=f"{action}: " + "{:.2f}".format(probability), color=color, fontcolor=color)
            
            current_state = end
            if stage < len(sequence) - 1:
                stage += 1
        else:
            break

    # Build plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    layout = {}

    ax.clear()
    labels = {}
    edge_labels = {}
    color_map = []

    # G.graph['edge'] = {'arrowsize': '1.0', 'fontsize':'10', 'penwidth': '5'}
    # G.graph['graph'] = {'scale': '3', 'splines': 'true'}
    G.graph['edge'] = {'arrowsize': '1.0', 'fontsize':'10', 'penwidth': '5'}
    G.graph['graph'] = {'scale': '3', 'splines': 'true'}

    A = to_agraph(G)

    A.node_attr['style']='filled'

    for node in G.nodes():
        labels[node] = f"{stateToStr(node)}"

        layout[node] = (node[0], -node[1])

        state_type = grid[node[1]][node[0]]

        n = A.get_node(node)
        n.attr['color'] = fourColor(node)

        if state_type != TYPE_WALL:
            n.attr['xlabel'] = "{:.4f}".format(values[node])

        color = None
        if state_type == TYPE_WALL:
            color = "#6a0dad"
        elif min_value is None and state_type == TYPE_GOAL:
            color = "#00FFFF"
        elif min_value is None:
            color = "#FFA500"
        else:
            value = values[node]
            frac = (value - min_value) / (max_value - min_value)
            hue = frac * 250.0 / 360.0 # red 0, blue 1

            col = colorsys.hsv_to_rgb(hue, 1, 1)
            col = (int(col[0] * 255), int(col[1] * 255), int(col[2] * 255))
            color = '#%02x%02x%02x' % col

        n.attr['fillcolor'] = color

        color_map.append(color)

    for s, e, d in G.edges(data=True):
        edge_labels[(s, e)] = "{:.2f}".format(d['prob'])

    # Set the title
    #ax.set_title("MDP")

    #plt.show()
    m = 0.7#1.5
    for k,v in layout.items():
        A.get_node(k).attr['pos']='{},{}!'.format(v[0]*m,v[1]*m)

    #A.layout('dot')
    A.layout(prog='neato')
    # A.draw(name + '.png')#, prog="neato")
    A.draw(name + '.pdf')#, prog="neato")


def runChains(grid, mdp, discount, discount_checkin, start_state, target_state, 
    checkin_periods, chain_length, do_filter, margin, distName, startName, distributions, initialDistribution, bounding_box, TRUTH, TRUTH_COSTS, drawIntermediate):

    title = distName[0].upper() + distName[1:]

    name = "c"+str(checkin_periods[-1]) + "-l" + str(chain_length)
    name += "-" + distName
    if startName != '':
        name += '-s' + startName
        title += " (Start " + startName[0].upper() + startName[1:] + ")"
    if do_filter:
        name += "-filtered"
        m_str = "{:.3f}".format(margin if margin > 0 else 0)
        
        name += "-margin" + m_str
        title += " (Margin " + m_str + ")"

    c_start = time.time()

    start_state_costs, indices = calculateChainValues(grid, mdp, discount, discount_checkin, start_state, target_state, 
        checkin_periods=checkin_periods, 
        # execution_cost_factor=1, 
        # checkin_costs={2: 10, 3: 5, 4: 2}, 
        chain_length=chain_length,
        do_filter = do_filter, 
        distributions=distributions, 
        initialDistribution=initialDistribution,
        margin=margin, 
        bounding_box=bounding_box,
        drawIntermediate=drawIntermediate,
        TRUTH=TRUTH, 
        TRUTH_COSTS=TRUTH_COSTS,
        name=name,
        title=title)

    numRemaining = len(start_state_costs)
    numWouldBeTotal = pow(len(checkin_periods), chain_length)
    numPruned = numWouldBeTotal - numRemaining
    fractionTrimmed = numPruned / numWouldBeTotal * 100

    is_efficient = calculateParetoFront(start_state_costs)

    c_end = time.time()
    running_time = c_end - c_start
    print("Chain evaluation time:", running_time)
    print("Trimmed:",numPruned,"/",numWouldBeTotal,"(" + str(int(fractionTrimmed)) + "%)")

    error = 0 if TRUTH is None else calculateError(start_state_costs, is_efficient, TRUTH, bounding_box)
    print("Error from true Pareto:",error)

    saveDataChains(start_state_costs, indices, is_efficient, TRUTH, TRUTH_COSTS, "pareto-" + name)
    drawParetoFront(start_state_costs, indices, is_efficient, TRUTH, TRUTH_COSTS, "pareto-" + name, title, bounding_box, prints=True)

    print("All costs:",start_state_costs)

    return running_time, error, fractionTrimmed


def convertToGreedyMDP(grid, mdp): # bad
    for state in mdp.rewards:
        (x, y) = state
        state_type = grid[y][x]

        if state_type == TYPE_GOAL:
            for action in mdp.rewards[state]:
                mdp.rewards[state][action] = 0
            continue
        
        for action in mdp.rewards[state]:
            mdp.rewards[state][action] = -1 # dont change mdp
    return mdp

def convertCompToCheckinMDP(grid, compMDP, checkin_period, discount):

    checkinMDP = MDP([], [], {}, {}, [])

    checkinMDP.states = compMDP.states.copy()
    checkinMDP.terminals = compMDP.terminals.copy()
    checkinMDP.actions = compMDP.actions.copy()
    checkinMDP.transitions = compMDP.transitions.copy()

    cost_per_stride = 1.0
    cost_per_action = cost_per_stride / checkin_period

    composed_cost = 0
    for i in range(checkin_period):
        composed_cost += pow(discount, i) * cost_per_action

    for state in compMDP.rewards:
        (x, y) = state
        state_type = grid[y][x]

        checkinMDP.rewards[state] = {}

        for action in compMDP.rewards[state]:
            checkinMDP.rewards[state][action] = 0 if state_type == TYPE_GOAL else (-composed_cost)
        
    return checkinMDP

# start = time.time()

# grid, mdp, discount, start_state = paper2An(3)#splitterGrid(rows = 50, discount=0.99)#paper2An(3)#, 0.9999)

grid, mdp, discount, start_state = corridorTwoCadence(n1=3, n2=6, cadence1=2, cadence2=3)
# grid, mdp, discount, start_state = splitterGrid2(rows = 12)
discount_checkin = discount

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
    

TRUTH_234 = [('223*', [-1471895.5491260004, 10.36207210840559]), ('23*', [-1466111.360834025, 10.272072943363703]), ('2343*', [-1459989.7576398463, 10.188573289663607]), ('2234*', [-1442990.0823133066, 9.440067985664028]), ('2334*', [-1437776.5013845253, 9.358111758033958]), ('234*', [-1432535.7535183069, 9.268943242061733]), ('24*', [-1425204.691571236, 9.26894324206173])]
TRUTH_C4L4 = [('1121*', [-1490162.871172391, 25.553586157256976]), ('221*', [-1490162.8711723909, 24.553586157256976]), ('2221*', [-1489466.9071525347, 23.64358615725698]), ('2231*', [-1488603.1401359926, 22.738537023683413]), ('2331*', [-1482744.2415537834, 22.468598326365132]), ('2341*', [-1475378.7537851958, 22.370432645136855]), ('2113*', [-1471922.145938075, 11.362072108405588]), ('223*', [-1471895.5491260004, 10.36207210840559]), ('23*', [-1466111.360834025, 10.272072943363703]), ('2343*', [-1459989.7576398463, 10.188573289663607]), ('2234*', [-1442990.0823133066, 9.440067985664028]), ('2334*', [-1437776.5013845253, 9.358111758033958]), ('234*', [-1432535.7535183069, 9.268943242061733]), ('24*', [-1425204.691571236, 9.26894324206173])]
TRUTH_12345 = [('1121*', [-1490162.871172391, 25.553586157256976]), ('221*', [-1490162.8711723909, 24.553586157256976]), ('2221*', [-1489466.9071525347, 23.64358615725698]), ('2231*', [-1488603.1401359926, 22.738537023683413]), ('2331*', [-1482744.2415537834, 22.468598326365132]), ('2341*', [-1475378.7537851958, 22.370432645136855]), ('2113*', [-1471922.145938075, 11.362072108405588]), ('223*', [-1471895.5491260004, 10.36207210840559]), ('23*', [-1466111.360834025, 10.272072943363703]), ('2343*', [-1459989.7576398463, 10.188573289663607]), ('2353*', [-1452372.0628502036, 10.188552523383924]), ('2234*', [-1442990.0823133066, 9.440067985664028]), ('2334*', [-1437776.5013845253, 9.358111758033958]), ('234*', [-1432535.7535183069, 9.268943242061733]), ('24*', [-1425204.691571236, 9.26894324206173]), ('2354*', [-1425055.5828949779, 9.268942257572757]), ('2345*', [-1391849.9571809277, 9.250849926302044]), ('235*', [-1384573.7339997415, 9.250848891530724]), ('5*', [-1349923.035067944, 9.250848891530723])]
TRUTH_L5 = [('1121*', [-1490162.871172391, 25.553586157256976]), ('221*', [-1490162.8711723909, 24.553586157256976]), ('22121*', [-1490093.7154607936, 23.562661157256983]), ('2231*', [-1488603.1401359926, 22.738537023683413]), ('22321*', [-1488014.5012545532, 21.81263881133117]), ('22331*', [-1487110.7462935874, 20.898994148966555]), ('23331*', [-1481126.516983305, 20.64488538064411]), ('23431*', [-1475016.2833382124, 20.384046321063327]), ('22123*', [-1472340.9786725063, 11.348287466249031]), ('223*', [-1471895.5491260004, 10.36207210840559]), ('22343*', [-1466580.9374717209, 10.230013965827665]), ('23343*', [-1460896.5458934568, 10.135201611316385]), ('23443*', [-1455043.3314877455, 10.038252487087178]), ('22334*', [-1446967.14106394, 9.58111947271551]), ('2234*', [-1442990.0823133066, 9.440067985664028]), ('2334*', [-1437776.5013845253, 9.358111758033958]), ('234*', [-1432535.7535183069, 9.268943242061733]), ('24*', [-1425204.691571236, 9.26894324206173])]
TRUTH_C5L5 = [('1121*', [-1490162.871172391, 25.553586157256976]), ('221*', [-1490162.8711723909, 24.553586157256976]), ('22121*', [-1490093.7154607936, 23.562661157256983]), ('2231*', [-1488603.1401359926, 22.738537023683413]), ('22321*', [-1488014.5012545532, 21.81263881133117]), ('22331*', [-1487110.7462935874, 20.898994148966555]), ('23331*', [-1481126.516983305, 20.64488538064411]), ('23431*', [-1475016.2833382124, 20.384046321063327]), ('22123*', [-1472340.9786725063, 11.348287466249031]), ('223*', [-1471895.5491260004, 10.36207210840559]), ('22343*', [-1466580.9374717209, 10.230013965827665]), ('23343*', [-1460896.5458934568, 10.135201611316385]), ('23443*', [-1455043.3314877455, 10.038252487087178]), ('23543*', [-1447450.3330322446, 10.038251498161248]), ('23453*', [-1447370.0097823401, 10.033706050024048]), ('22334*', [-1446967.14106394, 9.58111947271551]), ('2234*', [-1442990.0823133066, 9.440067985664028]), ('2334*', [-1437776.5013845253, 9.358111758033958]), ('234*', [-1432535.7535183069, 9.268943242061733]), ('24*', [-1425204.691571236, 9.26894324206173]), ('2354*', [-1425055.5828949779, 9.268942257572757]), ('23454*', [-1425007.591854658, 9.264434059049426]), ('23554*', [-1417565.1569894108, 9.264433020700213]), ('23445*', [-1399201.0231825132, 9.25057868710398]), ('23545*', [-1391887.9518619953, 9.250577649081068])]

TRUTH_C3L3 = [['221*', [-1490162.8711723909, 24.553586157256976]], ['231*', [-1482999.0148700937, 24.454578742677814]], ['223*', [-1471895.5491260004, 10.36207210840559]], ['23*', [-1466111.360834025, 10.272072943363703]], ['3*', [-1458762.385429146, 10.272072943363701]]]
TRUTH_COSTS_C3L3 = [['221*', [-1490162.8711723909, 24.553586157256976]], ['1*', [-1490162.8711723906, 26.553586157256984]], ['21*', [-1490162.8711723906, 25.553586157256984]], ['231*', [-1482999.0148700937, 24.454578742677814]], ['321*', [-1482693.3360592595, 24.553586157256976]], ['31*', [-1482693.3360592593, 25.553586157256984]], ['121*', [-1482693.3360592593, 26.553586157256984]], ['331*', [-1475565.3890372366, 24.454578742677814]], ['131*', [-1475261.2424606667, 26.553586157256984]], ['223*', [-1471895.5491260004, 10.36207210840559]], ['113*', [-1466111.3608340255, 11.272072943363703]], ['23*', [-1466111.360834025, 10.272072943363703]], ['323*', [-1464517.5801135206, 10.36207210840559]], ['213*', [-1458907.8531120669, 11.272072108405588]], ['3*', [-1458762.385429146, 10.272072943363701]], ['123*', [-1458762.3854291455, 11.272072943363703]], ['313*', [-1451594.9857426966, 11.272072108405588]], ['13*', [-1451450.2472256853, 11.272072943363701]], ['112*', [-1448337.591928803, 17.29809383395393]], ['2*', [-1448337.5919288027, 16.29809383395393]], ['212*', [-1442265.5665031679, 17.20809383395393]], ['232*', [-1442037.6096968977, 16.20359117794156]], ['32*', [-1441077.7086584133, 16.29809383395393]], ['12*', [-1441077.708658413, 17.29809383395393]], ['312*', [-1435036.1196421143, 17.20809383395393]], ['332*', [-1434809.3054836725, 16.20359117794156]], ['132*', [-1433854.2160095149, 17.29809383395393]]]

TRUTH_C2L2 = [['21*', [-1490162.8711723906, 25.553586157256984]], ['2*', [-1448337.5919288027, 16.29809383395393]]]
TRUTH_COSTS_C2L2 = [['1*', [-1490162.8711723906, 26.553586157256984]], ['21*', [-1490162.8711723906, 25.553586157256984]], ['2*', [-1448337.5919288027, 16.29809383395393]], ['12*', [-1441077.708658413, 17.29809383395393]]]

TRUTH_C1L1 = [['1*', [-1490162.8711723906, 26.553586157256984]]]
TRUTH_COSTS_C1L1 = [['1*', [-1490162.8711723906, 26.553586157256984]]]


TRUTH_COSTS_C4L4 = [('1121*', [-1490162.871172391, 25.553586157256976]), ('221*', [-1490162.8711723909, 24.553586157256976]), ('1*', [-1490162.8711723906, 26.553586157256984]), ('21*', [-1490162.8711723906, 25.553586157256984]), ('2221*', [-1489466.9071525347, 23.64358615725698]), ('2231*', [-1488603.1401359926, 22.738537023683413]), ('2121*', [-1483225.5813779982, 25.463586157256977]), ('231*', [-1482999.0148700937, 24.454578742677814]), ('1131*', [-1482999.0148700937, 25.454578742677814]), ('2321*', [-1482964.3910972718, 23.459078742677814]), ('2331*', [-1482744.2415537834, 22.468598326365132]), ('321*', [-1482693.3360592595, 24.553586157256976]), ('1221*', [-1482693.3360592595, 25.553586157256976]), ('31*', [-1482693.3360592593, 25.553586157256984]), ('121*', [-1482693.3360592593, 26.553586157256984]), ('3221*', [-1482000.8606028247, 23.64358615725698]), ('2241*', [-1481178.8955120144, 22.649247056417572]), ('3231*', [-1481141.423272778, 22.738537023683413]), ('3121*', [-1475790.8198662712, 25.463586157256977]), ('2131*', [-1475640.4329844825, 25.463586157256977]), ('1231*', [-1475565.3890372368, 25.454578742677814]), ('331*', [-1475565.3890372366, 24.454578742677814]), ('3321*', [-1475530.9388182536, 23.459078742677814]), ('241*', [-1475415.002651691, 24.454578742677814]), ('1141*', [-1475415.002651691, 25.454578742677814]), ('2421*', [-1475380.552432708, 23.459078742677814]), ('2341*', [-1475378.7537851958, 22.370432645136855]), ('3331*', [-1475311.892788198, 22.468598326365132]), ('41*', [-1475261.2424606672, 25.553586157256984]), ('421*', [-1475261.2424606672, 24.553586157256976]), ('1321*', [-1475261.242460667, 25.553586157256976]), ('131*', [-1475261.2424606667, 26.553586157256984]), ('2431*', [-1475161.5064026525, 22.468598326365132]), ('4221*', [-1474572.2380810096, 23.64358615725698]), ('3241*', [-1473754.393141913, 22.649247056417572]), ('4231*', [-1473717.1087346328, 22.738537023683413]), ('2113*', [-1471922.145938075, 11.362072108405588]), ('1123*', [-1471895.5491260008, 11.36207210840559]), ('223*', [-1471895.5491260004, 10.36207210840559]), ('4121*', [-1468393.3255642185, 25.463586157256977]), ('3131*', [-1468243.692506133, 25.463586157256977]), ('431*', [-1468169.024721393, 24.454578742677814]), ('1331*', [-1468169.0247213927, 25.454578742677814]), ('2213*', [-1468153.2436541691, 11.196475818657829]), ('4321*', [-1468134.7471862994, 23.459078742677814]), ('2141*', [-1468093.3056243437, 25.463586157256977]), ('341*', [-1468019.392157063, 24.454578742677814]), ('1241*', [-1468019.392157063, 25.454578742677814]), ('3421*', [-1467985.1146219694, 23.459078742677814]), ('3341*', [-1467983.3249902911, 22.370432645136855]), ('4331*', [-1467916.7991382459, 22.468598326365132]), ('141*', [-1467866.402698667, 26.553586157256984]), ('1421*', [-1467866.402698667, 25.553586157256976]), ('2441*', [-1467832.938604746, 22.370432645136855]), ('3431*', [-1467767.1665739163, 22.468598326365132]), ('2123*', [-1466408.705859376, 11.272072108405588]), ('4241*', [-1466367.106556894, 22.649247056417572]), ('113*', [-1466111.3608340255, 11.272072943363703]), ('23*', [-1466111.360834025, 10.272072943363703]), ('2243*', [-1465616.228584849, 10.286110505296184]), ('3113*', [-1464544.0436074014, 11.362072108405588]), ('323*', [-1464517.5801135206, 10.36207210840559]), ('1223*', [-1464517.5801135201, 11.36207210840559]), ('2223*', [-1461877.3528783931, 11.128724189611201]), ('4131*', [-1460884.0286546377, 25.463586157256977]), ('1431*', [-1460809.7351468646, 25.454578742677814]), ('3213*', [-1460794.0331832329, 11.196475818657829]), ('3141*', [-1460734.3955965517, 25.463586157256977]), ('441*', [-1460660.852625174, 24.454578742677814]), ('1341*', [-1460660.852625174, 25.454578742677814]), ('2313*', [-1460643.6467976875, 11.196475818657829]), ('4421*', [-1460626.746908381, 23.459078742677814]), ('4341*', [-1460624.9662473442, 22.370432645136855]), ('3441*', [-1460475.3336830148, 22.370432645136855]), ('4431*', [-1460409.8913386262, 22.468598326365132]), ('2343*', [-1459989.7576398463, 10.188573289663607]), ('3123*', [-1459058.2399938556, 11.272072108405588]), ('1113*', [-1458907.853112067, 12.272072108405588]), ('213*', [-1458907.8531120669, 11.272072108405588]), ('3*', [-1458762.385429146, 10.272072943363701]), ('123*', [-1458762.3854291455, 11.272072943363703]), ('1143*', [-1458611.9990436006, 11.272072943363703]), ('243*', [-1458611.9990436004, 10.272072943363703]), ('3243*', [-1458269.735061509, 10.286110505296184]), ('4113*', [-1457202.9244786943, 11.362072108405588]), ('423*', [-1457176.593634741, 10.36207210840559]), ('1323*', [-1457176.593634741, 11.36207210840559]), ('2212*', [-1454562.4917619782, 16.388660680798203]), ('3223*', [-1454549.6007046825, 11.128724189611201]), ('2323*', [-1454498.5290640278, 11.11071946063285]), ('4213*', [-1453471.7112176276, 11.196475818657829]), ('4141*', [-1453412.3725681007, 25.463586157256977]), ('1441*', [-1453339.1982354922, 25.454578742677814]), ('3313*', [-1453322.078653298, 11.196475818657829]), ('2413*', [-1453171.6922677532, 11.196475818657829]), ('4441*', [-1453154.6092186985, 22.370432645136855]), ('2232*', [-1453140.235002809, 15.483121320673867]), ('3343*', [-1452671.4671559858, 10.188573289663607]), ('2443*', [-1452521.0807704409, 10.188573289663607]), ('4123*', [-1451744.618800782, 11.272072108405588]), ('313*', [-1451594.9857426966, 11.272072108405588]), ('1213*', [-1451594.9857426966, 12.272072108405588]), ('13*', [-1451450.2472256853, 11.272072943363701]), ('43*', [-1451450.2472256853, 10.272072943363703]), ('2143*', [-1451444.5988609076, 11.272072108405588]), ('343*', [-1451300.614661356, 10.272072943363703]), ('1243*', [-1451300.6146613555, 11.272072943363703]), ('4243*', [-1450960.0662990008, 10.286110505296184]), ('1423*', [-1449872.4043123855, 11.36207210840559]), ('2112*', [-1448350.9237481316, 17.29809383395393]), ('112*', [-1448337.591928803, 17.29809383395393]), ('2*', [-1448337.5919288027, 16.29809383395393]), ('2312*', [-1448107.2364449599, 16.302645101108837]), ('2332*', [-1447962.523354243, 15.312092969513923]), ('3212*', [-1447271.4057896696, 16.388660680798203]), ('4223*', [-1447258.5793496093, 11.128724189611201]), ('3323*', [-1447207.7637087659, 11.11071946063285]), ('2423*', [-1447057.377323221, 11.11071946063285]), ('2242*', [-1446487.0383991455, 15.389392814878663]), ('4313*', [-1446037.2103297107, 11.196475818657829]), ('3413*', [-1445887.5777653817, 11.196475818657829]), ('3232*', [-1445856.2781819564, 15.483121320673867]), ('4343*', [-1445389.8600634478, 10.188573289663607]), ('3443*', [-1445240.2274991188, 10.188573289663607]), ('413*', [-1444318.7745809462, 11.272072108405588]), ('1313*', [-1444318.7745809462, 12.272072108405588]), ('143*', [-1444174.7615748546, 11.272072943363703]), ('3143*', [-1444169.1415228604, 11.272072108405588]), ('443*', [-1444025.8790531647, 10.272072943363703]), ('1343*', [-1444025.8790531647, 11.272072943363703]), ('2234*', [-1442990.0823133066, 9.440067985664028]), ('212*', [-1442265.5665031679, 17.20809383395393]), ('1112*', [-1442265.5665031674, 18.20809383395393]), ('1132*', [-1442037.609696898, 17.20359117794156]), ('232*', [-1442037.6096968977, 16.20359117794156]), ('2342*', [-1441420.148637307, 15.209072657030335]), ('3112*', [-1441090.9736511593, 17.29809383395393]), ('32*', [-1441077.7086584133, 16.29809383395393]), ('12*', [-1441077.708658413, 17.29809383395393]), ('3312*', [-1440848.507845921, 16.302645101108837]), ('3332*', [-1440704.5201386728, 15.312092969513923]), ('2412*', [-1440698.1214603756, 16.302645101108837]), ('2432*', [-1440554.1337531274, 15.312092969513923]), ('4212*', [-1440016.8668443586, 16.388660680798203]), ('4323*', [-1439953.5437733876, 11.11071946063285]), ('3423*', [-1439803.9112090587, 11.11071946063285]), ('3242*', [-1439236.4311447109, 15.389392814878663]), ('4413*', [-1438639.9753450758, 11.196475818657829]), ('4232*', [-1438608.8326527807, 15.483121320673867]), ('4443*', [-1437995.8699627367, 10.188573289663607]), ('2334*', [-1437776.5013845253, 9.358111758033958]), ('2114*', [-1437570.9978407982, 10.358944961148303]), ('1124*', [-1437531.2025275892, 10.358944961148303]), ('224*', [-1437531.202527589, 9.358944961148303]), ('1413*', [-1437079.0358852695, 12.272072108405588]), ('4143*', [-1436930.1528722984, 11.272072108405588]), ('1443*', [-1436787.6085147425, 11.272072943363703]), ('3234*', [-1435757.003771188, 9.440067985664028]), ('312*', [-1435036.1196421143, 17.20809383395393]), ('1212*', [-1435036.1196421143, 18.20809383395393]), ('2132*', [-1434885.7327603253, 17.20809383395393]), ('332*', [-1434809.3054836725, 16.20359117794156]), ('1232*', [-1434809.3054836725, 17.20359117794156]), ('1142*', [-1434658.9190981272, 17.20359117794156]), ('242*', [-1434658.919098127, 16.20359117794156]), ('3342*', [-1434194.9394864773, 15.209072657030335]), ('2442*', [-1434044.5531009324, 15.209072657030335]), ('4112*', [-1433867.4145106508, 17.29809383395393]), ('132*', [-1433854.2160095149, 17.29809383395393]), ('42*', [-1433854.2160095146, 16.29809383395393]), ('4312*', [-1433626.1640805104, 16.302645101108837]), ('4332*', [-1433482.8981207006, 15.312092969513923]), ('3412*', [-1433476.531516181, 16.302645101108837]), ('3432*', [-1433333.2655563713, 15.312092969513923]), ('2214*', [-1433188.3575476066, 10.268948137576677]), ('2124*', [-1432834.634993155, 10.268944961148303]), ('4423*', [-1432586.8035499887, 11.11071946063285]), ('1134*', [-1432535.7535183073, 10.268943242061733]), ('234*', [-1432535.7535183069, 9.268943242061733]), ('4242*', [-1432022.168015154, 15.389392814878663]), ('3334*', [-1430569.5562447114, 9.358111758033958]), ('2434*', [-1430419.1698591663, 9.358111758033958]), ('3114*', [-1430365.082800422, 10.358944961148303]), ('324*', [-1430325.486963723, 9.358944961148303]), ('1224*', [-1430325.486963723, 10.358944961148303]), ('4234*', [-1428560.1814901738, 9.440067985664028]), ('1312*', [-1427842.9108381362, 18.20809383395393]), ('412*', [-1427842.910838136, 17.20809383395393]), ('3132*', [-1427693.2777800509, 17.20809383395393]), ('432*', [-1427617.233599929, 16.20359117794156]), ('1332*', [-1427617.2335999287, 17.20359117794156]), ('2142*', [-1427542.8908982612, 17.20809383395393]), ('342*', [-1427467.6010355994, 16.20359117794156]), ('1242*', [-1427467.601035599, 17.20359117794156]), ('4342*', [-1427005.947150934, 15.209072657030335]), ('3442*', [-1426856.314586605, 15.209072657030335]), ('142*', [-1426666.931571829, 17.29809383395393]), ('4412*', [-1426291.140245772, 16.302645101108837]), ('4432*', [-1426148.5924155961, 15.312092969513923]), ('3214*', [-1426004.4107673392, 10.268948137576677]), ('2224*', [-1425995.1914461541, 10.256557578695377]), ('2314*', [-1425854.0243817943, 10.268948137576677]), ('3124*', [-1425652.4612694387, 10.268944961148303]), ('2134*', [-1425502.0743876493, 10.268944961148303]), ('334*', [-1425355.077956781, 9.268943242061733]), ('1234*', [-1425355.0779567808, 10.268943242061733]), ('114*', [-1425204.6915712361, 10.26894324206173]), ('24*', [-1425204.691571236, 9.26894324206173]), ('4334*', [-1423398.7363706802, 9.358111758033958]), ('3434*', [-1423249.1038063508, 9.358111758033958]), ('4114*', [-1423195.2878623903, 10.358944961148303]), ('1324*', [-1423155.890502313, 10.358944961148303]), ('424*', [-1423155.8905023127, 9.358944961148303]), ('1412*', [-1420685.758445693, 18.20809383395393]), ('4132*', [-1420536.8754327223, 17.20809383395393]), ('1432*', [-1420461.2124288362, 17.20359117794156]), ('3142*', [-1420387.2423746365, 17.20809383395393]), ('442*', [-1420312.329907146, 16.20359117794156]), ('1342*', [-1420312.329907146, 17.20359117794156]), ('4442*', [-1419704.1075699234, 15.209072657030335]), ('4214*', [-1418856.4739721308, 10.268948137576677]), ('3224*', [-1418847.3008633729, 10.256557578695377]), ('3314*', [-1418706.8414078015, 10.268948137576677]), ('2324*', [-1418696.914477828, 10.256557578695377]), ('2414*', [-1418556.4550222564, 10.268948137576677]), ('4124*', [-1418506.2886432235, 10.268944961148303]), ('3134*', [-1418356.6555851377, 10.268944961148303]), ('1334*', [-1418210.3959831242, 10.268943242061733]), ('434*', [-1418210.3959831237, 9.268943242061733]), ('1114*', [-1418206.2687033494, 11.268944961148302]), ('214*', [-1418206.268703349, 10.268944961148302]), ('34*', [-1418060.7634187948, 9.26894324206173]), ('124*', [-1418060.7634187948, 10.26894324206173]), ('4434*', [-1416114.9781605748, 9.358111758033958]), ('1424*', [-1416022.2320940855, 10.358944961148303]), ('4142*', [-1413267.461989279, 17.20809383395393]), ('1442*', [-1413192.9250252433, 17.20359117794156]), ('4224*', [-1411735.2395316928, 10.256557578695377]), ('4314*', [-1411595.484137976, 10.268948137576677]), ('3324*', [-1411585.6069673637, 10.256557578695377]), ('3414*', [-1411445.851573647, 10.268948137576677]), ('2424*', [-1411435.2205818188, 10.256557578695377]), ('4134*', [-1411247.0536437728, 10.268944961148303]), ('1434*', [-1411101.5271772128, 10.268943242061733]), ('314*', [-1411097.4205856877, 10.268944961148302]), ('1214*', [-1411097.4205856875, 11.268944961148302]), ('4*', [-1410952.6446555236, 9.26894324206173]), ('134*', [-1410952.6446555236, 10.26894324206173]), ('4324*', [-1404509.9453330499, 10.256557578695377]), ('4414*', [-1404370.890472034, 10.268948137576677]), ('3424*', [-1404360.3127687208, 10.256557578695377]), ('414*', [-1404024.2060163158, 10.268944961148302]), ('1314*', [-1404024.2060163156, 11.268944961148302]), ('14*', [-1403880.155784607, 10.26894324206173]), ('4424*', [-1397320.8683760006, 10.256557578695377]), ('1414*', [-1396986.446379831, 11.268944961148302])]
TRUTH_COSTS_C5L5 = [('1121*', [-1490162.871172391, 25.553586157256976]), ('221*', [-1490162.8711723909, 24.553586157256976]), ('1*', [-1490162.8711723906, 26.553586157256984]), ('21*', [-1490162.8711723906, 25.553586157256984]), ('22121*', [-1490093.7154607936, 23.562661157256983]), ('21121*', [-1489480.2389718634, 24.643586157256976]), ('2221*', [-1489466.9071525347, 23.64358615725698]), ('11221*', [-1489466.9071525347, 24.64358615725698]), ('21131*', [-1488629.7369480666, 23.738537023683406]), ('2231*', [-1488603.1401359926, 22.738537023683413]), ('11231*', [-1488603.1401359923, 23.738537023683413]), ('22321*', [-1488014.5012545532, 21.81263881133117]), ('22331*', [-1487110.7462935874, 20.898994148966555]), ('22131*', [-1483639.4424504724, 23.39125550125139]), ('22221*', [-1483518.8433910462, 23.38875362077114]), ('11121*', [-1483225.5813779985, 26.463586157256977]), ('2121*', [-1483225.5813779982, 25.463586157256977]), ('21221*', [-1483225.5813779982, 24.463586157256977]), ('21231*', [-1483041.7081346407, 23.468581367677814]), ('231*', [-1482999.0148700937, 24.454578742677814]), ('1131*', [-1482999.0148700937, 25.454578742677814]), ('2321*', [-1482964.3910972718, 23.459078742677814]), ('11321*', [-1482964.3910972718, 24.459078742677814]), ('23121*', [-1482927.7569119693, 23.4637130878875]), ('11331*', [-1482744.2415537836, 23.468598326365132]), ('2331*', [-1482744.2415537834, 22.468598326365132]), ('321*', [-1482693.3360592595, 24.553586157256976]), ('1221*', [-1482693.3360592595, 25.553586157256976]), ('31*', [-1482693.3360592593, 25.553586157256984]), ('121*', [-1482693.3360592593, 26.553586157256984]), ('32121*', [-1482624.5269950165, 23.562661157256983]), ('23321*', [-1482100.4581105423, 21.550049665483588]), ('31121*', [-1482014.1255955703, 24.643586157256976]), ('3221*', [-1482000.8606028247, 23.64358615725698]), ('12221*', [-1482000.8606028245, 24.64358615725698]), ('21141*', [-1481218.6908252237, 23.64924705641757]), ('2241*', [-1481178.8955120144, 22.649247056417572]), ('11241*', [-1481178.8955120142, 23.649247056417572]), ('31131*', [-1481167.8867666589, 23.738537023683406]), ('22421*', [-1481149.180724056, 21.652993740816775]), ('3231*', [-1481141.423272778, 22.738537023683413]), ('12231*', [-1481141.4232727778, 23.738537023683413]), ('23331*', [-1481126.516983305, 20.64488538064411]), ('22431*', [-1480834.3796356756, 20.66169034211818]), ('32321*', [-1480555.7349807539, 21.81263881133117]), ('22341*', [-1479858.6975980315, 20.73969142469713]), ('32331*', [-1479656.510148369, 20.898994148966555]), ('23221*', [-1476490.5568340058, 23.274195048280887]), ('22231*', [-1476399.5735853466, 23.28974994861517]), ('32131*', [-1476202.60643409, 23.39125550125139]), ('32221*', [-1476082.6118850345, 23.38875362077114]), ('23131*', [-1476052.2200485452, 23.39125550125139]), ('22141*', [-1475938.0587800897, 23.382375835998435]), ('3121*', [-1475790.8198662712, 25.463586157256977]), ('31221*', [-1475790.8198662712, 24.463586157256977]), ('12121*', [-1475790.8198662708, 26.463586157256977]), ('21241*', [-1475677.4978691833, 23.37041173515902]), ('11131*', [-1475640.4329844827, 26.463586157256977]), ('2131*', [-1475640.4329844825, 25.463586157256977]), ('21321*', [-1475640.4329844825, 24.463586157256977]), ('31231*', [-1475607.86829911, 23.468581367677814]), ('1231*', [-1475565.3890372368, 25.454578742677814]), ('331*', [-1475565.3890372366, 24.454578742677814]), ('12321*', [-1475530.9388182538, 24.459078742677814]), ('3321*', [-1475530.9388182536, 23.459078742677814]), ('33121*', [-1475494.488264109, 23.4637130878875]), ('21331*', [-1475457.4814173214, 23.468581367677814]), ('241*', [-1475415.002651691, 24.454578742677814]), ('1141*', [-1475415.002651691, 25.454578742677814]), ('2421*', [-1475380.552432708, 23.459078742677814]), ('11421*', [-1475380.552432708, 24.459078742677814]), ('11341*', [-1475378.7537851962, 23.370432645136855]), ('2341*', [-1475378.7537851958, 22.370432645136855]), ('23421*', [-1475346.1446520316, 21.374570364456936]), ('24121*', [-1475344.101878564, 23.4637130878875]), ('3331*', [-1475311.892788198, 22.468598326365132]), ('12331*', [-1475311.892788198, 23.468598326365132]), ('41*', [-1475261.2424606672, 25.553586157256984]), ('421*', [-1475261.2424606672, 24.553586157256976]), ('1321*', [-1475261.242460667, 25.553586157256976]), ('131*', [-1475261.2424606667, 26.553586157256984]), ('42121*', [-1475192.7783061862, 23.562661157256983]), ('11431*', [-1475161.5064026527, 23.468598326365132]), ('2431*', [-1475161.5064026525, 22.468598326365132]), ('23431*', [-1475016.2833382124, 20.384046321063327]), ('33321*', [-1474671.3363499558, 21.550049665483588]), ('41121*', [-1474585.4365821448, 24.643586157256976]), ('13221*', [-1474572.2380810098, 24.64358615725698]), ('4221*', [-1474572.2380810096, 23.64358615725698]), ('24321*', [-1474520.9499644104, 21.550049665483588]), ('23341*', [-1473914.6987655305, 20.47947625483115]), ('31141*', [-1473793.9889786125, 23.64924705641757]), ('12241*', [-1473754.3931419132, 23.649247056417572]), ('3241*', [-1473754.393141913, 22.649247056417572]), ('41131*', [-1473743.4395785863, 23.738537023683406]), ('32421*', [-1473724.8273011986, 21.652993740816775]), ('4231*', [-1473717.1087346328, 22.738537023683413]), ('13231*', [-1473717.1087346328, 23.738537023683413]), ('33331*', [-1473702.2771638734, 20.64488538064411]), ('22441*', [-1473667.9431263034, 20.487578481878145]), ('24331*', [-1473551.8907783285, 20.64488538064411]), ('21151*', [-1473509.6077779266, 23.64924705641757]), ('11251*', [-1473456.6801218991, 23.649247056417572]), ('2251*', [-1473456.680121899, 22.649247056417572]), ('22521*', [-1473427.1142811843, 21.652993740816775]), ('32431*', [-1473411.6041730724, 20.66169034211818]), ('42321*', [-1473134.3562420083, 21.81263881133117]), ('22531*', [-1473113.8911530585, 20.66169034211818]), ('32341*', [-1472440.8128030058, 20.73969142469713]), ('22123*', [-1472340.9786725063, 11.348287466249031]), ('42331*', [-1472239.6388306513, 20.898994148966555]), ('22213*', [-1472232.271023961, 11.353816748249992]), ('22351*', [-1472020.2642820075, 20.735643780880782]), ('11113*', [-1471922.1459380751, 12.362072108405588]), ('2113*', [-1471922.145938075, 11.362072108405588]), ('1123*', [-1471895.5491260008, 11.36207210840559]), ('223*', [-1471895.5491260004, 10.36207210840559]), ('22313*', [-1469172.0747023765, 11.15434639117689]), ('33221*', [-1469089.5550563936, 23.274195048280887]), ('32231*', [-1468999.0278669908, 23.28974994861517]), ('24221*', [-1468939.1686708485, 23.274195048280887]), ('23231*', [-1468848.6414814456, 23.28974994861517]), ('42131*', [-1468803.0480259673, 23.39125550125139]), ('42221*', [-1468683.6549571357, 23.38875362077114]), ('33131*', [-1468653.4154616382, 23.39125550125139]), ('22241*', [-1468604.906183406, 23.279977383689562]), ('32141*', [-1468539.8264337212, 23.382375835998435]), ('24131*', [-1468503.029076093, 23.39125550125139]), ('13121*', [-1468393.3255642187, 26.463586157256977]), ('4121*', [-1468393.3255642185, 25.463586157256977]), ('41221*', [-1468393.3255642185, 24.463586157256977]), ('23141*', [-1468389.4400481759, 23.382375835998435]), ('31241*', [-1468280.5716007682, 23.37041173515902]), ('12131*', [-1468243.6925061333, 26.463586157256977]), ('3131*', [-1468243.692506133, 25.463586157256977]), ('31321*', [-1468243.692506133, 24.463586157256977]), ('22151*', [-1468242.112524488, 23.382375835998435]), ('41231*', [-1468211.2910532942, 23.468581367677814]), ('431*', [-1468169.024721393, 24.454578742677814]), ('1331*', [-1468169.0247213927, 25.454578742677814]), ('11213*', [-1468153.2436541694, 12.196475818657829]), ('2213*', [-1468153.2436541691, 11.196475818657829]), ('21113*', [-1468153.243654169, 12.19647581865783]), ('4321*', [-1468134.7471862994, 23.459078742677814]), ('13321*', [-1468134.747186299, 24.459078742677814]), ('21341*', [-1468130.1847189795, 23.37041173515902]), ('43121*', [-1468098.4793428497, 23.4637130878875]), ('11141*', [-1468093.305624344, 26.463586157256977]), ('2141*', [-1468093.3056243437, 25.463586157256977]), ('21421*', [-1468093.3056243437, 24.463586157256977]), ('31331*', [-1468061.6579952093, 23.468581367677814]), ('341*', [-1468019.392157063, 24.454578742677814]), ('1241*', [-1468019.392157063, 25.454578742677814]), ('3421*', [-1467985.1146219694, 23.459078742677814]), ('12421*', [-1467985.1146219692, 24.459078742677814]), ('21251*', [-1467983.5297935521, 23.37041173515902]), ('3341*', [-1467983.3249902911, 22.370432645136855]), ('12341*', [-1467983.324990291, 23.370432645136855]), ('33421*', [-1467950.8793124577, 21.374570364456936]), ('34121*', [-1467948.8467785204, 23.4637130878875]), ('4331*', [-1467916.7991382459, 22.468598326365132]), ('13331*', [-1467916.7991382456, 23.468598326365132]), ('21431*', [-1467911.2711134197, 23.468581367677814]), ('23441*', [-1467894.4521537987, 20.202413924109898]), ('1151*', [-1467869.0057715185, 25.454578742677814]), ('251*', [-1467869.0057715182, 24.454578742677814]), ('521*', [-1467866.4026986673, 24.553586157256976]), ('51*', [-1467866.402698667, 25.553586157256984]), ('141*', [-1467866.402698667, 26.553586157256984]), ('1421*', [-1467866.402698667, 25.553586157256976]), ('2521*', [-1467834.7282364247, 23.459078742677814]), ('11521*', [-1467834.7282364247, 24.459078742677814]), ('2441*', [-1467832.938604746, 22.370432645136855]), ('11441*', [-1467832.9386047458, 23.370432645136855]), ('24421*', [-1467800.4929269126, 21.374570364456936]), ('25121*', [-1467798.4603929755, 23.4637130878875]), ('52121*', [-1467798.2817250662, 23.562661157256983]), ('3431*', [-1467767.1665739163, 22.468598326365132]), ('12431*', [-1467767.1665739159, 23.468598326365132]), ('22223*', [-1467754.1513115864, 11.212585352912901]), ('2351*', [-1467683.636501202, 22.370407761022523]), ('11351*', [-1467683.636501202, 23.370407761022523]), ('23521*', [-1467651.3257292216, 21.37452276593647]), ('33431*', [-1467622.67144922, 20.384046321063327]), ('11531*', [-1467616.7801883717, 23.468598326365132]), ('2531*', [-1467616.7801883714, 22.468598326365132]), ('24431*', [-1467472.285063675, 20.384046321063327]), ('23531*', [-1467323.2220258908, 20.38402761233011]), ('43321*', [-1467279.453529437, 21.550049665483588]), ('51121*', [-1467193.9843396144, 24.643586157256976]), ('14221*', [-1467180.8519967964, 24.64358615725698]), ('5221*', [-1467180.8519967962, 23.64358615725698]), ('34321*', [-1467129.8209651075, 21.550049665483588]), ('25321*', [-1466979.4345795624, 21.550049665483588]), ('22343*', [-1466580.9374717209, 10.230013965827665]), ('33341*', [-1466526.6086384908, 20.47947625483115]), ('2123*', [-1466408.705859376, 11.272072108405588]), ('11123*', [-1466408.7058593757, 12.272072108405588]), ('41141*', [-1466406.5039169716, 23.64924705641757]), ('24341*', [-1466376.2222529456, 20.47947625483115]), ('4241*', [-1466367.106556894, 22.649247056417572]), ('13241*', [-1466367.106556894, 23.649247056417572]), ('51131*', [-1466356.2078989923, 23.738537023683406]), ('42421*', [-1466337.688916816, 21.652993740816775]), ('14231*', [-1466330.0090400502, 23.738537023683413]), ('5231*', [-1466330.00904005, 22.738537023683413]), ('43331*', [-1466315.2518134722, 20.64488538064411]), ('23213*', [-1466294.172952132, 11.270618667271545]), ('32441*', [-1466281.0898774245, 20.487578481878145]), ('23123*', [-1466280.9567940237, 11.271080691106144]), ('34331*', [-1466165.6192491432, 20.64488538064411]), ('31151*', [-1466123.54819494, 23.64924705641757]), ('113*', [-1466111.3608340255, 11.272072943363703]), ('23*', [-1466111.360834025, 10.272072943363703]), ('23351*', [-1466093.9031496118, 20.475000740963115]), ('3251*', [-1466070.8858421172, 22.649247056417572]), ('12251*', [-1466070.885842117, 23.649247056417572]), ('32521*', [-1466041.4682020384, 21.652993740816775]), ('42431*', [-1466026.035839319, 20.66169034211818]), ('25331*', [-1466015.232863598, 20.64488538064411]), ('22541*', [-1465983.3768574104, 20.487578481878145]), ('22451*', [-1465859.8969362464, 20.483115023851386]), ('52321*', [-1465750.1776309463, 21.81263881133117]), ('32531*', [-1465729.815124542, 20.66169034211818]), ('21143*', [-1465656.0238980588, 11.286110505296184]), ('11243*', [-1465616.2285848495, 11.286110505296184]), ('2243*', [-1465616.228584849, 10.286110505296184]), ('42341*', [-1465060.1106220514, 20.73969142469713]), ('32123*', [-1464960.7769164096, 11.348287466249031]), ('52331*', [-1464859.9450468854, 20.898994148966555]), ('32213*', [-1464852.6141717895, 11.353816748249992]), ('32351*', [-1464641.670126964, 20.735643780880782]), ('3113*', [-1464544.0436074014, 11.362072108405588]), ('12113*', [-1464544.0436074014, 12.362072108405588]), ('323*', [-1464517.5801135206, 10.36207210840559]), ('1223*', [-1464517.5801135201, 11.36207210840559]), ('23313*', [-1463678.7965810117, 11.051997981496445]), ('22323*', [-1462979.5820875736, 11.078983070384934]), ('22113*', [-1462075.1627499901, 12.121206326957532]), ('23223*', [-1462005.698795934, 11.122298770757636]), ('21123*', [-1461890.6846977219, 12.128724189611201]), ('2223*', [-1461877.3528783931, 11.128724189611201]), ('11223*', [-1461877.352878393, 12.128724189611201]), ('32313*', [-1461807.7572767332, 11.15434639117689]), ('43221*', [-1461725.651265666, 23.274195048280887]), ('42231*', [-1461635.5778494936, 23.28974994861517]), ('34221*', [-1461576.0187013366, 23.274195048280887]), ('22413*', [-1461542.7549191753, 11.138016217027364]), ('33231*', [-1461485.9452851643, 23.28974994861517]), ('52131*', [-1461440.580369749, 23.39125550125139]), ('25221*', [-1461425.6323157912, 23.274195048280887]), ('24231*', [-1461335.558899619, 23.28974994861517]), ('52221*', [-1461321.7857661839, 23.38875362077114]), ('43131*', [-1461291.6978480597, 23.39125550125139]), ('32241*', [-1461243.4317256357, 23.279977383689562]), ('42141*', [-1461178.6781922888, 23.382375835998435]), ('34131*', [-1461142.0652837304, 23.39125550125139]), ('23241*', [-1461093.0453400905, 23.279977383689562]), ('5121*', [-1461032.9116676084, 25.463586157256977]), ('14121*', [-1461032.9116676084, 26.463586157256977]), ('51221*', [-1461032.9116676084, 24.463586157256977]), ('33141*', [-1461029.0456279593, 23.382375835998435]), ('25131*', [-1460991.6788981853, 23.39125550125139]), ('41241*', [-1460920.7228904914, 23.37041173515902]), ('23343*', [-1460896.5458934568, 10.135201611316385]), ('4131*', [-1460884.0286546377, 25.463586157256977]), ('41321*', [-1460884.0286546377, 24.463586157256977]), ('13131*', [-1460884.0286546375, 26.463586157256977]), ('32151*', [-1460882.45659275, 23.382375835998435]), ('24141*', [-1460878.6592424142, 23.382375835998435]), ('51231*', [-1460851.7896161191, 23.468581367677814]), ('21213*', [-1460816.048055965, 12.188957956004163]), ('22251*', [-1460812.9917006919, 23.279977383689562]), ('1431*', [-1460809.7351468646, 25.454578742677814]), ('531*', [-1460809.7351468643, 24.454578742677814]), ('3213*', [-1460794.0331832329, 11.196475818657829]), ('31113*', [-1460794.0331832326, 12.19647581865783]), ('12213*', [-1460794.0331832326, 12.196475818657829]), ('5321*', [-1460775.6294300715, 23.459078742677814]), ('14321*', [-1460775.6294300715, 24.459078742677814]), ('31341*', [-1460771.089832406, 23.37041173515902]), ('53121*', [-1460739.5433814677, 23.4637130878875]), ('3141*', [-1460734.3955965517, 25.463586157256977]), ('31421*', [-1460734.3955965517, 24.463586157256977]), ('12141*', [-1460734.3955965515, 26.463586157256977]), ('23151*', [-1460732.0702072044, 23.382375835998435]), ('41331*', [-1460702.9066031484, 23.468581367677814]), ('441*', [-1460660.852625174, 24.454578742677814]), ('1341*', [-1460660.852625174, 25.454578742677814]), ('2313*', [-1460643.6467976875, 11.196475818657829]), ('11313*', [-1460643.6467976875, 12.196475818657829]), ('22143*', [-1460634.4878847615, 11.188921626083413]), ('4421*', [-1460626.746908381, 23.459078742677814]), ('13421*', [-1460626.746908381, 24.459078742677814]), ('31251*', [-1460625.170024016, 23.37041173515902]), ('4341*', [-1460624.9662473442, 22.370432645136855]), ('13341*', [-1460624.9662473442, 23.370432645136855]), ('21441*', [-1460620.7029506168, 23.37041173515902]), ('43421*', [-1460592.6832055117, 21.374570364456936]), ('44121*', [-1460590.6608597783, 23.4637130878875]), ('11151*', [-1460584.0087147634, 26.463586157256977]), ('2151*', [-1460584.0087147632, 25.463586157256977]), ('21521*', [-1460584.0087147632, 24.463586157256977]), ('14331*', [-1460558.7738603163, 23.468598326365132]), ('5331*', [-1460558.773860316, 22.468598326365132]), ('31431*', [-1460553.2735450626, 23.468581367677814]), ('22443*', [-1460549.758126541, 10.141811829086237]), ('33441*', [-1460536.5388915343, 20.202413924109898]), ('1251*', [-1460511.2200608454, 25.454578742677814]), ('351*', [-1460511.2200608451, 24.454578742677814]), ('1521*', [-1460508.6300360607, 25.553586157256976]), ('151*', [-1460508.6300360605, 26.553586157256984]), ('3521*', [-1460477.1143440523, 23.459078742677814]), ('12521*', [-1460477.1143440523, 24.459078742677814]), ('3441*', [-1460475.3336830148, 22.370432645136855]), ('12441*', [-1460475.3336830148, 23.370432645136855]), ('21351*', [-1460474.7831422272, 23.37041173515902]), ('34421*', [-1460443.0506411823, 21.374570364456936]), ('35121*', [-1460441.0282954492, 23.4637130878875]), ('13431*', [-1460409.8913386264, 23.468598326365132]), ('4431*', [-1460409.8913386262, 22.468598326365132]), ('21531*', [-1460402.8866632737, 23.468581367677814]), ('32223*', [-1460396.9413161173, 11.212585352912901]), ('24441*', [-1460386.1525059887, 20.202413924109898]), ('3351*', [-1460326.779965655, 22.370407761022523]), ('12351*', [-1460326.779965655, 23.370407761022523]), ('2541*', [-1460324.94729747, 22.370432645136855]), ('11541*', [-1460324.94729747, 23.370432645136855]), ('33521*', [-1460294.6311534513, 21.37452276593647]), ('25421*', [-1460292.6642556372, 21.374570364456936]), ('21243*', [-1460288.728938946, 11.188556312721769]), ('43431*', [-1460266.1205048303, 20.384046321063327]), ('12531*', [-1460260.2587742973, 23.468598326365132]), ('3531*', [-1460260.258774297, 22.468598326365132]), ('23541*', [-1460237.0176194226, 20.202413357554597]), ('11451*', [-1460176.3935801103, 23.370407761022523]), ('2451*', [-1460176.39358011, 22.370407761022523]), ('24521*', [-1460144.2447679061, 21.37452276593647]), ('34431*', [-1460116.4879405012, 20.384046321063327]), ('23451*', [-1460102.9929885636, 20.19750673647386]), ('11343*', [-1459989.7576398465, 11.188573289663607]), ('2343*', [-1459989.7576398463, 10.188573289663607]), ('33531*', [-1459968.1720905693, 20.38402761233011]), ('25431*', [-1459966.1015549558, 20.384046321063327]), ('53321*', [-1459924.6229864561, 21.550049665483588]), ('15221*', [-1459826.5157001994, 24.64358615725698]), ('24531*', [-1459817.7857050244, 20.38402761233011]), ('44321*', [-1459775.7404647665, 21.550049665483588]), ('35321*', [-1459626.1079004377, 21.550049665483588]), ('32343*', [-1459229.608284412, 10.230013965827665]), ('43341*', [-1459175.551777875, 20.47947625483115]), ('12123*', [-1459058.2399938558, 12.272072108405588]), ('3123*', [-1459058.2399938556, 11.272072108405588]), ('51141*', [-1459056.0490888262, 23.64924705641757]), ('34341*', [-1459025.9192135457, 20.47947625483115]), ('5241*', [-1459016.849210494, 22.649247056417572]), ('14241*', [-1459016.849210494, 23.649247056417572]), ('52421*', [-1458987.5790281866, 21.652993740816775]), ('15231*', [-1458979.937647286, 23.738537023683413]), ('53331*', [-1458965.2543922346, 20.64488538064411]), ('33213*', [-1458944.281190013, 11.270618667271545]), ('42441*', [-1458931.2636950403, 20.487578481878145]), ('33123*', [-1458931.1312787281, 11.271080691106144]), ('1113*', [-1458907.853112067, 12.272072108405588]), ('213*', [-1458907.8531120669, 11.272072108405588]), ('25341*', [-1458875.5328280004, 20.47947625483115]), ('22353*', [-1458853.8191209824, 10.226264269241685]), ('44331*', [-1458816.371870545, 20.64488538064411]), ('24213*', [-1458793.894804468, 11.270618667271545]), ('24123*', [-1458780.7448931828, 11.271080691106144]), ('41151*', [-1458774.5117001478, 23.64924705641757]), ('3*', [-1458762.385429146, 10.272072943363701]), ('123*', [-1458762.3854291455, 11.272072943363703]), ('33351*', [-1458745.015252473, 20.475000740963115]), ('13251*', [-1458722.1133206803, 23.649247056417572]), ('4251*', [-1458722.11332068, 22.649247056417572]), ('42521*', [-1458692.8431383725, 21.652993740816775]), ('52431*', [-1458677.4881313418, 20.66169034211818]), ('35331*', [-1458666.739306216, 20.64488538064411]), ('32541*', [-1458635.0429802632, 20.487578481878145]), ('1143*', [-1458611.9990436006, 11.272072943363703]), ('243*', [-1458611.9990436004, 10.272072943363703]), ('24351*', [-1458594.6288669284, 20.475000740963115]), ('32451*', [-1458512.1820099696, 20.483115023851386]), ('22312*', [-1458498.5481248607, 15.628380394152776]), ('42531*', [-1458382.7522415277, 20.66169034211818]), ('31143*', [-1458309.3308982085, 11.286110505296184]), ('3243*', [-1458269.735061509, 10.286110505296184]), ('12243*', [-1458269.735061509, 11.286110505296184]), ('22551*', [-1458214.468989956, 20.483115023851386]), ('21153*', [-1458024.9496975234, 11.286110505296184]), ('11253*', [-1457972.0220414957, 11.286110505296184]), ('2253*', [-1457972.0220414952, 10.286110505296184]), ('52341*', [-1457716.4046749761, 20.73969142469713]), ('42123*', [-1457617.5688857813, 11.348287466249031]), ('23323*', [-1457513.528267052, 10.976088810772813]), ('42213*', [-1457509.9483137215, 11.353816748249992]), ('42351*', [-1457300.0616391876, 20.735643780880782]), ('4113*', [-1457202.9244786943, 11.362072108405588]), ('13113*', [-1457202.9244786943, 12.362072108405588]), ('423*', [-1457176.593634741, 10.36207210840559]), ('1323*', [-1457176.593634741, 11.36207210840559]), ('22332*', [-1457138.9737053295, 14.71085741999978]), ('33313*', [-1456342.0145574424, 11.051997981496445]), ('24313*', [-1456191.6281718975, 11.051997981496445]), ('23413*', [-1456078.2690165613, 11.034037422094219]), ('32323*', [-1455646.304920629, 11.078983070384934]), ('22423*', [-1455357.1510842296, 11.062921062012627]), ('23443*', [-1455043.3314877455, 10.038252487087178]), ('32113*', [-1454746.4190418571, 12.121206326957532]), ('21223*', [-1454697.319149502, 12.114921082079972]), ('33223*', [-1454677.303280239, 11.122298770757636]), ('23113*', [-1454596.0326563118, 12.121206326957532]), ('31123*', [-1454562.8656974284, 12.128724189611201]), ('2212*', [-1454562.4917619782, 16.388660680798203]), ('21112*', [-1454562.4917619782, 17.388660680798203]), ('11212*', [-1454562.4917619782, 17.388660680798203]), ('22212*', [-1454552.0326899688, 16.388269517634285]), ('3223*', [-1454549.6007046825, 11.128724189611201]), ('12223*', [-1454549.6007046823, 12.128724189611201]), ('24223*', [-1454526.9168946939, 11.122298770757636]), ('11323*', [-1454498.529064028, 12.11071946063285]), ('2323*', [-1454498.5290640278, 11.11071946063285]), ('42313*', [-1454480.353955353, 11.15434639117689]), ('53221*', [-1454398.65950583, 23.274195048280887]), ('52231*', [-1454309.0375883211, 23.28974994861517]), ('22243*', [-1454270.4606904134, 11.12041038100569]), ('44221*', [-1454249.7769841405, 23.274195048280887]), ('32413*', [-1454216.6799387787, 11.138016217027364]), ('43231*', [-1454160.1550666315, 23.28974994861517]), ('35221*', [-1454100.144419811, 23.274195048280887]), ('34231*', [-1454010.522502302, 23.28974994861517]), ('53131*', [-1453966.8813070217, 23.39125550125139]), ('22513*', [-1453918.9669187649, 11.138016217027364]), ('42241*', [-1453918.8571215721, 23.279977383689562]), ('25231*', [-1453860.1361167566, 23.28974994861517]), ('52141*', [-1453854.4281693837, 23.382375835998435]), ('44131*', [-1453817.9987853323, 23.39125550125139]), ('33241*', [-1453769.224557243, 23.279977383689562]), ('15121*', [-1453709.3923085763, 26.463586157256977]), ('43141*', [-1453705.5456476938, 23.382375835998435]), ('35131*', [-1453668.366221003, 23.39125550125139]), ('24241*', [-1453618.8381716977, 23.279977383689562]), ('51241*', [-1453597.7658847605, 23.37041173515902]), ('33343*', [-1453573.7100764443, 10.135201611316385]), ('14131*', [-1453561.255581072, 26.463586157256977]), ('5131*', [-1453561.2555810716, 25.463586157256977]), ('51321*', [-1453561.2555810716, 24.463586157256977]), ('42151*', [-1453559.691399243, 23.382375835998435]), ('34141*', [-1453555.9130833645, 23.382375835998435]), ('31213*', [-1453493.6157394256, 12.188957956004163]), ('32251*', [-1453490.5747043258, 23.279977383689562]), ('1531*', [-1453487.3344741787, 25.454578742677814]), ('4213*', [-1453471.7112176276, 11.196475818657829]), ('13213*', [-1453471.7112176276, 12.196475818657829]), ('41113*', [-1453471.7112176274, 12.19647581865783]), ('15321*', [-1453453.3997144364, 24.459078742677814]), ('41341*', [-1453448.8828717899, 23.37041173515902]), ('24343*', [-1453423.3236908992, 10.135201611316385]), ('4141*', [-1453412.3725681007, 25.463586157256977]), ('41421*', [-1453412.3725681007, 24.463586157256977]), ('13141*', [-1453412.3725681005, 26.463586157256977]), ('33151*', [-1453410.058834914, 23.382375835998435]), ('25141*', [-1453405.5266978194, 23.382375835998435]), ('51331*', [-1453381.0414152571, 23.468581367677814]), ('21313*', [-1453343.2288576362, 12.188957956004163]), ('23251*', [-1453340.1883187806, 23.279977383689562]), ('541*', [-1453339.1982354922, 24.454578742677814]), ('1441*', [-1453339.1982354922, 25.454578742677814]), ('3313*', [-1453322.078653298, 11.196475818657829]), ('12313*', [-1453322.078653298, 12.196475818657829]), ('32143*', [-1453312.9656499994, 11.188921626083413]), ('14421*', [-1453305.2634757499, 24.459078742677814]), ('5421*', [-1453305.2634757496, 23.459078742677814]), ('41251*', [-1453303.6944956167, 23.37041173515902]), ('14341*', [-1453303.4917403883, 23.370432645136855]), ('5341*', [-1453303.491740388, 22.370432645136855]), ('31441*', [-1453299.249813704, 23.37041173515902]), ('53421*', [-1453271.370519333, 21.374570364456936]), ('54121*', [-1453269.3583107356, 23.4637130878875]), ('12151*', [-1453262.7395100154, 26.463586157256977]), ('3151*', [-1453262.7395100151, 25.463586157256977]), ('31521*', [-1453262.7395100151, 24.463586157256977]), ('24151*', [-1453259.6724493683, 23.382375835998435]), ('15331*', [-1453237.631146863, 23.468598326365132]), ('41431*', [-1453232.1584022855, 23.468581367677814]), ('32443*', [-1453228.660605021, 10.141811829086237]), ('23312*', [-1453223.5031162363, 15.462372684492793]), ('43441*', [-1453215.5076322607, 20.202413924109898]), ('451*', [-1453190.315713803, 24.454578742677814]), ('1351*', [-1453190.315713803, 25.454578742677814]), ('23353*', [-1453190.0135154526, 10.13105307111866]), ('11413*', [-1453171.6922677534, 12.196475818657829]), ('2413*', [-1453171.6922677532, 11.196475818657829]), ('21132*', [-1453166.831814883, 16.483121320673867]), ('23143*', [-1453162.5792644543, 11.188921626083413]), ('4521*', [-1453156.3809540605, 23.459078742677814]), ('13521*', [-1453156.3809540605, 24.459078742677814]), ('4441*', [-1453154.6092186985, 22.370432645136855]), ('13441*', [-1453154.6092186985, 23.370432645136855]), ('31351*', [-1453154.0614375314, 23.37041173515902]), ('21541*', [-1453148.8629319156, 23.37041173515902]), ('2232*', [-1453140.235002809, 15.483121320673867]), ('11232*', [-1453140.235002809, 16.483121320673867]), ('44421*', [-1453122.4879976436, 21.374570364456936]), ('45121*', [-1453120.475789046, 23.4637130878875]), ('5431*', [-1453089.494908177, 22.468598326365132]), ('14431*', [-1453089.494908177, 23.468598326365132]), ('31531*', [-1453082.5253442004, 23.468581367677814]), ('42223*', [-1453076.6097984703, 11.212585352912901]), ('34441*', [-1453065.8750679314, 20.202413924109898]), ('22153*', [-1453015.2517407662, 11.188921626083413]), ('4351*', [-1453006.8001361901, 22.370407761022523]), ('13351*', [-1453006.80013619, 23.370407761022523]), ('3541*', [-1453004.9766543694, 22.370432645136855]), ('12541*', [-1453004.9766543694, 23.370432645136855]), ('21451*', [-1453003.6745557424, 23.37041173515902]), ('43521*', [-1452974.8124719297, 21.37452276593647]), ('35421*', [-1452972.855433314, 21.374570364456936]), ('31243*', [-1452968.9398426453, 11.188556312721769]), ('53431*', [-1452946.4447347277, 20.384046321063327]), ('4531*', [-1452940.612386488, 22.468598326365132]), ('13531*', [-1452940.6123864874, 23.468598326365132]), ('22543*', [-1452930.9475850074, 10.141811829086237]), ('33541*', [-1452917.4877293636, 20.202413357554597]), ('25441*', [-1452915.4886823862, 20.202413924109898]), ('3451*', [-1452857.167571861, 22.370407761022523]), ('12451*', [-1452857.1675718608, 23.370407761022523]), ('22453*', [-1452856.576604276, 10.13767650341595]), ('34521*', [-1452825.1799076004, 21.37452276593647]), ('21343*', [-1452818.5529608566, 11.188556312721769]), ('44431*', [-1452797.5622130383, 20.384046321063327]), ('33451*', [-1452784.1349053958, 20.19750673647386]), ('24541*', [-1452767.1013438187, 20.202413357554597]), ('2551*', [-1452706.7811863155, 22.370407761022523]), ('11551*', [-1452706.7811863155, 23.370407761022523]), ('25521*', [-1452674.793522055, 21.37452276593647]), ('21253*', [-1452671.8980354292, 11.188556312721769]), ('12343*', [-1452671.467155986, 11.188573289663607]), ('3343*', [-1452671.4671559858, 10.188573289663607]), ('43531*', [-1452649.9898056325, 20.38402761233011]), ('35431*', [-1452647.9296487088, 20.384046321063327]), ('24451*', [-1452633.7485198507, 20.19750673647386]), ('11443*', [-1452521.080770441, 11.188573289663607]), ('2443*', [-1452521.0807704409, 10.188573289663607]), ('34531*', [-1452500.3572413032, 20.38402761233011]), ('23551*', [-1452484.6131179705, 20.197506111560305]), ('54321*', [-1452458.5227554569, 21.550049665483588]), ('2353*', [-1452372.0628502036, 10.188552523383924]), ('11353*', [-1452372.0628502036, 11.188552523383924]), ('25531*', [-1452349.970855758, 20.38402761233011]), ('45321*', [-1452309.6402337672, 21.550049665483588]), ('42343*', [-1451915.1280970038, 10.230013965827665]), ('53341*', [-1451861.342552106, 20.47947625483115]), ('4123*', [-1451744.618800782, 11.272072108405588]), ('13123*', [-1451744.618800782, 12.272072108405588]), ('23332*', [-1451743.8234806855, 14.552503040524357]), ('44341*', [-1451712.4600304163, 20.47947625483115]), ('15241*', [-1451703.435491325, 23.649247056417572]), ('43213*', [-1451631.2312226112, 11.270618667271545]), ('22412*', [-1451627.9486450101, 15.556732536613483]), ('52441*', [-1451618.2789786505, 20.487578481878145]), ('43123*', [-1451618.1472260836, 11.271080691106144]), ('313*', [-1451594.9857426966, 11.272072108405588]), ('1213*', [-1451594.9857426966, 12.272072108405588]), ('35341*', [-1451562.8274660867, 20.47947625483115]), ('32353*', [-1451541.222600391, 10.226264269241685]), ('54331*', [-1451503.9630566514, 20.64488538064411]), ('34213*', [-1451481.5986582816, 11.270618667271545]), ('34123*', [-1451468.514661754, 11.271080691106144]), ('51151*', [-1451462.312712991, 23.64924705641757]), ('13*', [-1451450.2472256853, 11.272072943363701]), ('43*', [-1451450.2472256853, 10.272072943363703]), ('2143*', [-1451444.5988609076, 11.272072108405588]), ('11143*', [-1451444.5988609076, 12.272072108405588]), ('43351*', [-1451432.9641181156, 20.475000740963115]), ('22432*', [-1451431.0385182905, 14.55269527021214]), ('5251*', [-1451410.176983696, 22.649247056417572]), ('14251*', [-1451410.176983696, 23.649247056417572]), ('52521*', [-1451381.0535200182, 21.652993740816775]), ('45331*', [-1451355.080534962, 20.64488538064411]), ('25213*', [-1451331.2122727365, 11.270618667271545]), ('42541*', [-1451323.5430888366, 20.487578481878145]), ('25123*', [-1451318.1282762089, 11.271080691106144]), ('343*', [-1451300.614661356, 10.272072943363703]), ('1243*', [-1451300.6146613555, 11.272072943363703]), ('34351*', [-1451283.3315537865, 20.475000740963115]), ('42451*', [-1451201.2979668838, 20.483115023851386]), ('32312*', [-1451187.7324224813, 15.628380394152776]), ('253*', [-1451150.228275811, 10.272072943363703]), ('1153*', [-1451150.228275811, 11.272072943363703]), ('25351*', [-1451132.9451682416, 20.475000740963115]), ('22342*', [-1451077.6658670814, 14.563764675020478]), ('52531*', [-1451072.5169732964, 20.66169034211818]), ('41143*', [-1450999.4636590783, 11.286110505296184]), ('4243*', [-1450960.0662990008, 10.286110505296184]), ('13243*', [-1450960.0662990005, 11.286110505296184]), ('32551*', [-1450905.0772521074, 20.483115023851386]), ('31153*', [-1450716.5079370472, 11.286110505296184]), ('3253*', [-1450663.8455842237, 10.286110505296184]), ('12253*', [-1450663.8455842237, 11.286110505296184]), ('52123*', [-1450311.1691472456, 11.348287466249031]), ('33323*', [-1450207.6500386614, 10.976088810772813]), ('52213*', [-1450204.0880300722, 11.353816748249992]), ('24323*', [-1450057.263653116, 10.976088810772813]), ('52351*', [-1449995.2534256945, 20.735643780880782]), ('23423*', [-1449917.1870543307, 10.958433848456583]), ('5113*', [-1449898.6031713276, 11.362072108405588]), ('14113*', [-1449898.6031713276, 12.362072108405588]), ('1423*', [-1449872.4043123855, 11.36207210840559]), ('523*', [-1449872.4043123852, 10.36207210840559]), ('32332*', [-1449834.9729552364, 14.71085741999978]), ('22112*', [-1449567.029405851, 17.216647995518553]), ('22132*', [-1449425.463172834, 16.226210680798204]), ('43313*', [-1449042.0086152016, 11.051997981496445]), ('34313*', [-1448892.3760508725, 11.051997981496445]), ('33413*', [-1448779.5851154318, 11.034037422094219]), ('25313*', [-1448741.9896653271, 11.051997981496445]), ('24413*', [-1448629.1987298867, 11.034037422094219]), ('23513*', [-1448479.7944567748, 11.03403360147314]), ('11112*', [-1448350.923748132, 18.29809383395393]), ('2112*', [-1448350.9237481316, 17.29809383395393]), ('42323*', [-1448349.786266698, 11.078983070384934]), ('112*', [-1448337.591928803, 17.29809383395393]), ('2*', [-1448337.5919288027, 16.29809383395393]), ('21212*', [-1448337.0585730947, 17.30721484236283]), ('21232*', [-1448259.850563201, 16.312121320673867]), ('23212*', [-1448221.4644517652, 16.29400916987495]), ('11312*', [-1448107.23644496, 17.302645101108837]), ('2312*', [-1448107.2364449599, 16.302645101108837]), ('32423*', [-1448062.0818320892, 11.062921062012627]), ('2332*', [-1447962.523354243, 15.312092969513923]), ('11332*', [-1447962.523354243, 16.312092969513923]), ('22523*', [-1447764.3688120756, 11.062921062012627]), ('33443*', [-1447749.83527607, 10.038252487087178]), ('24443*', [-1447599.4488905247, 10.038252487087178]), ('42113*', [-1447454.4111224904, 12.121206326957532]), ('23543*', [-1447450.3330322446, 10.038251498161248]), ('31223*', [-1447405.5573464339, 12.114921082079972]), ('43223*', [-1447385.6418079745, 11.122298770757636]), ('23453*', [-1447370.0097823401, 10.033706050024048]), ('33113*', [-1447304.7785581609, 12.121206326957532]), ('41123*', [-1447271.7778507448, 12.128724189611201]), ('12212*', [-1447271.4057896698, 17.388660680798203]), ('3212*', [-1447271.4057896696, 16.388660680798203]), ('31112*', [-1447271.4057896696, 17.388660680798203]), ('32212*', [-1447260.999144417, 16.388269517634285]), ('13223*', [-1447258.5793496096, 12.128724189611201]), ('4223*', [-1447258.5793496093, 11.128724189611201]), ('21323*', [-1447255.1704646456, 12.114921082079972]), ('34223*', [-1447236.0092436452, 11.122298770757636]), ('3323*', [-1447207.7637087659, 11.11071946063285]), ('12323*', [-1447207.7637087656, 12.11071946063285]), ('52313*', [-1447189.679703966, 11.15434639117689]), ('24113*', [-1447154.3921726162, 12.121206326957532]), ('25223*', [-1447085.6228580999, 11.122298770757636]), ('2423*', [-1447057.377323221, 11.11071946063285]), ('11423*', [-1447057.3773232207, 12.11071946063285]), ('32243*', [-1446980.8385422179, 11.12041038100569]), ('22334*', [-1446967.14106394, 9.58111947271551]), ('54221*', [-1446960.2585143235, 23.274195048280887]), ('42413*', [-1446927.3273699833, 11.138016217027364]), ('53231*', [-1446871.0858323127, 23.28974994861517]), ('23243*', [-1446830.4521566727, 11.12041038100569]), ('45221*', [-1446811.3759926336, 23.274195048280887]), ('44231*', [-1446722.2033106228, 23.28974994861517]), ('32513*', [-1446631.1066552068, 11.138016217027364]), ('52241*', [-1446630.997408379, 23.279977383689562]), ('35231*', [-1446572.5707462933, 23.28974994861517]), ('22253*', [-1446550.3985172748, 11.12041038100569]), ('54131*', [-1446530.6446308934, 23.39125550125139]), ('21142*', [-1446526.8337123552, 16.389392814878665]), ('11242*', [-1446487.038399146, 16.38939281487866]), ('2242*', [-1446487.0383991455, 15.389392814878663]), ('43241*', [-1446482.1148866895, 23.279977383689562]), ('23412*', [-1446446.3667938581, 15.383617706208009]), ('53141*', [-1446418.7551716797, 23.382375835998435]), ('45131*', [-1446381.7621092035, 23.39125550125139]), ('34241*', [-1446332.48232236, 23.279977383689562]), ('43343*', [-1446287.5804345221, 10.135201611316385]), ('15131*', [-1446275.1883680911, 26.463586157256977]), ('52151*', [-1446273.6320268225, 23.382375835998435]), ('44141*', [-1446269.8726499898, 23.382375835998435]), ('23432*', [-1446238.215945695, 14.378676631280943]), ('41213*', [-1446207.8875754054, 12.188957956004163]), ('42251*', [-1446204.8617836854, 23.279977383689562]), ('14213*', [-1446186.0928514006, 12.196475818657829]), ('5213*', [-1446186.0928514004, 11.196475818657829]), ('51113*', [-1446186.0928514001, 12.19647581865783]), ('25241*', [-1446182.095936815, 23.279977383689562]), ('51341*', [-1446163.378934082, 23.37041173515902]), ('34343*', [-1446137.947870193, 10.135201611316385]), ('14141*', [-1446127.0516405865, 26.463586157256977]), ('5141*', [-1446127.0516405858, 25.463586157256977]), ('51421*', [-1446127.0516405858, 24.463586157256977]), ('43151*', [-1446124.749505133, 23.382375835998435]), ('35141*', [-1446120.2400856607, 23.382375835998435]), ('31313*', [-1446058.2545173196, 12.188957956004163]), ('33251*', [-1446055.229219356, 23.279977383689562]), ('1541*', [-1446054.2440989222, 25.454578742677814]), ('4313*', [-1446037.2103297107, 11.196475818657829]), ('13313*', [-1446037.2103297107, 12.196475818657829]), ('42143*', [-1446028.1430059143, 11.188921626083413]), ('15421*', [-1446020.4794392972, 24.459078742677814]), ('51251*', [-1446018.918323776, 23.37041173515902]), ('15341*', [-1446018.7165848706, 23.370432645136855]), ('41441*', [-1446014.495921111, 23.37041173515902]), ('25343*', [-1445987.5614846474, 10.135201611316385]), ('4151*', [-1445978.1686276156, 25.463586157256977]), ('41521*', [-1445978.1686276156, 24.463586157256977]), ('13151*', [-1445978.1686276153, 26.463586157256977]), ('34151*', [-1445975.1169408034, 23.382375835998435]), ('51431*', [-1445947.7408096117, 23.468581367677814]), ('42443*', [-1445944.2605452756, 10.141811829086237]), ('33312*', [-1445939.1289087282, 15.462372684492793]), ('53441*', [-1445931.1735026187, 20.202413924109898]), ('21413*', [-1445907.867635531, 12.188957956004163]), ('551*', [-1445906.1078602371, 24.454578742677814]), ('1451*', [-1445906.107860237, 25.454578742677814]), ('33353*', [-1445905.8071766743, 10.13105307111866]), ('24251*', [-1445904.842833811, 23.279977383689562]), ('3413*', [-1445887.5777653817, 11.196475818657829]), ('12413*', [-1445887.5777653817, 12.196475818657829]), ('31132*', [-1445882.7416758372, 16.483121320673867]), ('33143*', [-1445878.510441585, 11.188921626083413]), ('5521*', [-1445872.3432006116, 23.459078742677814]), ('14521*', [-1445872.3432006116, 24.459078742677814]), ('5441*', [-1445870.5803461848, 22.370432645136855]), ('14441*', [-1445870.5803461845, 23.370432645136855]), ('41351*', [-1445870.0353108048, 23.37041173515902]), ('31541*', [-1445864.8628630259, 23.37041173515902]), ('3232*', [-1445856.2781819564, 15.483121320673867]), ('12232*', [-1445856.2781819564, 16.483121320673867]), ('54421*', [-1445838.6201347704, 21.374570364456936]), ('55121*', [-1445836.618012495, 23.4637130878875]), ('25151*', [-1445824.7305552585, 23.382375835998435]), ('15431*', [-1445805.79242524, 23.468598326365132]), ('41531*', [-1445798.857796641, 23.468581367677814]), ('52223*', [-1445792.971902956, 11.212585352912901]), ('24312*', [-1445788.742523183, 15.462372684492793]), ('44441*', [-1445782.2909809286, 20.202413924109898]), ('24353*', [-1445755.4207911296, 10.13105307111866]), ('23342*', [-1445753.7933361204, 14.401054514937526]), ('11513*', [-1445737.1913798363, 12.196475818657829]), ('2513*', [-1445737.191379836, 11.196475818657829]), ('32153*', [-1445731.9214063755, 11.188921626083413]), ('24143*', [-1445728.12405604, 11.188921626083413]), ('14351*', [-1445723.5121659986, 23.370407761022523]), ('5351*', [-1445723.5121659983, 22.370407761022523]), ('4541*', [-1445721.6978244954, 22.370432645136855]), ('13541*', [-1445721.6978244951, 23.370432645136855]), ('31451*', [-1445720.40225272, 23.37041173515902]), ('53521*', [-1445691.684841917, 21.37452276593647]), ('45421*', [-1445689.7376130808, 21.374570364456936]), ('41243*', [-1445685.8416495568, 11.188556312721769]), ('14531*', [-1445657.6561865546, 23.468598326365132]), ('5531*', [-1445657.6561865543, 22.468598326365132]), ('32543*', [-1445648.0398304993, 10.141811829086237]), ('43541*', [-1445634.6474432286, 20.202413357554597]), ('35441*', [-1445632.6584165997, 20.202413924109898]), ('23153*', [-1445581.5350208303, 11.188921626083413]), ('13451*', [-1445574.6296443092, 23.370407761022523]), ('4451*', [-1445574.629644309, 22.370407761022523]), ('32453*', [-1445574.0416389862, 10.13767650341595]), ('21551*', [-1445570.0153709303, 23.37041173515902]), ('22442*', [-1445550.861394299, 14.39130561387371]), ('44521*', [-1445542.8023202273, 21.37452276593647]), ('31343*', [-1445536.208591471, 11.188556312721769]), ('54431*', [-1445515.323061096, 20.384046321063327]), ('43451*', [-1445501.963058678, 20.19750673647386]), ('34541*', [-1445485.0148788996, 20.202413357554597]), ('3551*', [-1445424.9970799794, 22.370407761022523]), ('12551*', [-1445424.9970799794, 23.370407761022523]), ('35521*', [-1445393.1697558977, 21.37452276593647]), ('31253*', [-1445390.288783081, 11.188556312721769]), ('4343*', [-1445389.8600634478, 10.188573289663607]), ('13343*', [-1445389.8600634478, 11.188573289663607]), ('21443*', [-1445385.8217096825, 11.188556312721769]), ('53531*', [-1445368.4903696638, 20.38402761233011]), ('45431*', [-1445366.4405394064, 20.384046321063327]), ('34451*', [-1445352.3304943487, 20.19750673647386]), ('25541*', [-1445334.628493354, 20.202413357554597]), ('22553*', [-1445276.3286189726, 10.13767650341595]), ('3443*', [-1445240.2274991188, 10.188573289663607]), ('12443*', [-1445240.2274991188, 11.188573289663607]), ('21353*', [-1445239.9019012928, 11.188556312721769]), ('44531*', [-1445219.6078479742, 20.38402761233011]), ('33551*', [-1445203.9426430499, 20.197506111560305]), ('25451*', [-1445201.944108803, 20.19750673647386]), ('3353*', [-1445091.9565405787, 10.188552523383924]), ('12353*', [-1445091.9565405787, 11.188552523383924]), ('2543*', [-1445089.8411135734, 10.188573289663607]), ('11543*', [-1445089.8411135734, 11.188573289663607]), ('35531*', [-1445069.975283645, 20.38402761233011]), ('24551*', [-1445053.5562575047, 20.197506111560305]), ('55321*', [-1445029.8468214332, 21.550049665483588]), ('11453*', [-1444941.5701550336, 11.188552523383924]), ('2453*', [-1444941.5701550334, 10.188552523383924]), ('52343*', [-1444637.3122015677, 10.230013965827665]), ('5123*', [-1444467.6575939169, 11.272072108405588]), ('14123*', [-1444467.6575939169, 12.272072108405588]), ('33332*', [-1444466.8662604126, 14.552503040524357]), ('54341*', [-1444435.6600214103, 20.47947625483115]), ('53213*', [-1444354.838378113, 11.270618667271545]), ('32412*', [-1444351.5722546387, 15.556732536613483]), ('53123*', [-1444341.8199659411, 11.271080691106144]), ('413*', [-1444318.7745809462, 11.272072108405588]), ('1313*', [-1444318.7745809462, 12.272072108405588]), ('24332*', [-1444316.4798748675, 14.552503040524357]), ('45341*', [-1444286.7774997205, 20.47947625483115]), ('42353*', [-1444265.280929773, 10.226264269241685]), ('44213*', [-1444205.9558564234, 11.270618667271545]), ('44123*', [-1444192.9374442513, 11.271080691106144]), ('143*', [-1444174.7615748546, 11.272072943363703]), ('53*', [-1444174.7615748544, 10.272072943363703]), ('12143*', [-1444169.1415228606, 12.272072108405588]), ('3143*', [-1444169.1415228604, 11.272072108405588]), ('53351*', [-1444157.5650999483, 20.475000740963115]), ('32432*', [-1444155.6491523138, 14.55269527021214]), ('15251*', [-1444134.8921874734, 23.649247056417572]), ('55331*', [-1444080.0719131539, 20.64488538064411]), ('35213*', [-1444056.323292094, 11.270618667271545]), ('22512*', [-1444053.8592346248, 15.556732536613483]), ('52541*', [-1444048.6925504608, 20.487578481878145]), ('35123*', [-1444043.3048799222, 11.271080691106144]), ('443*', [-1444025.8790531647, 10.272072943363703]), ('1343*', [-1444025.8790531647, 11.272072943363703]), ('11153*', [-1444018.7546410724, 12.272072108405588]), ('2153*', [-1444018.754641072, 11.272072108405588]), ('44351*', [-1444008.6825782591, 20.475000740963115]), ('52451*', [-1443927.0601898702, 20.483115023851386]), ('42312*', [-1443913.5626436123, 15.628380394152776]), ('353*', [-1443876.2464888357, 10.272072943363703]), ('1253*', [-1443876.2464888357, 11.272072943363703]), ('35351*', [-1443859.0500139298, 20.475000740963115]), ('22532*', [-1443857.9361322997, 14.55269527021214]), ('32342*', [-1443804.0478037433, 14.563764675020478]), ('51143*', [-1443726.2375892266, 11.286110505296184]), ('5243*', [-1443687.0377108944, 10.286110505296184]), ('14243*', [-1443687.0377108941, 11.286110505296184]), ('42551*', [-1443632.3243000566, 20.483115023851386]), ('41153*', [-1443444.7002005482, 11.286110505296184]), ('22352*', [-1443406.824282713, 14.55956266518849]), ('4253*', [-1443392.3018210805, 10.286110505296184]), ('13253*', [-1443392.3018210803, 11.286110505296184]), ('22124*', [-1443188.660323922, 10.4392890044345]), ('22214*', [-1443177.2933123286, 10.44006876024345]), ('21134*', [-1443016.6791253812, 10.44006798566403]), ('2234*', [-1442990.0823133066, 9.440067985664028]), ('11234*', [-1442990.0823133066, 10.440067985664028]), ('43323*', [-1442938.3929843814, 10.976088810772813]), ('34323*', [-1442788.760420052, 10.976088810772813]), ('33423*', [-1442649.3859640283, 10.958433848456583]), ('25323*', [-1442638.3740345067, 10.976088810772813]), ('15113*', [-1442630.8952339075, 12.362072108405588]), ('1523*', [-1442604.8276983933, 11.36207210840559]), ('42332*', [-1442567.5839682764, 14.71085741999978]), ('24423*', [-1442498.999578483, 10.958433848456583]), ('23523*', [-1442349.6137834536, 10.958427243083227]), ('32112*', [-1442300.9835027843, 17.216647995518553]), ('212*', [-1442265.5665031679, 17.20809383395393]), ('1112*', [-1442265.5665031674, 18.20809383395393]), ('32132*', [-1442160.1268794138, 16.226210680798204]), ('23112*', [-1442150.5971172394, 17.216647995518553]), ('22232*', [-1442106.0219980066, 16.208946471387527]), ('1132*', [-1442037.609696898, 17.20359117794156]), ('232*', [-1442037.6096968977, 16.20359117794156]), ('23132*', [-1442009.740493869, 16.226210680798204]), ('22142*', [-1441954.2844934994, 16.21688961392354]), ('53313*', [-1441778.5944118681, 11.051997981496445]), ('21242*', [-1441719.005899105, 16.209071748004003]), ('44313*', [-1441629.7118901785, 11.051997981496445]), ('23334*', [-1441614.654253814, 9.504300614497462]), ('43413*', [-1441517.4863263955, 11.034037422094219]), ('35313*', [-1441480.0793258492, 11.051997981496445]), ('2342*', [-1441420.148637307, 15.209072657030335]), ('11342*', [-1441420.148637307, 16.209072657030333]), ('34413*', [-1441367.8537620662, 11.034037422094219]), ('33513*', [-1441219.19838727, 11.03403360147314]), ('25413*', [-1441217.4673765213, 11.034037422094219]), ('22434*', [-1441156.9170860173, 9.51302885640185]), ('3112*', [-1441090.9736511593, 17.29809383395393]), ('12112*', [-1441090.973651159, 18.29809383395393]), ('52323*', [-1441089.8418714227, 11.078983070384934]), ('32*', [-1441077.7086584133, 16.29809383395393]), ('12*', [-1441077.708658413, 17.29809383395393]), ('31212*', [-1441077.177976184, 17.30721484236283]), ('24513*', [-1441068.8120017252, 11.03403360147314]), ('31232*', [-1441000.3569762958, 16.312121320673867]), ('33212*', [-1440962.1632776582, 16.29400916987495]), ('21312*', [-1440926.791094395, 17.30721484236283]), ('21332*', [-1440849.9700945066, 16.312121320673867]), ('3312*', [-1440848.507845921, 16.302645101108837]), ('12312*', [-1440848.5078459207, 17.302645101108837]), ('24212*', [-1440811.7768921128, 16.29400916987495]), ('42423*', [-1440803.579573387, 11.062921062012627]), ('3332*', [-1440704.5201386728, 15.312092969513923]), ('12332*', [-1440704.5201386726, 16.312092969513923]), ('11412*', [-1440698.1214603758, 17.302645101108837]), ('2412*', [-1440698.1214603756, 16.302645101108837]), ('11432*', [-1440554.1337531277, 16.312092969513923]), ('2432*', [-1440554.1337531274, 15.312092969513923]), ('32523*', [-1440507.3588586103, 11.062921062012627]), ('22134*', [-1440501.6629864897, 10.277015620677687]), ('43443*', [-1440492.898172868, 10.038252487087178]), ('23442*', [-1440443.0349906129, 14.21152013751759]), ('34443*', [-1440343.2656085389, 10.038252487087178]), ('52113*', [-1440198.9548514385, 12.121206326957532]), ('33543*', [-1440194.8972028766, 10.038251498161248]), ('25443*', [-1440192.8792229937, 10.038252487087178]), ('41223*', [-1440150.345958007, 12.114921082079972]), ('53223*', [-1440130.5302474368, 11.122298770757636]), ('33453*', [-1440114.976578314, 10.033706050024048]), ('43113*', [-1440050.0723297487, 12.121206326957532]), ('24543*', [-1440044.5108173315, 10.038251498161248]), ('51123*', [-1440017.2370404543, 12.128724189611201]), ('4212*', [-1440016.8668443586, 16.388660680798203]), ('41112*', [-1440016.8668443586, 17.388660680798203]), ('13212*', [-1440016.8668443584, 17.388660680798203]), ('42212*', [-1440006.5123630695, 16.388269517634285]), ('5223*', [-1440004.1046976356, 11.128724189611201]), ('14223*', [-1440004.1046976356, 12.128724189611201]), ('31323*', [-1440000.7128999222, 12.114921082079972]), ('44223*', [-1439981.6477257474, 11.122298770757636]), ('24453*', [-1439964.5901927687, 10.033706050024048]), ('13323*', [-1439953.5437733878, 12.11071946063285]), ('4323*', [-1439953.5437733876, 11.11071946063285]), ('34113*', [-1439900.4397654196, 12.121206326957532]), ('21423*', [-1439850.3260181334, 12.114921082079972]), ('35223*', [-1439832.015161418, 11.122298770757636]), ('23553*', [-1439815.4743593913, 10.033705007005459]), ('3423*', [-1439803.9112090587, 11.11071946063285]), ('12423*', [-1439803.9112090587, 12.11071946063285]), ('25113*', [-1439750.0533798742, 12.121206326957532]), ('42243*', [-1439727.7560835092, 11.12041038100569]), ('32334*', [-1439714.1272647032, 9.58111947271551]), ('52413*', [-1439674.5131393913, 11.138016217027364]), ('2523*', [-1439653.524823513, 11.11071946063285]), ('11523*', [-1439653.524823513, 12.11071946063285]), ('33243*', [-1439578.12351918, 11.12041038100569]), ('55221*', [-1439559.142975613, 23.274195048280887]), ('54231*', [-1439470.4172772786, 23.28974994861517]), ('24243*', [-1439427.737133635, 11.12041038100569]), ('42513*', [-1439379.7772495775, 11.138016217027364]), ('45231*', [-1439321.5347555887, 23.28974994861517]), ('32253*', [-1439299.473666263, 11.12041038100569]), ('31142*', [-1439276.0269814099, 16.389392814878665]), ('3242*', [-1439236.4311447109, 15.389392814878663]), ('12242*', [-1439236.4311447106, 16.38939281487866]), ('53241*', [-1439231.5323116707, 23.279977383689562]), ('33412*', [-1439195.9634084033, 15.383617706208009]), ('23253*', [-1439149.087280718, 11.12041038100569]), ('55131*', [-1439131.6825587929, 23.39125550125139]), ('44241*', [-1439082.649789981, 23.279977383689562]), ('24412*', [-1439045.5770228587, 15.383617706208009]), ('53343*', [-1439037.9729756801, 10.135201611316385]), ('54141*', [-1439020.3539525312, 23.382375835998435]), ('21152*', [-1438991.6457807252, 16.389392814878665]), ('33432*', [-1438988.8559294576, 14.378676631280943]), ('51213*', [-1438958.6795820314, 12.188957956004163]), ('52251*', [-1438955.6689572826, 23.279977383689562]), ('2252*', [-1438938.718124697, 15.389392814878663]), ('11252*', [-1438938.718124697, 16.38939281487866]), ('15213*', [-1438936.9941054513, 12.196475818657829]), ('35241*', [-1438933.017225652, 23.279977383689562]), ('23512*', [-1438896.3777652683, 15.38359480608649]), ('44343*', [-1438889.0904539905, 10.135201611316385]), ('15141*', [-1438878.2488424191, 26.463586157256977]), ('53151*', [-1438875.9582465647, 23.382375835998435]), ('45141*', [-1438871.4714308416, 23.382375835998435]), ('24432*', [-1438838.4695439124, 14.378676631280943]), ('41313*', [-1438809.79656906, 12.188957956004163]), ('43251*', [-1438806.786435593, 23.279977383689562]), ('5313*', [-1438788.8578667652, 11.196475818657829]), ('14313*', [-1438788.8578667652, 12.196475818657829]), ('52143*', [-1438779.8359934993, 11.188921626083413]), ('51441*', [-1438766.2573155672, 23.37041173515902]), ('35343*', [-1438739.457889661, 10.135201611316385]), ('5151*', [-1438730.1121149152, 25.463586157256977]), ('51521*', [-1438730.1121149152, 24.463586157256977]), ('14151*', [-1438730.1121149152, 26.463586157256977]), ('44151*', [-1438727.0757248749, 23.382375835998435]), ('52443*', [-1438696.3739989707, 10.141811829086237]), ('22314*', [-1438693.4760927428, 10.35521086595188]), ('43312*', [-1438691.2680850741, 15.462372684492793]), ('23532*', [-1438689.5318531203, 14.378659098082164]), ('31413*', [-1438660.1635109745, 12.188957956004163]), ('1551*', [-1438658.4125566653, 25.454578742677814]), ('43353*', [-1438658.113380298, 10.13105307111866]), ('34251*', [-1438657.1538712636, 23.279977383689562]), ('4413*', [-1438639.9753450758, 11.196475818657829]), ('13413*', [-1438639.9753450758, 12.196475818657829]), ('41132*', [-1438635.1634967346, 16.483121320673867]), ('43143*', [-1438630.9534718096, 11.188921626083413]), ('15521*', [-1438624.8171445199, 24.459078742677814]), ('15441*', [-1438623.0631265116, 23.370432645136855]), ('51351*', [-1438622.5208231558, 23.37041173515902]), ('41541*', [-1438617.3743025963, 23.37041173515902]), ('13232*', [-1438608.832652781, 16.483121320673867]), ('4232*', [-1438608.8326527807, 15.483121320673867]), ('35151*', [-1438577.443160546, 23.382375835998435]), ('51531*', [-1438551.7000907585, 23.468581367677814]), ('34312*', [-1438541.6355207448, 15.462372684492793]), ('54441*', [-1438535.2163172518, 20.202413924109898]), ('21513*', [-1438509.7766291858, 12.188957956004163]), ('34353*', [-1438508.4808159692, 10.13105307111866]), ('33342*', [-1438506.8615186806, 14.401054514937526]), ('25251*', [-1438506.7674857185, 23.279977383689562]), ('3513*', [-1438490.3427807463, 11.196475818657829]), ('12513*', [-1438490.342780746, 12.196475818657829]), ('42153*', [-1438485.0992233586, 11.188921626083413]), ('34143*', [-1438481.3209074805, 11.188921626083413]), ('15351*', [-1438476.732134828, 23.370407761022523]), ('5541*', [-1438474.9268878258, 22.370432645136855]), ('14541*', [-1438474.9268878258, 23.370432645136855]), ('41451*', [-1438473.6378101853, 23.37041173515902]), ('55421*', [-1438443.1268789812, 21.374570364456936]), ('51243*', [-1438439.2504442192, 11.188556312721769]), ('15531*', [-1438411.206262623, 23.468598326365132]), ('42543*', [-1438401.6381091573, 10.141811829086237]), ('25312*', [-1438391.2491351992, 15.462372684492793]), ('53541*', [-1438388.31285207, 20.202413357554597]), ('45441*', [-1438386.3337955624, 20.202413924109898]), ('22224*', [-1438379.485473899, 10.354752045773337]), ('25353*', [-1438358.0944304238, 10.13105307111866]), ('24342*', [-1438356.4751331354, 14.401054514937526]), ('33153*', [-1438335.4666590297, 11.188921626083413]), ('25143*', [-1438330.9345219354, 11.188921626083413]), ('5451*', [-1438328.5958961425, 22.370407761022523]), ('14451*', [-1438328.5958961423, 23.370407761022523]), ('42453*', [-1438328.0108382334, 10.13767650341595]), ('31551*', [-1438324.004752099, 23.37041173515902]), ('32442*', [-1438304.9467859806, 14.39130561387371]), ('54521*', [-1438296.928108524, 21.37452276593647]), ('41343*', [-1438290.367431248, 11.188556312721769]), ('53451*', [-1438256.2935563421, 20.19750673647386]), ('44541*', [-1438239.4303303806, 20.202413357554597]), ('24153*', [-1438185.0802734846, 11.188921626083413]), ('4551*', [-1438179.7133744524, 22.370407761022523]), ('13551*', [-1438179.7133744524, 23.370407761022523]), ('45521*', [-1438148.0455868342, 21.37452276593647]), ('41253*', [-1438145.1790550754, 11.188556312721769]), ('5343*', [-1438144.7524844261, 10.188573289663607]), ('14343*', [-1438144.7524844261, 11.188573289663607]), ('31443*', [-1438140.734373163, 11.188556312721769]), ('55431*', [-1438121.4503522215, 20.384046321063327]), ('44451*', [-1438107.4110346523, 20.19750673647386]), ('23352*', [-1438099.7954024589, 14.396409570991995]), ('35541*', [-1438089.7977660513, 20.202413357554597]), ('21234*', [-1438073.5711787988, 10.35813546876504]), ('32553*', [-1438031.7901234569, 10.13767650341595]), ('22542*', [-1438007.2337659667, 14.39130561387371]), ('4443*', [-1437995.8699627367, 10.188573289663607]), ('13443*', [-1437995.8699627365, 11.188573289663607]), ('31353*', [-1437995.5459969898, 11.188556312721769]), ('21543*', [-1437990.347491374, 11.188556312721769]), ('54531*', [-1437975.35366889, 20.38402761233011]), ('43551*', [-1437959.7669867906, 20.197506111560305]), ('35451*', [-1437957.778470323, 20.19750673647386]), ('23214*', [-1437933.9256984023, 10.358090269315449]), ('23124*', [-1437929.4169743059, 10.358055039205992]), ('22452*', [-1437908.3736740144, 14.386638892342694]), ('4353*', [-1437848.3422217015, 10.188552523383924]), ('13353*', [-1437848.3422217015, 11.188552523383924]), ('3543*', [-1437846.2373984072, 10.188573289663607]), ('12543*', [-1437846.237398407, 11.188573289663607]), ('21453*', [-1437845.159115201, 11.188556312721769]), ('45531*', [-1437826.4711472003, 20.38402761233011]), ('34551*', [-1437810.134422462, 20.197506111560305]), ('2334*', [-1437776.5013845253, 9.358111758033958]), ('11334*', [-1437776.5013845253, 10.358111758033958]), ('3453*', [-1437698.709657372, 10.188552523383924]), ('12453*', [-1437698.709657372, 11.188552523383924]), ('25551*', [-1437659.7480369161, 20.197506111560305]), ('11114*', [-1437570.9978407985, 11.358944961148303]), ('2114*', [-1437570.9978407982, 10.358944961148303]), ('11553*', [-1437548.3232718273, 11.188552523383924]), ('2553*', [-1437548.323271827, 10.188552523383924]), ('1124*', [-1437531.2025275892, 10.358944961148303]), ('224*', [-1437531.202527589, 9.358944961148303]), ('15123*', [-1437227.172612774, 12.272072108405588]), ('43332*', [-1437226.385245879, 14.552503040524357]), ('42412*', [-1437111.6691585598, 15.556732536613483]), ('513*', [-1437079.0358852698, 11.272072108405588]), ('1413*', [-1437079.0358852695, 12.272072108405588]), ('34332*', [-1437076.7526815496, 14.552503040524357]), ('55341*', [-1437047.1991914262, 20.47947625483115]), ('52353*', [-1437025.8103743873, 10.226264269241685]), ('54213*', [-1436966.7826716988, 11.270618667271545]), ('54123*', [-1436953.8295151365, 11.271080691106144]), ('153*', [-1436935.7447534283, 11.272072943363703]), ('4143*', [-1436930.1528722984, 11.272072108405588]), ('13143*', [-1436930.1528722984, 12.272072108405588]), ('25332*', [-1436926.3662960045, 14.552503040524357]), ('42432*', [-1436916.7281331078, 14.55269527021214]), ('45213*', [-1436817.9001500092, 11.270618667271545]), ('32512*', [-1436815.4484437834, 15.556732536613483]), ('45123*', [-1436804.9469934471, 11.271080691106144]), ('543*', [-1436787.6085147425, 10.272072943363703]), ('1443*', [-1436787.6085147425, 11.272072943363703]), ('3153*', [-1436780.5198142135, 11.272072108405588]), ('12153*', [-1436780.5198142133, 12.272072108405588]), ('54351*', [-1436770.4982382488, 20.475000740963115]), ('52312*', [-1436675.8550982568, 15.628380394152776]), ('453*', [-1436638.7259930528, 10.272072943363703]), ('1353*', [-1436638.7259930528, 11.272072943363703]), ('45351*', [-1436621.615716559, 20.475000740963115]), ('32532*', [-1436620.5074183308, 14.55269527021214]), ('42342*', [-1436566.8892084104, 14.563764675020478]), ('15243*', [-1436450.465636011, 11.286110505296184]), ('52551*', [-1436396.026479586, 20.483115023851386]), ('51153*', [-1436209.3428576768, 11.286110505296184]), ('32352*', [-1436171.6567952621, 14.55956266518849]), ('5253*', [-1436157.2071283818, 10.286110505296184]), ('14253*', [-1436157.2071283818, 11.286110505296184]), ('23434*', [-1435992.2246519898, 9.429453136259543]), ('32124*', [-1435954.5863970355, 10.4392890044345]), ('32214*', [-1435943.2763633025, 10.44006876024345]), ('31134*', [-1435783.4672650688, 10.44006798566403]), ('3234*', [-1435757.003771188, 9.440067985664028]), ('12234*', [-1435757.0037711875, 10.440067985664028]), ('53323*', [-1435705.5735382747, 10.976088810772813]), ('44323*', [-1435556.6910165849, 10.976088810772813]), ('43423*', [-1435418.0151837876, 10.958433848456583]), ('22354*', [-1435408.1157974869, 9.43635232063727]), ('35323*', [-1435407.0584522556, 10.976088810772813]), ('52332*', [-1435336.623225684, 14.71085741999978]), ('34423*', [-1435268.3826194583, 10.958433848456583]), ('33523*', [-1435119.745630122, 10.958427243083227]), ('25423*', [-1435117.996233913, 10.958433848456583]), ('42112*', [-1435071.3591117924, 17.216647995518553]), ('312*', [-1435036.1196421143, 17.20809383395393]), ('1212*', [-1435036.1196421143, 18.20809383395393]), ('24523*', [-1434969.359244577, 10.958427243083227]), ('42132*', [-1434931.2085411057, 16.226210680798204]), ('33112*', [-1434921.7265474633, 17.216647995518553]), ('11132*', [-1434885.7327603258, 18.20809383395393]), ('2132*', [-1434885.7327603253, 17.20809383395393]), ('32232*', [-1434877.3748638192, 16.208946471387527]), ('332*', [-1434809.3054836725, 16.20359117794156]), ('1232*', [-1434809.3054836725, 17.20359117794156]), ('33132*', [-1434781.575976777, 16.226210680798204]), ('24112*', [-1434771.3401619177, 17.216647995518553]), ('23232*', [-1434726.988478274, 16.208946471387527]), ('32142*', [-1434726.3979530972, 16.21688961392354]), ('22242*', [-1434667.5899012315, 16.19868811945491]), ('1142*', [-1434658.9190981272, 17.20359117794156]), ('242*', [-1434658.919098127, 16.20359117794156]), ('24132*', [-1434631.1895912315, 16.226210680798204]), ('23142*', [-1434576.011567552, 16.21688961392354]), ('31242*', [-1434492.2987074547, 16.209071748004003]), ('22152*', [-1434428.6840438643, 16.21688961392354]), ('54313*', [-1434403.4522903636, 11.051997981496445]), ('33334*', [-1434388.4701313484, 9.504300614497462]), ('21342*', [-1434341.9118256655, 16.209071748004003]), ('53413*', [-1434291.7892642778, 11.034037422094219]), ('45313*', [-1434254.569768674, 11.051997981496445]), ('24334*', [-1434238.0837458035, 9.504300614497462]), ('21252*', [-1434195.256900238, 16.209071748004003]), ('12342*', [-1434194.9394864775, 16.209072657030333]), ('3342*', [-1434194.9394864773, 15.209072657030335]), ('44413*', [-1434142.906742588, 11.034037422094219]), ('11442*', [-1434044.5531009326, 16.209072657030333]), ('2442*', [-1434044.5531009324, 15.209072657030335]), ('43513*', [-1433994.9965122072, 11.03403360147314]), ('35413*', [-1433993.2741782588, 11.034037422094219]), ('32434*', [-1433933.0273998943, 9.51302885640185]), ('2352*', [-1433895.3652911324, 15.209069597115706]), ('11352*', [-1433895.3652911324, 16.209069597115707]), ('4112*', [-1433867.4145106508, 17.29809383395393]), ('13112*', [-1433867.4145106506, 18.29809383395393]), ('132*', [-1433854.2160095149, 17.29809383395393]), ('42*', [-1433854.2160095146, 16.29809383395393]), ('41212*', [-1433853.687987364, 17.30721484236283]), ('34513*', [-1433845.3639478781, 11.03403360147314]), ('41232*', [-1433777.252057569, 16.312121320673867]), ('23314*', [-1433756.175265828, 10.264837240985697]), ('43212*', [-1433739.2498072477, 16.29400916987495]), ('31312*', [-1433704.054929278, 17.30721484236283]), ('25513*', [-1433694.977562333, 11.03403360147314]), ('22534*', [-1433635.3143798807, 9.51302885640185]), ('31332*', [-1433627.618999483, 16.312121320673867]), ('13312*', [-1433626.1640805106, 17.302645101108837]), ('4312*', [-1433626.1640805104, 16.302645101108837]), ('34212*', [-1433589.6172429183, 16.29400916987495]), ('52423*', [-1433581.4610137683, 11.062921062012627]), ('21412*', [-1433553.6680474894, 17.30721484236283]), ('13332*', [-1433482.8981207008, 16.312092969513923]), ('4332*', [-1433482.8981207006, 15.312092969513923]), ('21432*', [-1433477.2321176943, 16.312121320673867]), ('3412*', [-1433476.531516181, 16.302645101108837]), ('12412*', [-1433476.531516181, 17.302645101108837]), ('25212*', [-1433439.2308573727, 16.29400916987495]), ('23224*', [-1433399.8183473728, 10.265140578703047]), ('22234*', [-1433386.071367483, 10.265436292041546]), ('3432*', [-1433333.2655563713, 15.312092969513923]), ('12432*', [-1433333.2655563713, 16.312092969513923]), ('11512*', [-1433326.1451306362, 17.302645101108837]), ('2512*', [-1433326.145130636, 16.302645101108837]), ('42523*', [-1433286.7251239552, 11.062921062012627]), ('32134*', [-1433281.0578027514, 10.277015620677687]), ('53443*', [-1433272.3369233096, 10.038252487087178]), ('33442*', [-1433222.7236833917, 14.21152013751759]), ('23134*', [-1433197.0434561097, 10.264831888268116]), ('21114*', [-1433188.3575476068, 11.268948137576675]), ('2214*', [-1433188.3575476066, 10.268948137576677]), ('11214*', [-1433188.3575476066, 11.268948137576677]), ('11532*', [-1433182.8791708266, 16.312092969513923]), ('2532*', [-1433182.8791708264, 15.312092969513923]), ('44443*', [-1433123.4544016197, 10.038252487087178]), ('24442*', [-1433072.3372978456, 14.21152013751759]), ('43543*', [-1432975.8297019221, 10.038251498161248]), ('35443*', [-1432973.821837291, 10.038252487087178]), ('51223*', [-1432931.5017729697, 12.114921082079972]), ('23542*', [-1432923.2030390056, 14.21151981593748]), ('43453*', [-1432896.3096845169, 10.033706050024048]), ('11124*', [-1432834.6349931553, 11.268944961148303]), ('2124*', [-1432834.634993155, 10.268944961148303]), ('53113*', [-1432831.730772579, 12.121206326957532]), ('34543*', [-1432826.197137593, 10.038251498161248]), ('23452*', [-1432816.1095138802, 14.206389452191276]), ('14212*', [-1432798.6917317733, 17.388660680798203]), ('5212*', [-1432798.691731773, 16.388660680798203]), ('51112*', [-1432798.691731773, 17.388660680798203]), ('52212*', [-1432788.3891529727, 16.388269517634285]), ('15223*', [-1432785.9935561132, 12.128724189611201]), ('41323*', [-1432782.618759999, 12.114921082079972]), ('54223*', [-1432763.649151209, 11.122298770757636]), ('34453*', [-1432746.6771201876, 10.033706050024048]), ('5323*', [-1432735.6860716783, 11.11071946063285]), ('14323*', [-1432735.686071678, 12.11071946063285]), ('44113*', [-1432682.8482508897, 12.121206326957532]), ('25543*', [-1432675.8107520477, 10.038251498161248]), ('31423*', [-1432632.985701914, 12.114921082079972]), ('45223*', [-1432614.766629519, 11.122298770757636]), ('33553*', [-1432598.3087393031, 10.033705007005459]), ('25453*', [-1432596.2907346424, 10.033706050024048]), ('4423*', [-1432586.8035499887, 11.11071946063285]), ('13423*', [-1432586.8035499887, 12.11071946063285]), ('1134*', [-1432535.7535183073, 10.268943242061733]), ('234*', [-1432535.7535183069, 9.268943242061733]), ('35113*', [-1432533.2156865601, 12.121206326957532]), ('52243*', [-1432511.0301567956, 11.12041038100569]), ('42334*', [-1432497.4696533012, 9.58111947271551]), ('21523*', [-1432482.5988201245, 12.114921082079972]), ('24553*', [-1432447.922353758, 10.033705007005459]), ('3523*', [-1432437.1709856591, 11.11071946063285]), ('12523*', [-1432437.170985659, 12.11071946063285]), ('43243*', [-1432362.1476351062, 11.12041038100569]), ('34243*', [-1432212.515070777, 11.12041038100569]), ('52513*', [-1432164.7955886542, 11.138016217027364]), ('55231*', [-1432106.8450388305, 23.28974994861517]), ('42253*', [-1432084.8945321022, 11.12041038100569]), ('25243*', [-1432062.1286852318, 11.12041038100569]), ('41142*', [-1432061.5653752317, 16.389392814878665]), ('4242*', [-1432022.168015154, 15.389392814878663]), ('13242*', [-1432022.168015154, 16.38939281487866]), ('43412*', [-1431981.9031259196, 15.383617706208009]), ('33253*', [-1431935.2619677729, 11.12041038100569]), ('54241*', [-1431869.1574991364, 23.279977383689562]), ('34412*', [-1431832.270561591, 15.383617706208009]), ('24253*', [-1431784.8755822277, 11.12041038100569]), ('31152*', [-1431778.6096532007, 16.389392814878665]), ('43432*', [-1431775.8337862385, 14.378676631280943]), ('12252*', [-1431725.9473003773, 16.38939281487866]), ('3252*', [-1431725.947300377, 15.389392814878663]), ('45241*', [-1431720.2749774472, 23.279977383689562]), ('33512*', [-1431683.8191746632, 15.38359480608649]), ('25412*', [-1431681.8841760457, 15.383617706208009]), ('54343*', [-1431676.5683914914, 10.135201611316385]), ('55141*', [-1431659.037684804, 23.382375835998435]), ('34432*', [-1431626.201221909, 14.378676631280943]), ('51313*', [-1431597.6719721463, 12.188957956004163]), ('53251*', [-1431594.6769271626, 23.279977383689562]), ('15313*', [-1431576.8382264136, 12.196475818657829]), ('24512*', [-1431533.4327891185, 15.38359480608649]), ('45343*', [-1431527.6858698016, 10.135201611316385]), ('15151*', [-1431518.3869413394, 26.463586157256977]), ('54151*', [-1431515.3657713952, 23.382375835998435]), ('32314*', [-1431481.9345595324, 10.35521086595188]), ('53312*', [-1431479.737619641, 15.462372684492793]), ('22324*', [-1431478.403046314, 10.339836394201303]), ('33532*', [-1431478.0100906591, 14.378659098082164]), ('25432*', [-1431475.8148363638, 14.378676631280943]), ('41413*', [-1431448.7889591753, 12.188957956004163]), ('53353*', [-1431446.749104908, 10.13105307111866]), ('44251*', [-1431445.7944054727, 23.279977383689562]), ('14413*', [-1431428.701987728, 12.196475818657829]), ('5413*', [-1431428.7019877278, 11.196475818657829]), ('51132*', [-1431423.9142590791, 16.483121320673867]), ('53143*', [-1431419.725337169, 11.188921626083413]), ('51541*', [-1431406.2142343952, 23.37041173515902]), ('5232*', [-1431397.7154001368, 15.483121320673867]), ('14232*', [-1431397.7154001365, 16.483121320673867]), ('45151*', [-1431366.483249706, 23.382375835998435]), ('44312*', [-1431330.8550979514, 15.462372684492793]), ('24532*', [-1431327.6237051138, 14.378659098082164]), ('31513*', [-1431299.15590109, 12.188957956004163]), ('44353*', [-1431297.8665832188, 10.13105307111866]), ('43342*', [-1431296.2554027596, 14.401054514937526]), ('35251*', [-1431296.1618411436, 23.279977383689562]), ('4513*', [-1431279.819466038, 11.196475818657829]), ('13513*', [-1431279.819466038, 12.196475818657829]), ('52153*', [-1431274.6021923115, 11.188921626083413]), ('44143*', [-1431270.8428154795, 11.188921626083413]), ('15541*', [-1431264.4808462504, 23.370432645136855]), ('51451*', [-1431263.1982301925, 23.37041173515902]), ('52543*', [-1431191.5594321943, 10.141811829086237]), ('22414*', [-1431184.221539519, 10.35521086595188]), ('35312*', [-1431181.2225336216, 15.462372684492793]), ('55441*', [-1431176.3318324338, 20.202413924109898]), ('32224*', [-1431169.5178384134, 10.354752045773337]), ('35353*', [-1431148.2340188893, 10.13105307111866]), ('34342*', [-1431146.6228384306, 14.401054514937526]), ('43153*', [-1431125.719670622, 11.188921626083413]), ('35143*', [-1431121.2102511502, 11.188921626083413]), ('15451*', [-1431118.8833478661, 23.370407761022523]), ('52453*', [-1431118.3012225966, 10.13767650341595]), ('41551*', [-1431114.3152172212, 23.37041173515902]), ('42442*', [-1431095.3527803558, 14.39130561387371]), ('51343*', [-1431080.8465055563, 11.188556312721769]), ('54541*', [-1431030.1647301102, 20.202413357554597]), ('25342*', [-1430996.2364528852, 14.401054514937526]), ('34153*', [-1430976.0871062926, 11.188921626083413]), ('5551*', [-1430970.7471091799, 22.370407761022523]), ('14551*', [-1430970.7471091796, 23.370407761022523]), ('55521*', [-1430939.2380583386, 21.37452276593647]), ('51253*', [-1430936.3858952501, 11.188556312721769]), ('15343*', [-1430935.9614628134, 11.188573289663607]), ('41443*', [-1430931.9634925858, 11.188556312721769]), ('54451*', [-1430898.8071894054, 20.19750673647386]), ('33352*', [-1430891.2297310468, 14.396409570991995]), ('45541*', [-1430881.2822084206, 20.202413357554597]), ('31234*', [-1430865.1369579574, 10.35813546876504]), ('25153*', [-1430825.7007207475, 11.188921626083413]), ('42553*', [-1430823.565332783, 10.13767650341595]), ('32542*', [-1430799.1320655795, 14.39130561387371]), ('5443*', [-1430787.8252241279, 10.188573289663607]), ('14443*', [-1430787.8252241279, 11.188573289663607]), ('41353*', [-1430787.50288228, 11.188556312721769]), ('31543*', [-1430782.3304344998, 11.188556312721769]), ('53551*', [-1430751.9032166195, 20.197506111560305]), ('45451*', [-1430749.9246677153, 20.19750673647386]), ('24352*', [-1430740.8433455017, 14.396409570991995]), ('33214*', [-1430726.1914593144, 10.358090269315449]), ('33124*', [-1430721.705335481, 10.358055039205992]), ('21334*', [-1430714.7500761687, 10.35813546876504]), ('32452*', [-1430700.7675160551, 14.386638892342694]), ('5353*', [-1430641.036975173, 10.188552523383924]), ('14353*', [-1430641.036975173, 11.188552523383924]), ('4543*', [-1430638.942702438, 10.188573289663607]), ('13543*', [-1430638.9427024378, 11.188573289663607]), ('31453*', [-1430637.869824194, 11.188556312721769]), ('55531*', [-1430619.2755308084, 20.38402761233011]), ('44551*', [-1430603.02069493, 20.197506111560305]), ('24214*', [-1430575.805073769, 10.358090269315449]), ('24124*', [-1430571.318949936, 10.358055039205992]), ('3334*', [-1430569.5562447114, 9.358111758033958]), ('12334*', [-1430569.5562447114, 10.358111758033958]), ('4453*', [-1430492.1544534832, 10.188552523383924]), ('13453*', [-1430492.1544534832, 11.188552523383924]), ('21553*', [-1430487.4829424054, 11.188556312721769]), ('35551*', [-1430453.3881306006, 20.197506111560305]), ('11434*', [-1430419.1698591665, 10.358111758033958]), ('2434*', [-1430419.1698591663, 9.358111758033958]), ('22552*', [-1430403.0544960413, 14.386638892342694]), ('3114*', [-1430365.082800422, 10.358944961148303]), ('12114*', [-1430365.082800422, 11.358944961148303]), ('3553*', [-1430342.521889154, 10.188552523383924]), ('12553*', [-1430342.521889154, 11.188552523383924]), ('324*', [-1430325.486963723, 9.358944961148303]), ('1224*', [-1430325.486963723, 10.358944961148303]), ('23354*', [-1430215.3936397207, 9.354003223902144]), ('21154*', [-1430080.7015997374, 10.358944961148302]), ('2254*', [-1430027.7739437094, 9.358944961148302]), ('11254*', [-1430027.7739437092, 10.358944961148302]), ('53332*', [-1430022.1975978087, 14.552503040524357]), ('22454*', [-1429982.9660000124, 9.354843541793123]), ('52412*', [-1429908.0565320926, 15.556732536613483]), ('1513*', [-1429875.5868351369, 12.272072108405588]), ('44332*', [-1429873.315076119, 14.552503040524357]), ('5143*', [-1429727.4501076317, 11.272072108405588]), ('14143*', [-1429727.4501076317, 12.272072108405588]), ('35332*', [-1429723.68251179, 14.552503040524357]), ('52432*', [-1429714.0926607908, 14.55269527021214]), ('55213*', [-1429615.760059173, 11.270618667271545]), ('42512*', [-1429613.320642279, 15.556732536613483]), ('55123*', [-1429602.871831123, 11.271080691106144]), ('1543*', [-1429585.6202626333, 11.272072943363703]), ('13153*', [-1429578.5670946615, 12.272072108405588]), ('4153*', [-1429578.5670946613, 11.272072108405588]), ('1453*', [-1429437.4840239473, 11.272072943363703]), ('553*', [-1429437.484023947, 10.272072943363703]), ('55351*', [-1429420.4595137907, 20.475000740963115]), ('42532*', [-1429419.356770977, 14.55269527021214]), ('52342*', [-1429366.0073257063, 14.563764675020478]), ('42352*', [-1428972.756039886, 14.55956266518849]), ('15253*', [-1428958.3788028697, 11.286110505296184]), ('33434*', [-1428794.223311517, 9.429453136259543]), ('42124*', [-1428756.7737206828, 10.4392890044345]), ('42214*', [-1428745.5203792055, 10.44006876024345]), ('24434*', [-1428643.836925972, 9.429453136259543]), ('41134*', [-1428586.5123341277, 10.44006798566403]), ('13234*', [-1428560.181490174, 10.440067985664028]), ('4234*', [-1428560.1814901738, 9.440067985664028]), ('23534*', [-1428494.9150822468, 9.429434419461789]), ('54323*', [-1428360.8728158518, 10.976088810772813]), ('53423*', [-1428222.8921043882, 10.958433848456583]), ('32354*', [-1428213.042339384, 9.43635232063727]), ('45323*', [-1428211.990294162, 10.976088810772813]), ('44423*', [-1428074.0095826986, 10.958433848456583]), ('43523*', [-1427926.1176456192, 10.958427243083227]), ('35423*', [-1427924.3770183693, 10.958433848456583]), ('52112*', [-1427877.9736677562, 17.216647995518553]), ('1312*', [-1427842.9108381362, 18.20809383395393]), ('412*', [-1427842.910838136, 17.20809383395393]), ('34523*', [-1427776.4850812901, 10.958427243083227]), ('52132*', [-1427738.5256106197, 16.226210680798204]), ('43112*', [-1427729.0911460668, 17.216647995518553]), ('3132*', [-1427693.2777800509, 17.20809383395393]), ('12132*', [-1427693.2777800504, 18.20809383395393]), ('42232*', [-1427684.9617780263, 16.208946471387527]), ('25523*', [-1427626.098695745, 10.958427243083227]), ('432*', [-1427617.233599929, 16.20359117794156]), ('1332*', [-1427617.2335999287, 17.20359117794156]), ('43132*', [-1427589.6430889305, 16.226210680798204]), ('34112*', [-1427579.4585817372, 17.216647995518553]), ('2142*', [-1427542.8908982612, 17.20809383395393]), ('11142*', [-1427542.8908982612, 18.20809383395393]), ('33232*', [-1427535.329213697, 16.208946471387527]), ('42142*', [-1427534.7416485648, 16.21688961392354]), ('32242*', [-1427476.2283757578, 16.19868811945491]), ('342*', [-1427467.6010355994, 16.20359117794156]), ('1242*', [-1427467.601035599, 17.20359117794156]), ('34132*', [-1427440.010524601, 16.226210680798204]), ('25112*', [-1427429.0721961919, 17.216647995518553]), ('33142*', [-1427385.1090842357, 16.21688961392354]), ('24232*', [-1427384.942828152, 16.208946471387527]), ('23242*', [-1427325.8419902124, 16.19868811945491]), ('252*', [-1427317.2146500542, 16.20359117794156]), ('1152*', [-1427317.2146500542, 17.20359117794156]), ('41242*', [-1427301.8158401144, 16.209071748004003]), ('25132*', [-1427289.6241390556, 16.226210680798204]), ('32152*', [-1427238.5200490262, 16.21688961392354]), ('24142*', [-1427234.7226986908, 16.21688961392354]), ('43334*', [-1427198.507711276, 9.504300614497462]), ('22252*', [-1427165.840516591, 16.19868811945491]), ('31342*', [-1427152.1827820286, 16.209071748004003]), ('23152*', [-1427088.133663481, 16.21688961392354]), ('55313*', [-1427065.2785325907, 11.051997981496445]), ('34334*', [-1427048.8751469469, 9.504300614497462]), ('31252*', [-1427006.2629736387, 16.209071748004003]), ('4342*', [-1427005.947150934, 15.209072657030335]), ('13342*', [-1427005.947150934, 16.209072657030333]), ('21442*', [-1427001.79590024, 16.209071748004003]), ('54413*', [-1426954.175224446, 11.034037422094219]), ('25334*', [-1426898.4887614015, 9.504300614497462]), ('3442*', [-1426856.314586605, 15.209072657030335]), ('12442*', [-1426856.314586605, 16.209072657030333]), ('21352*', [-1426855.8760918495, 16.209071748004003]), ('53513*', [-1426807.0064033975, 11.03403360147314]), ('45413*', [-1426805.2927027564, 11.034037422094219]), ('42434*', [-1426745.3479151577, 9.51302885640185]), ('3352*', [-1426707.8745900844, 15.209069597115706]), ('12352*', [-1426707.8745900844, 16.209069597115707]), ('11542*', [-1426705.9282010598, 16.209072657030333]), ('2542*', [-1426705.9282010596, 15.209072657030335]), ('14112*', [-1426680.063914648, 18.29809383395393]), ('5112*', [-1426680.0639146476, 17.29809383395393]), ('52*', [-1426666.931571829, 16.29809383395393]), ('142*', [-1426666.931571829, 17.29809383395393]), ('51212*', [-1426666.4061964222, 17.30721484236283]), ('44513*', [-1426658.1238817084, 11.03403360147314]), ('51232*', [-1426590.353406533, 16.312121320673867]), ('23324*', [-1426575.7870208435, 10.248799295525998]), ('33314*', [-1426569.382263536, 10.264837240985697]), ('11452*', [-1426557.4882045395, 16.209069597115707]), ('2452*', [-1426557.4882045393, 15.209069597115706]), ('53212*', [-1426552.5416448812, 16.29400916987495]), ('41312*', [-1426517.5231834515, 17.30721484236283]), ('35513*', [-1426508.4913173786, 11.03403360147314]), ('32534*', [-1426449.127200381, 9.51302885640185]), ('41332*', [-1426441.4703935615, 16.312121320673867]), ('5312*', [-1426440.0227674616, 16.302645101108837]), ('14312*', [-1426440.0227674616, 17.302645101108837]), ('24314*', [-1426418.9958779907, 10.264837240985697]), ('44212*', [-1426403.6591231916, 16.29400916987495]), ('31412*', [-1426367.8901253657, 17.30721484236283]), ('5332*', [-1426297.474937286, 15.312092969513923]), ('14332*', [-1426297.474937286, 16.312092969513923]), ('31432*', [-1426291.837335476, 16.312121320673867]), ('4412*', [-1426291.140245772, 16.302645101108837]), ('13412*', [-1426291.140245772, 17.302645101108837]), ('23414*', [-1426269.6118228184, 10.26483655524162]), ('35212*', [-1426254.0265588618, 16.29400916987495]), ('21512*', [-1426217.503243577, 17.30721484236283]), ('33224*', [-1426214.811606547, 10.265140578703047]), ('32234*', [-1426201.1335342587, 10.265436292041546]), ('4432*', [-1426148.5924155961, 15.312092969513923]), ('13432*', [-1426148.5924155961, 16.312092969513923]), ('3512*', [-1426141.5076814427, 16.302645101108837]), ('12512*', [-1426141.5076814427, 17.302645101108837]), ('21532*', [-1426141.4504536872, 16.312121320673867]), ('23234*', [-1426137.7967768249, 10.252855434969943]), ('52523*', [-1426102.2852700243, 11.062921062012627]), ('42134*', [-1426096.646356625, 10.277015620677687]), ('22114*', [-1426067.331587543, 11.256557893728486]), ('24224*', [-1426064.4252210024, 10.265140578703047]), ('43442*', [-1426038.604640707, 14.21152013751759]), ('33134*', [-1426013.0531371797, 10.264831888268116]), ('21124*', [-1426008.5232654829, 11.256557578695375]), ('31114*', [-1426004.4107673394, 11.268948137576675]), ('3214*', [-1426004.4107673392, 10.268948137576677]), ('12214*', [-1426004.4107673392, 11.268948137576677]), ('21214*', [-1426004.410767339, 11.268948137576677]), ('3532*', [-1425998.9598512673, 15.312092969513923]), ('12532*', [-1425998.959851267, 16.312092969513923]), ('11224*', [-1425995.1914461544, 11.256557578695377]), ('2224*', [-1425995.1914461541, 10.256557578695377]), ('54443*', [-1425939.8329524538, 10.038252487087178]), ('34442*', [-1425888.9720763771, 14.21152013751759]), ('24134*', [-1425862.6667516348, 10.264831888268116]), ('11314*', [-1425854.0243817945, 11.268948137576677]), ('2314*', [-1425854.0243817943, 10.268948137576677]), ('53543*', [-1425792.948230848, 10.038251498161248]), ('45443*', [-1425790.9504307641, 10.038252487087178]), ('33542*', [-1425740.5853623892, 14.21151981593748]), ('25442*', [-1425738.5856908322, 14.21152013751759]), ('53453*', [-1425713.826812531, 10.033706050024048]), ('22154*', [-1425706.6968581066, 10.268948137576675]), ('3124*', [-1425652.4612694387, 10.268944961148303]), ('12124*', [-1425652.4612694385, 11.268944961148303]), ('44543*', [-1425644.0657091583, 10.038251498161248]), ('33452*', [-1425634.0286502943, 14.206389452191276]), ('15212*', [-1425616.698175915, 17.388660680798203]), ('51323*', [-1425600.705770923, 12.114921082079972]), ('24542*', [-1425590.1989768443, 14.21151981593748]), ('44453*', [-1425564.9442908412, 10.033706050024048]), ('15323*', [-1425554.008335654, 12.11071946063285]), ('11134*', [-1425502.0743876498, 11.268944961148303]), ('2134*', [-1425502.0743876493, 10.268944961148303]), ('54113*', [-1425501.4353677656, 12.121206326957532]), ('35543*', [-1425494.433144829, 10.038251498161248]), ('24452*', [-1425483.6422647492, 14.206389452191276]), ('41423*', [-1425451.822757952, 12.114921082079972]), ('55223*', [-1425433.6950098036, 11.122298770757636]), ('43553*', [-1425417.3196157976, 10.033705007005459]), ('35453*', [-1425415.311726512, 10.033706050024048]), ('5423*', [-1425405.872096968, 11.11071946063285]), ('14423*', [-1425405.872096968, 12.11071946063285]), ('21254*', [-1425355.4194622226, 10.268944961148302]), ('334*', [-1425355.077956781, 9.268943242061733]), ('1234*', [-1425355.0779567808, 10.268943242061733]), ('45113*', [-1425352.5528460757, 12.121206326957532]), ('23552*', [-1425334.5078063882, 14.2063890693482]), ('52334*', [-1425316.985992056, 9.58111947271551]), ('31523*', [-1425302.1896998663, 12.114921082079972]), ('34553*', [-1425267.6870514683, 10.033705007005459]), ('4523*', [-1425256.9895752782, 11.11071946063285]), ('13523*', [-1425256.9895752782, 12.11071946063285]), ('114*', [-1425204.6915712361, 10.26894324206173]), ('24*', [-1425204.691571236, 9.26894324206173]), ('53243*', [-1425182.3422839886, 11.12041038100569]), ('25553*', [-1425117.3006659232, 10.033705007005459]), ('2354*', [-1425055.5828949779, 9.268942257572757]), ('11354*', [-1425055.5828949776, 10.268942257572757]), ('44243*', [-1425033.459762299, 11.12041038100569]), ('23454*', [-1425007.591854658, 9.264434059049426]), ('52253*', [-1424906.4789296004, 11.12041038100569]), ('35243*', [-1424883.8271979697, 11.12041038100569]), ('51142*', [-1424883.2667115962, 16.389392814878665]), ('5242*', [-1424844.066833264, 15.389392814878663]), ('14242*', [-1424844.0668332637, 16.38939281487866]), ('53412*', [-1424804.003774319, 15.383617706208009]), ('43253*', [-1424757.5964079106, 11.12041038100569]), ('44412*', [-1424655.1212526301, 15.383617706208009]), ('34253*', [-1424607.9638435815, 11.12041038100569]), ('41152*', [-1424601.7293229178, 16.389392814878665]), ('53432*', [-1424598.9673701632, 14.378676631280943]), ('4252*', [-1424549.33094345, 15.389392814878663]), ('13252*', [-1424549.3309434499, 16.38939281487866]), ('55241*', [-1424543.6870533954, 23.279977383689562]), ('43512*', [-1424507.4139876158, 15.38359480608649]), ('35412*', [-1424505.4886883006, 15.383617706208009]), ('25253*', [-1424457.577458036, 11.12041038100569]), ('44432*', [-1424450.0848484735, 14.378676631280943]), ('34512*', [-1424357.781423287, 15.38359480608649]), ('55343*', [-1424352.0633107643, 10.135201611316385]), ('42314*', [-1424306.5413318158, 10.35521086595188]), ('32324*', [-1424303.0275205292, 10.339836394201303]), ('43532*', [-1424302.636534589, 14.378659098082164]), ('35432*', [-1424300.4522841442, 14.378676631280943]), ('51413*', [-1424273.5618758649, 12.188957956004163]), ('54251*', [-1424270.582332551, 23.279977383689562]), ('15413*', [-1424253.575591625, 12.196475818657829]), ('15232*', [-1424222.7443262532, 16.483121320673867]), ('25512*', [-1424207.3950377416, 15.38359480608649]), ('55151*', [-1424191.668728941, 23.382375835998435]), ('54312*', [-1424156.2191655375, 15.462372684492793]), ('34532*', [-1424153.00397026, 14.378659098082164]), ('41513*', [-1424124.6788628942, 12.188957956004163]), ('54353*', [-1424123.3960078096, 10.13105307111866]), ('53342*', [-1424121.7929034939, 14.401054514937526]), ('45251*', [-1424121.6998108614, 23.279977383689562]), ('5513*', [-1424105.4393529387, 11.196475818657829]), ('14513*', [-1424105.4393529387, 12.196475818657829]), ('54143*', [-1424096.5076984058, 11.188921626083413]), ('32414*', [-1424010.320617039, 10.35521086595188]), ('45312*', [-1424007.3366438474, 15.462372684492793]), ('22424*', [-1424005.3145005156, 10.339836394201303]), ('25532*', [-1424002.617584715, 14.378659098082164]), ('42224*', [-1423995.6906191602, 10.354752045773337]), ('45353*', [-1423974.5134861197, 10.13105307111866]), ('44342*', [-1423972.9103818045, 14.401054514937526]), ('53153*', [-1423952.111992439, 11.188921626083413]), ('45143*', [-1423947.6251767164, 11.188921626083413]), ('51551*', [-1423940.764704578, 23.37041173515902]), ('52442*', [-1423921.8973181206, 14.39130561387371]), ('35342*', [-1423823.2778174751, 14.401054514937526]), ('44153*', [-1423803.2294707494, 11.188921626083413]), ('15551*', [-1423797.916240708, 23.370407761022523]), ('51443*', [-1423759.3270294312, 11.188556312721769]), ('43352*', [-1423718.7974484342, 14.396409570991995]), ('22514*', [-1423712.6075970253, 10.35521086595188]), ('55541*', [-1423708.8997883908, 20.202413357554597]), ('41234*', [-1423692.835467011, 10.35813546876504]), ('35153*', [-1423653.59690642, 11.188921626083413]), ('52553*', [-1423651.4722222222, 10.13767650341595]), ('42542*', [-1423627.1614283072, 14.39130561387371]), ('15443*', [-1423615.9112631094, 11.188573289663607]), ('51353*', [-1423615.59053702, 11.188556312721769]), ('41543*', [-1423610.4440164603, 11.188556312721769]), ('55451*', [-1423578.2006856196, 20.19750673647386]), ('34352*', [-1423569.164884105, 14.396409570991995]), ('43214*', [-1423554.5864414182, 10.358090269315449]), ('43124*', [-1423550.1228045633, 10.358055039205992]), ('31334*', [-1423543.2024089252, 10.35813546876504]), ('42452*', [-1423529.2899372738, 14.386638892342694]), ('15353*', [-1423469.8587994846, 11.188552523383924]), ('14543*', [-1423467.7750244234, 11.188573289663607]), ('5543*', [-1423467.7750244231, 10.188573289663607]), ('41453*', [-1423466.7075240493, 11.188556312721769]), ('54551*', [-1423432.033078237, 20.197506111560305]), ('25352*', [-1423418.7784985597, 14.396409570991995]), ('34214*', [-1423404.9538770893, 10.358090269315449]), ('34124*', [-1423400.490240234, 10.358055039205992]), ('4334*', [-1423398.7363706802, 9.358111758033958]), ('13334*', [-1423398.73637068, 10.358111758033958]), ('21434*', [-1423392.8155271364, 10.35813546876504]), ('14453*', [-1423321.7225607983, 11.188552523383924]), ('5453*', [-1423321.722560798, 10.188552523383924]), ('31553*', [-1423317.0744659638, 11.188556312721769]), ('45551*', [-1423283.150556547, 20.197506111560305]), ('25214*', [-1423254.5674915442, 10.358090269315449]), ('25124*', [-1423250.1038546888, 10.358055039205992]), ('3434*', [-1423249.1038063508, 9.358111758033958]), ('12434*', [-1423249.1038063506, 10.358111758033958]), ('32552*', [-1423233.0692224973, 14.386638892342694]), ('4114*', [-1423195.2878623903, 10.358944961148303]), ('13114*', [-1423195.2878623903, 11.358944961148303]), ('4553*', [-1423172.8400391089, 10.188552523383924]), ('13553*', [-1423172.8400391089, 11.188552523383924]), ('1324*', [-1423155.890502313, 10.358944961148303]), ('424*', [-1423155.8905023127, 9.358944961148303]), ('11534*', [-1423098.717420806, 10.358111758033958]), ('2534*', [-1423098.7174208057, 9.358111758033958]), ('33354*', [-1423046.3490280216, 9.354003223902144]), ('31154*', [-1422912.3321403598, 10.358944961148302]), ('24354*', [-1422895.9626424764, 9.354003223902144]), ('3254*', [-1422859.6697875364, 9.358944961148302]), ('12254*', [-1422859.6697875364, 10.358944961148302]), ('32454*', [-1422815.0864464752, 9.354843541793123]), ('54332*', [-1422705.9851547342, 14.552503040524357]), ('15143*', [-1422560.8513435754, 12.272072108405588]), ('45332*', [-1422557.1026330448, 14.552503040524357]), ('22554*', [-1422517.3734264614, 9.354843541793123]), ('52512*', [-1422447.2939593454, 15.556732536613483]), ('14153*', [-1422412.7146160712, 12.272072108405588]), ('5153*', [-1422412.714616071, 11.272072108405588]), ('1553*', [-1422272.338733122, 11.272072943363703]), ('52532*', [-1422254.3023441476, 14.55269527021214]), ('52352*', [-1421809.9402273095, 14.55956266518849]), ('43434*', [-1421632.3024054698, 9.429453136259543]), ('52124*', [-1421595.0405330653, 10.4392890044345]), ('52214*', [-1421583.8435996696, 10.44006876024345]), ('34434*', [-1421482.6698411405, 9.429453136259543]), ('51134*', [-1421425.6325924182, 10.44006798566403]), ('5234*', [-1421399.4337334759, 9.440067985664028]), ('14234*', [-1421399.4337334759, 10.440067985664028]), ('33534*', [-1421334.4944775235, 9.429434419461789]), ('25434*', [-1421332.2834555956, 9.429453136259543]), ('24534*', [-1421184.1080919784, 9.429434419461789]), ('42354*', [-1421054.0346395124, 9.43635232063727]), ('55323*', [-1421052.987867733, 10.976088810772813]), ('54423*', [-1420915.6987932636, 10.958433848456583]), ('53523*', [-1420768.5481738206, 10.958427243083227]), ('45423*', [-1420766.816271574, 10.958433848456583]), ('512*', [-1420685.7584456932, 17.20809383395393]), ('1412*', [-1420685.758445693, 18.20809383395393]), ('44523*', [-1420619.665652131, 10.958427243083227]), ('53112*', [-1420572.5092819887, 17.216647995518553]), ('13132*', [-1420536.8754327225, 18.20809383395393]), ('4132*', [-1420536.8754327223, 17.20809383395393]), ('52232*', [-1420528.601115181, 16.208946471387527]), ('35523*', [-1420470.0330878017, 10.958427243083227]), ('1432*', [-1420461.2124288362, 17.20359117794156]), ('532*', [-1420461.212428836, 16.20359117794156]), ('53132*', [-1420433.7602170089, 16.226210680798204]), ('44112*', [-1420423.6267602989, 17.216647995518553]), ('3142*', [-1420387.2423746365, 17.20809383395393]), ('12142*', [-1420387.2423746362, 18.20809383395393]), ('43232*', [-1420379.7185934915, 16.208946471387527]), ('52142*', [-1420379.1339735661, 16.21688961392354]), ('42242*', [-1420320.9140022192, 16.19868811945491]), ('442*', [-1420312.329907146, 16.20359117794156]), ('1342*', [-1420312.329907146, 17.20359117794156]), ('44132*', [-1420284.8776953192, 16.226210680798204]), ('35112*', [-1420273.9941959693, 17.216647995518553]), ('11152*', [-1420236.8554928477, 18.20809383395393]), ('2152*', [-1420236.8554928475, 17.20809383395393]), ('43142*', [-1420230.2514518767, 16.21688961392354]), ('34232*', [-1420230.0860291624, 16.208946471387527]), ('33242*', [-1420171.2814378897, 16.19868811945491]), ('352*', [-1420162.6973428167, 16.20359117794156]), ('1252*', [-1420162.6973428167, 17.20359117794156]), ('51242*', [-1420147.3757203803, 16.209071748004003]), ('35132*', [-1420135.2451309897, 16.226210680798204]), ('42152*', [-1420084.3972034259, 16.21688961392354]), ('34142*', [-1420080.6188875476, 16.21688961392354]), ('25232*', [-1420079.6996436173, 16.208946471387527]), ('53334*', [-1420044.5854300347, 9.504300614497462]), ('24242*', [-1420020.8950523448, 16.19868811945491]), ('32252*', [-1420012.0819817178, 16.19868811945491]), ('41342*', [-1419998.4927074092, 16.209071748004003]), ('33152*', [-1419934.764639096, 16.21688961392354]), ('25142*', [-1419930.2325020023, 16.21688961392354]), ('44334*', [-1419895.7029083455, 9.504300614497462]), ('23252*', [-1419861.6955961727, 16.19868811945491]), ('41252*', [-1419853.3043312358, 16.209071748004003]), ('5342*', [-1419852.990091613, 15.209072657030335]), ('14342*', [-1419852.9900916126, 16.209072657030333]), ('31442*', [-1419848.8596493236, 16.209071748004003]), ('24152*', [-1419784.378253551, 16.21688961392354]), ('35334*', [-1419746.070344016, 9.504300614497462]), ('4442*', [-1419704.1075699234, 15.209072657030335]), ('13442*', [-1419704.1075699232, 16.209072657030333]), ('31352*', [-1419703.6712731505, 16.209071748004003]), ('21542*', [-1419698.4727675344, 16.209071748004003]), ('55413*', [-1419653.3414364762, 11.034037422094219]), ('52434*', [-1419593.6971258954, 9.51302885640185]), ('4352*', [-1419556.411638221, 15.209069597115706]), ('13352*', [-1419556.411638221, 16.209069597115707]), ('3542*', [-1419554.475005594, 15.209072657030335]), ('12542*', [-1419554.4750055936, 16.209072657030333]), ('21452*', [-1419553.2843913618, 16.209071748004003]), ('15112*', [-1419528.7403655439, 18.29809383395393]), ('152*', [-1419515.6738494197, 17.29809383395393]), ('54513*', [-1419506.9103083995, 11.03403360147314]), ('33324*', [-1419424.9861662283, 10.248799295525998]), ('43314*', [-1419418.6135131696, 10.264837240985697]), ('3452*', [-1419406.7790738917, 15.209069597115706]), ('12452*', [-1419406.7790738917, 16.209069597115707]), ('51312*', [-1419367.0143799854, 17.30721484236283]), ('45513*', [-1419358.0277867098, 11.03403360147314]), ('42534*', [-1419298.961236082, 9.51302885640185]), ('51332*', [-1419291.3428094885, 16.312121320673867]), ('15312*', [-1419289.9024397053, 17.302645101108837]), ('24324*', [-1419274.5997806832, 10.248799295525998]), ('34314*', [-1419268.9809488405, 10.264837240985697]), ('11552*', [-1419256.3926883468, 16.209069597115707]), ('2552*', [-1419256.3926883466, 15.209069597115706]), ('54212*', [-1419253.7210704891, 16.29400916987495]), ('41412*', [-1419218.1313670143, 17.30721484236283]), ('15332*', [-1419148.0691394936, 16.312092969513923]), ('41432*', [-1419142.4597965174, 16.312121320673867]), ('14412*', [-1419141.7662010193, 17.302645101108837]), ('5412*', [-1419141.766201019, 16.302645101108837]), ('23424*', [-1419125.1456040647, 10.248798412583756]), ('33414*', [-1419120.3456906397, 10.26483655524162]), ('25314*', [-1419118.5945632954, 10.264837240985697]), ('45212*', [-1419104.838548799, 16.29400916987495]), ('31512*', [-1419068.4983089291, 17.30721484236283]), ('43224*', [-1419065.8201638993, 10.265140578703047]), ('42234*', [-1419052.2106538082, 10.265436292041546]), ('5432*', [-1418999.9329008078, 15.312092969513923]), ('14432*', [-1418999.9329008076, 16.312092969513923]), ('13512*', [-1418992.8836793297, 17.302645101108837]), ('4512*', [-1418992.8836793294, 16.302645101108837]), ('31532*', [-1418992.8267384318, 16.312121320673867]), ('33234*', [-1418989.1913758547, 10.252855434969943]), ('24414*', [-1418969.9593050946, 10.26483655524162]), ('52134*', [-1418948.247224724, 10.277015620677687]), ('32114*', [-1418919.0793977657, 11.256557893728486]), ('34224*', [-1418916.18759957, 10.265140578703047]), ('53442*', [-1418890.4964465576, 14.21152013751759]), ('43134*', [-1418865.073021549, 10.264831888268116]), ('31124*', [-1418860.5658561187, 11.256557578695375]), ('4214*', [-1418856.4739721308, 10.268948137576677]), ('41114*', [-1418856.4739721308, 11.268948137576675]), ('13214*', [-1418856.4739721306, 11.268948137576677]), ('31214*', [-1418856.4739721303, 11.268948137576677]), ('4532*', [-1418851.0503791184, 15.312092969513923]), ('13532*', [-1418851.0503791182, 16.312092969513923]), ('3224*', [-1418847.3008633729, 10.256557578695377]), ('21224*', [-1418847.3008633729, 11.256557578695377]), ('12224*', [-1418847.3008633729, 11.256557578695377]), ('24234*', [-1418838.8049903095, 10.252855434969943]), ('23514*', [-1418820.5095152934, 10.264835070903667]), ('23114*', [-1418768.6930122208, 11.256557893728486]), ('25224*', [-1418765.8012140247, 10.265140578703047]), ('44442*', [-1418741.6139248675, 14.21152013751759]), ('34134*', [-1418715.44045722, 10.264831888268116]), ('3314*', [-1418706.8414078015, 10.268948137576677]), ('21314*', [-1418706.8414078015, 11.268948137576677]), ('12314*', [-1418706.8414078015, 11.268948137576677]), ('11324*', [-1418696.9144778282, 11.256557578695377]), ('2324*', [-1418696.914477828, 10.256557578695377]), ('55443*', [-1418644.0836189177, 10.038252487087178]), ('43542*', [-1418593.971008616, 14.21151981593748]), ('35442*', [-1418591.9813605384, 14.21152013751759]), ('25134*', [-1418565.0540716746, 10.264831888268116]), ('32154*', [-1418560.2523725925, 10.268948137576675]), ('11414*', [-1418556.4550222566, 11.268948137576677]), ('2414*', [-1418556.4550222564, 10.268948137576677]), ('13124*', [-1418506.2886432237, 11.268944961148303]), ('4124*', [-1418506.2886432235, 10.268944961148303]), ('54543*', [-1418497.9351662172, 10.038251498161248]), ('22254*', [-1418492.3572952915, 10.256557578695377]), ('43452*', [-1418487.9484187416, 14.206389452191276]), ('34542*', [-1418444.338444287, 14.21151981593748]), ('54453*', [-1418419.2103489859, 10.033706050024048]), ('23154*', [-1418409.865987047, 10.268948137576675]), ('3134*', [-1418356.6555851377, 10.268944961148303]), ('12134*', [-1418356.6555851374, 11.268944961148303]), ('45543*', [-1418349.0526445273, 10.038251498161248]), ('34452*', [-1418338.3158544125, 14.206389452191276]), ('51423*', [-1418306.6558448947, 12.114921082079972]), ('25542*', [-1418293.9520587416, 14.21151981593748]), ('53553*', [-1418272.3256519097, 10.033705007005459]), ('45453*', [-1418270.3278272962, 10.033706050024048]), ('15423*', [-1418260.935514489, 12.11071946063285]), ('31254*', [-1418210.735776748, 10.268944961148302]), ('1334*', [-1418210.3959831242, 10.268943242061733]), ('434*', [-1418210.3959831237, 9.268943242061733]), ('55113*', [-1418207.883529695, 12.121206326957532]), ('1114*', [-1418206.2687033494, 11.268944961148302]), ('214*', [-1418206.268703349, 10.268944961148302]), ('33552*', [-1418189.9289419036, 14.2063890693482]), ('25452*', [-1418187.9294688674, 14.206389452191276]), ('41523*', [-1418157.772831923, 12.114921082079972]), ('44553*', [-1418123.4431302203, 10.033705007005459]), ('14523*', [-1418112.7992758027, 12.11071946063285]), ('5523*', [-1418112.7992758024, 11.11071946063285]), ('34*', [-1418060.7634187948, 9.26894324206173]), ('124*', [-1418060.7634187948, 10.26894324206173]), ('21354*', [-1418060.3488949595, 10.268944961148302]), ('24552*', [-1418039.5425563583, 14.2063890693482]), ('35553*', [-1417973.810565891, 10.033705007005459]), ('12354*', [-1417912.4021591544, 10.268942257572757]), ('3354*', [-1417912.4021591542, 9.268942257572757]), ('1154*', [-1417910.37703325, 10.268943242061733]), ('254*', [-1417910.3770332495, 9.268943242061733]), ('54243*', [-1417890.3899200694, 11.12041038100569]), ('33454*', [-1417864.6516769426, 9.264434059049426]), ('11454*', [-1417762.0157736095, 10.268942257572757]), ('2454*', [-1417762.015773609, 9.268942257572757]), ('45243*', [-1417741.50739838, 11.12041038100569]), ('24454*', [-1417714.2652913972, 9.264434059049426]), ('15242*', [-1417701.9463350028, 16.38939281487866]), ('53253*', [-1417615.9093480953, 11.12041038100569]), ('23554*', [-1417565.1569894108, 9.264433020700213]), ('54412*', [-1417513.9478559752, 15.383617706208009]), ('44253*', [-1417467.0268264057, 11.12041038100569]), ('51152*', [-1417460.8235566688, 16.389392814878665]), ('14252*', [-1417408.6878273736, 16.38939281487866]), ('5252*', [-1417408.6878273734, 15.389392814878663]), ('53512*', [-1417366.980982917, 15.38359480608649]), ('45412*', [-1417365.0653342851, 15.383617706208009]), ('35253*', [-1417317.3942620761, 11.12041038100569]), ('54432*', [-1417309.9392096899, 14.378676631280943]), ('44512*', [-1417218.0984612275, 15.38359480608649]), ('52314*', [-1417167.1152139371, 10.35521086595188]), ('42324*', [-1417163.6190158508, 10.339836394201303]), ('53532*', [-1417163.2299897524, 14.378659098082164]), ('45432*', [-1417161.0566880002, 14.378676631280943]), ('35512*', [-1417068.4658968982, 15.38359480608649]), ('44532*', [-1417014.347468063, 14.378659098082164]), ('51513*', [-1416986.164342079, 12.188957956004163]), ('55251*', [-1416983.2002227323, 23.279977383689562]), ('15513*', [-1416967.0212713776, 12.196475818657829]), ('42414*', [-1416872.3793241242, 10.35521086595188]), ('55312*', [-1416869.4103082856, 15.462372684492793]), ('32424*', [-1416867.3983010745, 10.339836394201303]), ('35532*', [-1416864.7149037337, 14.378659098082164]), ('52224*', [-1416857.8226600294, 10.354752045773337]), ('55353*', [-1416836.7516787003, 10.13105307111866]), ('54342*', [-1416835.156610046, 14.401054514937526]), ('55143*', [-1416809.9981486388, 11.188921626083413]), ('45342*', [-1416686.2740883564, 14.401054514937526]), ('54153*', [-1416666.3262352298, 11.188921626083413]), ('53352*', [-1416582.3174337365, 14.396409570991995]), ('32514*', [-1416576.1586093474, 10.35521086595188]), ('22524*', [-1416569.6852810606, 10.339836394201303]), ('51234*', [-1416556.4855883778, 10.35813546876504]), ('45153*', [-1416517.4437135402, 11.188921626083413]), ('52542*', [-1416491.1407449236, 14.39130561387371]), ('51543*', [-1416474.507130155, 11.188556312721769]), ('44352*', [-1416433.4349120469, 14.396409570991995]), ('53214*', [-1416418.929544721, 10.358090269315449]), ('53124*', [-1416414.4882821261, 10.358055039205992]), ('41334*', [-1416407.6025754071, 10.35813546876504]), ('52452*', [-1416393.7598408947, 14.386638892342694]), ('15543*', [-1416332.5532754136, 11.188573289663607]), ('51453*', [-1416331.4911259522, 11.188556312721769]), ('35352*', [-1416283.8023477173, 14.396409570991995]), ('44214*', [-1416270.0470230319, 10.358090269315449]), ('44124*', [-1416265.6057604367, 10.358055039205992]), ('5334*', [-1416263.8606822644, 9.358111758033958]), ('14334*', [-1416263.8606822644, 10.358111758033958]), ('31434*', [-1416257.9695173213, 10.35813546876504]), ('15453*', [-1416187.232908948, 11.188552523383924]), ('41553*', [-1416182.6081129813, 11.188556312721769]), ('55551*', [-1416148.8542492944, 20.197506111560305]), ('35214*', [-1416120.4144587023, 10.358090269315449]), ('35124*', [-1416115.9731961072, 10.358055039205992]), ('4434*', [-1416114.9781605748, 9.358111758033958]), ('13434*', [-1416114.9781605748, 10.358111758033958]), ('21534*', [-1416107.5826355326, 10.35813546876504]), ('42552*', [-1416099.0239510813, 14.386638892342694]), ('5114*', [-1416061.431972418, 10.358944961148303]), ('14114*', [-1416061.431972418, 11.358944961148303]), ('5553*', [-1416039.0966702625, 10.188552523383924]), ('14553*', [-1416039.0966702625, 11.188552523383924]), ('524*', [-1416022.2320940858, 9.358944961148303]), ('1424*', [-1416022.2320940855, 10.358944961148303]), ('3534*', [-1415965.3455962455, 9.358111758033958]), ('12534*', [-1415965.3455962455, 10.358111758033958]), ('43354*', [-1415913.239703324, 9.354003223902144]), ('41154*', [-1415779.8945837403, 10.358944961148302]), ('34354*', [-1415763.6071389946, 9.354003223902144]), ('13254*', [-1415727.4962042724, 10.358944961148302]), ('4254*', [-1415727.4962042721, 9.358944961148302]), ('42454*', [-1415683.1363400123, 9.354843541793123]), ('25354*', [-1415613.2207534497, 9.354003223902144]), ('55332*', [-1415426.445686672, 14.552503040524357]), ('32554*', [-1415386.9156252355, 9.354843541793123]), ('15153*', [-1415282.7814237145, 12.272072108405588]), ('53434*', [-1414506.281078402, 9.429453136259543]), ('44434*', [-1414357.3985567125, 9.429453136259543]), ('15234*', [-1414274.579675272, 10.440067985664028]), ('43534*', [-1414209.9659314242, 9.429434419461789]), ('35434*', [-1414207.7659923832, 9.429453136259543]), ('34534*', [-1414060.3333670949, 9.429434419461789]), ('52354*', [-1413930.91191599, 9.43635232063727]), ('25534*', [-1413909.9469815497, 9.429434419461789]), ('55423*', [-1413645.1332481855, 10.958433848456583]), ('1512*', [-1413564.4817297547, 18.20809383395393]), ('54523*', [-1413498.7202304774, 10.958427243083227]), ('5132*', [-1413416.3450022505, 17.20809383395393]), ('14132*', [-1413416.3450022503, 18.20809383395393]), ('45523*', [-1413349.8377087878, 10.958427243083227]), ('1532*', [-1413341.0612639296, 17.20359117794156]), ('54112*', [-1413303.6639959197, 17.216647995518553]), ('4142*', [-1413267.461989279, 17.20809383395393]), ('13142*', [-1413267.461989279, 18.20809383395393]), ('53232*', [-1413259.9759215603, 16.208946471387527]), ('52242*', [-1413201.466092, 16.19868811945491]), ('1442*', [-1413192.9250252433, 17.20359117794156]), ('542*', [-1413192.925025243, 16.20359117794156]), ('54132*', [-1413165.610419355, 16.226210680798204]), ('45112*', [-1413154.78147423, 17.216647995518553]), ('3152*', [-1413117.8289311933, 17.20809383395393]), ('12152*', [-1413117.8289311933, 18.20809383395393]), ('53142*', [-1413111.2579933933, 16.21688961392354]), ('44232*', [-1413111.0933998707, 16.208946471387527]), ('43242*', [-1413052.5835703104, 16.19868811945491]), ('452*', [-1413044.0425035537, 16.20359117794156]), ('1352*', [-1413044.0425035537, 17.20359117794156]), ('45132*', [-1413016.7278976655, 16.226210680798204]), ('52152*', [-1412966.1348485358, 16.21688961392354]), ('44142*', [-1412962.3754717042, 16.21688961392354]), ('35232*', [-1412961.4608355416, 16.208946471387527]), ('34242*', [-1412902.951005981, 16.19868811945491]), ('42252*', [-1412894.1821114253, 16.19868811945491]), ('51342*', [-1412880.6609542083, 16.209071748004003]), ('43152*', [-1412817.2523268461, 16.21688961392354]), ('35142*', [-1412812.7429073749, 16.21688961392354]), ('54334*', [-1412778.3863954775, 9.504300614497462]), ('25242*', [-1412752.5646204362, 16.19868811945491]), ('33252*', [-1412744.549547096, 16.19868811945491]), ('51252*', [-1412736.2003439025, 16.209071748004003]), ('15342*', [-1412735.887679425, 16.209072657030333]), ('41442*', [-1412731.7779412377, 16.209071748004003]), ('34152*', [-1412667.6197625168, 16.21688961392354]), ('45334*', [-1412629.5038737876, 9.504300614497462]), ('24252*', [-1412594.163161551, 16.19868811945491]), ('14442*', [-1412587.7514407393, 16.209072657030333]), ('5442*', [-1412587.7514407388, 15.209072657030335]), ('41352*', [-1412587.3173309313, 16.209071748004003]), ('31542*', [-1412582.144883152, 16.209071748004003]), ('25152*', [-1412517.2333769717, 16.21688961392354]), ('5352*', [-1412440.7958441835, 15.209069597115706]), ('14352*', [-1412440.7958441835, 16.209069597115707]), ('13542*', [-1412438.8689190494, 16.209072657030333]), ('4542*', [-1412438.8689190492, 15.209072657030335]), ('31452*', [-1412437.684272846, 16.209071748004003]), ('43324*', [-1412310.0291506352, 10.248799295525998]), ('53314*', [-1412303.6884409008, 10.264837240985697]), ('4452*', [-1412291.913322494, 15.209069597115706]), ('13452*', [-1412291.913322494, 16.209069597115707]), ('21552*', [-1412287.2973910565, 16.209071748004003]), ('55513*', [-1412243.4064042051, 11.03403360147314]), ('52534*', [-1412184.6359283773, 9.51302885640185]), ('34324*', [-1412160.396586306, 10.248799295525998]), ('44314*', [-1412154.8059192114, 10.264837240985697]), ('3552*', [-1412142.2807581646, 15.209069597115706]), ('12552*', [-1412142.2807581646, 16.209069597115707]), ('51412*', [-1412104.2112241122, 17.30721484236283]), ('51432*', [-1412028.918962121, 16.312121320673867]), ('15412*', [-1412028.228843314, 17.302645101108837]), ('33424*', [-1412011.6915581468, 10.248798412583756]), ('25324*', [-1412010.0102007606, 10.248799295525998]), ('43414*', [-1412006.9157045903, 10.26483655524162]), ('35314*', [-1412005.173354882, 10.264837240985697]), ('55212*', [-1411991.4862932735, 16.29400916987495]), ('41512*', [-1411955.328211141, 17.30721484236283]), ('53224*', [-1411952.663490482, 10.265140578703047]), ('52234*', [-1411939.1221989163, 10.265436292041546]), ('15432*', [-1411887.1064914404, 16.312092969513923]), ('5512*', [-1411880.0926046283, 16.302645101108837]), ('14512*', [-1411880.092604628, 17.302645101108837]), ('41532*', [-1411880.0359491506, 16.312121320673867]), ('43234*', [-1411876.4188090567, 10.252855434969943]), ('24424*', [-1411861.305172602, 10.248798412583756]), ('34414*', [-1411857.2831402612, 10.26483655524162]), ('42114*', [-1411806.6582716675, 11.256557893728486]), ('44224*', [-1411803.7809687925, 10.265140578703047]), ('53134*', [-1411752.922605808, 10.264831888268116]), ('41124*', [-1411748.4380328283, 11.256557578695375]), ('51114*', [-1411744.366659666, 11.268948137576675]), ('14214*', [-1411744.366659666, 11.268948137576677]), ('5214*', [-1411744.3666596657, 10.268948137576677]), ('41214*', [-1411744.3666596657, 11.268948137576677]), ('14532*', [-1411738.9702527549, 16.312092969513923]), ('5532*', [-1411738.9702527544, 15.312092969513923]), ('4224*', [-1411735.2395316928, 10.256557578695377]), ('31224*', [-1411735.2395316926, 11.256557578695377]), ('13224*', [-1411735.2395316926, 11.256557578695377]), ('34234*', [-1411726.7862447274, 10.252855434969943]), ('23524*', [-1411711.8832424965, 10.248795734083544]), ('33514*', [-1411708.582476931, 10.264835070903667]), ('25414*', [-1411706.8967547156, 10.26483655524162]), ('33114*', [-1411657.025707338, 11.256557893728486]), ('35224*', [-1411654.148404463, 10.265140578703047]), ('54442*', [-1411630.0823556136, 14.21152013751759]), ('44134*', [-1411604.0400841185, 10.264831888268116]), ('31314*', [-1411595.4841379763, 11.268948137576677]), ('13314*', [-1411595.4841379763, 11.268948137576677]), ('4314*', [-1411595.484137976, 10.268948137576677]), ('3324*', [-1411585.6069673637, 10.256557578695377]), ('12324*', [-1411585.6069673637, 11.256557578695377]), ('21324*', [-1411585.6069673635, 11.256557578695377]), ('25234*', [-1411576.3998591818, 10.252855434969943]), ('24514*', [-1411558.1960913856, 10.264835070903667]), ('24114*', [-1411506.639321793, 11.256557893728486]), ('53542*', [-1411483.1795087655, 14.21151981593748]), ('45442*', [-1411481.1998339242, 14.21152013751759]), ('35134*', [-1411454.4075197892, 10.264831888268116]), ('42154*', [-1411449.6298895255, 10.268948137576675]), ('3414*', [-1411445.851573647, 10.268948137576677]), ('21414*', [-1411445.851573647, 11.268948137576677]), ('12414*', [-1411445.851573647, 11.268948137576677]), ('2424*', [-1411435.2205818188, 10.256557578695377]), ('11424*', [-1411435.2205818188, 11.256557578695377]), ('5124*', [-1411395.9366567442, 10.268944961148303]), ('14124*', [-1411395.9366567442, 11.268944961148303]), ('32254*', [-1411382.0751405698, 10.256557578695377]), ('53452*', [-1411377.6883637914, 14.206389452191276]), ('44542*', [-1411334.296987076, 14.21151981593748]), ('33154*', [-1411299.9973251962, 10.268948137576675]), ('11514*', [-1411295.465188102, 11.268948137576677]), ('2514*', [-1411295.4651881019, 10.268948137576677]), ('13134*', [-1411247.053643773, 11.268944961148303]), ('4134*', [-1411247.0536437728, 10.268944961148303]), ('55543*', [-1411239.4888133807, 10.038251498161248]), ('23254*', [-1411231.688755025, 10.256557578695377]), ('44452*', [-1411228.8058421018, 14.206389452191276]), ('35542*', [-1411184.6644227467, 14.21151981593748]), ('55453*', [-1411161.158609247, 10.033706050024048]), ('24154*', [-1411149.6109396508, 10.268948137576675]), ('41254*', [-1411101.8652676004, 10.268944961148302]), ('534*', [-1411101.527177213, 9.268943242061733]), ('1434*', [-1411101.5271772128, 10.268943242061733]), ('314*', [-1411097.4205856877, 10.268944961148302]), ('1214*', [-1411097.4205856875, 11.268944961148302]), ('43552*', [-1411081.1627283243, 14.2063890693482]), ('35452*', [-1411079.1732777725, 14.206389452191276]), ('51523*', [-1411049.1678028675, 12.114921082079972]), ('54553*', [-1411015.0101809537, 10.033705007005459]), ('15523*', [-1411004.4196795253, 12.11071946063285]), ('22335*', [-1410982.1161243299, 9.563954901300571]), ('4*', [-1410952.6446555236, 9.26894324206173]), ('134*', [-1410952.6446555236, 10.26894324206173]), ('31354*', [-1410952.2322095148, 10.268944961148302]), ('2154*', [-1410947.033703899, 10.268944961148303]), ('11154*', [-1410947.0337038988, 11.268944961148303]), ('34552*', [-1410931.5301639948, 14.2063890693482]), ('45553*', [-1410866.1276592643, 10.033705007005459]), ('4354*', [-1410805.027066028, 9.268942257572757]), ('13354*', [-1410805.0270660277, 10.268942257572757]), ('354*', [-1410803.0120911943, 9.268943242061733]), ('1254*', [-1410803.012091194, 10.268943242061733]), ('21454*', [-1410801.845327726, 10.268944961148302]), ('25552*', [-1410781.1437784499, 14.2063890693482]), ('43454*', [-1410757.5159361113, 9.264434059049426]), ('3454*', [-1410655.3945016987, 9.268942257572757]), ('12454*', [-1410655.3945016987, 10.268942257572757]), ('55243*', [-1410634.9889259902, 11.12041038100569]), ('34454*', [-1410607.8833717818, 9.264434059049426]), ('2554*', [-1410505.0081161535, 9.268942257572757]), ('11554*', [-1410505.0081161535, 10.268942257572757]), ('33554*', [-1410459.5224845374, 9.264433020700213]), ('25454*', [-1410457.496986237, 9.264434059049426]), ('54253*', [-1410361.8842051458, 11.12041038100569]), ('24554*', [-1410309.1360989923, 9.264433020700213]), ('15252*', [-1410303.8376340154, 16.38939281487866]), ('55412*', [-1410260.4338014175, 15.383617706208009]), ('45253*', [-1410213.001683456, 11.12041038100569]), ('54512*', [-1410114.203609054, 15.38359480608649]), ('52324*', [-1410059.9972453238, 10.339836394201303]), ('55432*', [-1410057.4477613028, 14.378676631280943]), ('45512*', [-1409965.321087364, 15.38359480608649]), ('54532*', [-1409911.4739305573, 14.378659098082164]), ('52414*', [-1409770.217410869, 10.35521086595188]), ('42424*', [-1409765.2613555104, 10.339836394201303]), ('45532*', [-1409762.5914088676, 14.378659098082164]), ('55342*', [-1409585.0450393, 14.401054514937526]), ('42514*', [-1409475.4815210553, 10.35521086595188]), ('32524*', [-1409469.0406407341, 10.339836394201303]), ('55153*', [-1409417.060937356, 11.188921626083413]), ('54352*', [-1409333.473235264, 14.396409570991995]), ('51334*', [-1409307.770384836, 10.35813546876504]), ('45352*', [-1409184.590713574, 14.396409570991995]), ('54214*', [-1409170.9043383186, 10.358090269315449]), ('54124*', [-1409166.4853378315, 10.358055039205992]), ('15334*', [-1409164.7490069733, 10.358111758033958]), ('41434*', [-1409158.8873718653, 10.35813546876504]), ('51553*', [-1409083.9037213044, 11.188556312721769]), ('45214*', [-1409022.0218166285, 10.358090269315449]), ('45124*', [-1409017.602816142, 10.358055039205992]), ('14434*', [-1409016.6127682875, 10.358111758033958]), ('5434*', [-1409016.6127682873, 9.358111758033958]), ('31534*', [-1409009.2543137798, 10.35813546876504]), ('52552*', [-1409000.7385302722, 14.386638892342694]), ('15114*', [-1408963.3349837665, 11.358944961148303]), ('15553*', [-1408941.1116387178, 11.188552523383924]), ('1524*', [-1408924.3315972898, 10.358944961148303]), ('4534*', [-1408867.7302465977, 9.358111758033958]), ('13534*', [-1408867.7302465977, 10.358111758033958]), ('53354*', [-1408815.885537741, 9.354003223902144]), ('51154*', [-1408683.2088189563, 10.358944961148302]), ('44354*', [-1408667.0030160518, 9.354003223902144]), ('5254*', [-1408631.0730896608, 9.358944961148302]), ('14254*', [-1408631.0730896608, 10.358944961148302]), ('52454*', [-1408586.9355820105, 9.354843541793123]), ('35354*', [-1408517.3704517223, 9.354003223902144]), ('42554*', [-1408292.199692197, 9.354843541793123]), ('22345*', [-1408177.3050091772, 9.422119606590757]), ('54434*', [-1407267.8431427295, 9.429453136259543]), ('53534*', [-1407121.1495327482, 9.429434419461789]), ('45434*', [-1407118.9606210396, 9.429453136259543]), ('44534*', [-1406972.2670110587, 9.429434419461789]), ('35534*', [-1406822.6344467294, 9.429434419461789]), ('15132*', [-1406331.5066783954, 18.20809383395393]), ('23335*', [-1406310.050177904, 9.48718777837109]), ('55523*', [-1406265.332756924, 10.958427243083227]), ('5142*', [-1406183.36995089, 17.20809383395393]), ('14142*', [-1406183.36995089, 18.20809383395393]), ('1542*', [-1406109.2066080743, 17.20359117794156]), ('55112*', [-1406071.2542540098, 17.216647995518553]), ('13152*', [-1406034.4869379192, 18.20809383395393]), ('4152*', [-1406034.486937919, 17.20809383395393]), ('54232*', [-1406027.7851688708, 16.208946471387527]), ('53242*', [-1405969.568623511, 16.19868811945491]), ('552*', [-1405961.0703693887, 16.20359117794156]), ('1452*', [-1405961.0703693887, 17.20359117794156]), ('55132*', [-1405933.89267968, 16.226210680798204]), ('54142*', [-1405879.8126986723, 16.21688961392354]), ('45232*', [-1405878.9026471812, 16.208946471387527]), ('44242*', [-1405820.6861018215, 16.19868811945491]), ('52252*', [-1405811.9611619008, 16.19868811945491]), ('22435*', [-1405806.5401571991, 9.495549945695954]), ('53152*', [-1405735.4169927053, 16.21688961392354]), ('45142*', [-1405730.9301769827, 16.21688961392354]), ('35242*', [-1405671.0535374922, 16.19868811945491]), ('43252*', [-1405663.0786402114, 16.19868811945491]), ('51442*', [-1405650.3710528307, 16.209071748004003]), ('44152*', [-1405586.5344710157, 16.21688961392354]), ('55334*', [-1405548.6096405764, 9.504300614497462]), ('34252*', [-1405513.4460758825, 16.19868811945491]), ('15442*', [-1405507.066494224, 16.209072657030333]), ('51352*', [-1405506.634560419, 16.209071748004003]), ('41542*', [-1405501.4880398593, 16.209071748004003]), ('35152*', [-1405436.9019066866, 16.21688961392354]), ('25252*', [-1405363.0596903367, 16.19868811945491]), ('15352*', [-1405360.8475218387, 16.209069597115707]), ('5542*', [-1405358.9302555379, 15.209072657030335]), ('14542*', [-1405358.9302555379, 16.209072657030333]), ('41452*', [-1405357.751547448, 16.209071748004003]), ('53324*', [-1405230.736304566, 10.248799295525998]), ('5452*', [-1405212.711283153, 15.209069597115706]), ('14452*', [-1405212.711283153, 16.209069597115707]), ('31552*', [-1405208.1184893623, 16.209071748004003]), ('44324*', [-1405081.8537828766, 10.248799295525998]), ('54314*', [-1405076.2911393521, 10.264837240985697]), ('4552*', [-1405063.8287614633, 15.209069597115706]), ('13552*', [-1405063.828761463, 16.209069597115707]), ('43424*', [-1404933.8941480238, 10.248798412583756]), ('35324*', [-1404932.221218547, 10.248799295525998]), ('53414*', [-1404929.1422337333, 10.26483655524162]), ('45314*', [-1404927.4086176625, 10.264837240985697]), ('51512*', [-1404877.81332584, 17.30721484236283]), ('15512*', [-1404802.9548425365, 17.302645101108837]), ('51532*', [-1404802.8984710476, 16.312121320673867]), ('53234*', [-1404799.2994620963, 10.252855434969943]), ('34424*', [-1404784.2615836945, 10.248798412583756]), ('44414*', [-1404780.2597120437, 10.26483655524162]), ('52114*', [-1404729.888603788, 11.256557893728486]), ('54224*', [-1404727.0257235745, 10.265140578703047]), ('51124*', [-1404671.9601975577, 11.256557578695375]), ('51214*', [-1404667.9092324092, 11.268948137576677]), ('15214*', [-1404667.9092324092, 11.268948137576677]), ('15532*', [-1404662.539875327, 16.312092969513923]), ('5224*', [-1404658.8278547393, 10.256557578695377]), ('41224*', [-1404658.8278547393, 11.256557578695377]), ('14224*', [-1404658.8278547393, 11.256557578695377]), ('44234*', [-1404650.4169404067, 10.252855434969943]), ('33524*', [-1404635.5886404114, 10.248795734083544]), ('25424*', [-1404633.8751981496, 10.248798412583756]), ('43514*', [-1404632.3044201408, 10.264835070903667]), ('35414*', [-1404630.6271477146, 10.26483655524162]), ('43114*', [-1404581.0060820982, 11.256557893728486]), ('45224*', [-1404578.1432018848, 10.265140578703047]), ('54134*', [-1404528.2860526477, 10.264831888268116]), ('5314*', [-1404519.7729937236, 10.268948137576677]), ('41314*', [-1404519.7729937236, 11.268948137576677]), ('14314*', [-1404519.7729937232, 11.268948137576677]), ('4324*', [-1404509.9453330499, 10.256557578695377]), ('13324*', [-1404509.9453330499, 11.256557578695377]), ('31324*', [-1404509.9453330496, 11.256557578695377]), ('35234*', [-1404500.784376077, 10.252855434969943]), ('24524*', [-1404485.202254866, 10.248795734083544]), ('34514*', [-1404482.6718558113, 10.264835070903667]), ('34114*', [-1404431.3735177694, 11.256557893728486]), ('55442*', [-1404406.0615469331, 14.21152013751759]), ('45134*', [-1404379.403530958, 10.264831888268116]), ('52154*', [-1404374.6498488667, 10.268948137576675]), ('4414*', [-1404370.890472034, 10.268948137576677]), ('31414*', [-1404370.8904720338, 11.268948137576677]), ('13414*', [-1404370.8904720338, 11.268948137576677]), ('3424*', [-1404360.3127687208, 10.256557578695377]), ('12424*', [-1404360.3127687208, 11.256557578695377]), ('21424*', [-1404360.3127687206, 11.256557578695377]), ('25514*', [-1404332.285470266, 10.264835070903667]), ('15124*', [-1404321.2257567912, 11.268944961148303]), ('42254*', [-1404307.4337223384, 10.256557578695377]), ('25114*', [-1404280.9871322243, 11.256557893728486]), ('54542*', [-1404259.895059844, 14.21151981593748]), ('43154*', [-1404225.7673271769, 10.268948137576675]), ('3514*', [-1404221.2579077047, 10.268948137576677]), ('21514*', [-1404221.2579077045, 11.268948137576677]), ('12514*', [-1404221.2579077045, 11.268948137576677]), ('11524*', [-1404209.9263831757, 11.256557578695377]), ('2524*', [-1404209.9263831754, 10.256557578695377]), ('5134*', [-1404173.0890292863, 10.268944961148303]), ('14134*', [-1404173.089029286, 11.268944961148303]), ('33254*', [-1404157.8011580093, 10.256557578695377]), ('54452*', [-1404154.9326958684, 14.206389452191276]), ('45542*', [-1404111.0125381544, 14.21151981593748]), ('34154*', [-1404076.1347628476, 10.268948137576675]), ('51254*', [-1404028.6284189804, 10.268944961148302]), ('1534*', [-1404028.2920232925, 10.268943242061733]), ('414*', [-1404024.2060163158, 10.268944961148302]), ('1314*', [-1404024.2060163156, 11.268944961148302]), ('53552*', [-1404008.029652485, 14.2063890693482]), ('24254*', [-1404007.414772464, 10.256557578695377]), ('45452*', [-1404006.0501741788, 14.206389452191276]), ('25154*', [-1403925.7483773022, 10.268948137576675]), ('32335*', [-1403909.4795258222, 9.563954901300571]), ('14*', [-1403880.155784607, 10.26894324206173]), ('54*', [-1403880.155784607, 9.26894324206173]), ('41354*', [-1403879.74540601, 10.268944961148302]), ('12154*', [-1403874.5729582303, 11.268944961148303]), ('3154*', [-1403874.5729582298, 10.268944961148303]), ('44552*', [-1403859.1471307948, 14.2063890693482]), ('55553*', [-1403794.0724602323, 10.033705007005459]), ('5354*', [-1403733.2781375628, 9.268942257572757]), ('14354*', [-1403733.2781375628, 10.268942257572757]), ('454*', [-1403731.2732629173, 9.268943242061733]), ('1354*', [-1403731.273262917, 10.268943242061733]), ('31454*', [-1403730.1123479242, 10.268944961148302]), ('35552*', [-1403709.5145664657, 14.2063890693482]), ('53454*', [-1403686.0051601734, 9.264434059049426]), ('23345*', [-1403676.1619714054, 9.340177597919457]), ('4454*', [-1403584.3956158732, 9.268942257572757]), ('13454*', [-1403584.3956158732, 10.268942257572757]), ('21554*', [-1403579.7254661357, 10.268944961148302]), ('44454*', [-1403537.122638483, 9.264434059049426]), ('22445*', [-1403436.677346947, 9.34060482569283]), ('3554*', [-1403434.7630515439, 9.268942257572757]), ('12554*', [-1403434.7630515439, 10.268942257572757]), ('43554*', [-1403389.505419517, 9.264433020700213]), ('35454*', [-1403387.4900741538, 9.264434059049426]), ('34554*', [-1403239.8728551879, 9.264433020700213]), ('55253*', [-1403144.2203194553, 11.12041038100569]), ('25554*', [-1403089.4864696427, 9.264433020700213]), ('55512*', [-1402897.7812379291, 15.38359480608649]), ('52424*', [-1402698.7243180638, 10.339836394201303]), ('55532*', [-1402696.0677546964, 14.378659098082164]), ('52514*', [-1402410.3970232538, 10.35521086595188]), ('42524*', [-1402403.9884282502, 10.339836394201303]), ('55352*', [-1402120.9643242406, 14.396409570991995]), ('51434*', [-1402095.3898221483, 10.35813546876504]), ('55214*', [-1401959.210314115, 10.358090269315449]), ('55124*', [-1401954.8134641463, 10.358055039205992]), ('15434*', [-1401953.8283789689, 10.358111758033958]), ('41534*', [-1401946.5068091778, 10.35813546876504]), ('5534*', [-1401805.692140283, 9.358111758033958]), ('14534*', [-1401805.692140283, 10.358111758033958]), ('54354*', [-1401605.9710676044, 9.354003223902144]), ('15254*', [-1401570.2212422295, 10.358944961148302]), ('45354*', [-1401457.088545915, 9.354003223902144]), ('23435*', [-1401361.4220286617, 9.411994980247256]), ('52554*', [-1401233.046468983, 9.354843541793123]), ('32345*', [-1401118.7277027885, 9.422119606590757]), ('22125*', [-1400989.9293657187, 10.421853630008183]), ('22215*', [-1400965.479680384, 10.422344383713629]), ('21135*', [-1400804.7635330446, 10.42234292917528]), ('2235*', [-1400778.1667209705, 9.42234292917528]), ('11235*', [-1400778.1667209705, 10.42234292917528]), ('55434*', [-1400065.6883324592, 9.429453136259543]), ('54534*', [-1399919.730033424, 9.429434419461789]), ('45534*', [-1399770.8475117346, 9.429434419461789]), ('22135*', [-1399558.3156428046, 10.258957352402968]), ('33335*', [-1399260.832603795, 9.48718777837109]), ('23445*', [-1399201.0231825132, 9.25057868710398]), ('15142*', [-1399134.787369386, 18.20809383395393]), ('24335*', [-1399110.4462182499, 9.48718777837109]), ('5152*', [-1398986.6506418814, 17.20809383395393]), ('14152*', [-1398986.6506418814, 18.20809383395393]), ('1552*', [-1398913.6020785181, 17.20359117794156]), ('55232*', [-1398831.8462271863, 16.208946471387527]), ('54242*', [-1398773.9214959217, 16.19868811945491]), ('32435*', [-1398759.8464587363, 9.495549945695954]), ('55142*', [-1398684.6154783012, 16.21688961392354]), ('45242*', [-1398625.0389742323, 16.19868811945491]), ('53252*', [-1398617.1040516251, 16.19868811945491]), ('54152*', [-1398540.9435648918, 16.21688961392354]), ('44252*', [-1398468.2215299357, 16.19868811945491]), ('22535*', [-1398462.1334387225, 9.495549945695954]), ('51542*', [-1398456.3234343203, 16.209071748004003]), ('45152*', [-1398392.0610432024, 16.21688961392354]), ('35252*', [-1398318.5889656062, 16.19868811945491]), ('15542*', [-1398314.4802298588, 16.209072657030333]), ('51452*', [-1398313.3074301172, 16.209071748004003]), ('15452*', [-1398168.9941892691, 16.209069597115707]), ('41552*', [-1398164.4244171465, 16.209071748004003]), ('54324*', [-1398038.7926204428, 10.248799295525998]), ('14552*', [-1398020.857950583, 16.209069597115707]), ('5552*', [-1398020.8579505829, 15.209069597115706]), ('53424*', [-1397891.5746425656, 10.248798412583756]), ('45324*', [-1397889.9100987532, 10.248799295525998]), ('55314*', [-1397885.121621333, 10.264837240985697]), ('44424*', [-1397742.692120876, 10.248798412583756]), ('54414*', [-1397738.7103088584, 10.26483655524162]), ('51224*', [-1397617.887136376, 11.256557578695377]), ('15224*', [-1397617.887136376, 11.256557578695377]), ('54234*', [-1397609.5183822801, 10.252855434969943]), ('43524*', [-1397594.7644100718, 10.248795734083544]), ('35424*', [-1397593.0595565469, 10.248798412583756]), ('53514*', [-1397591.4966521615, 10.264835070903667]), ('45414*', [-1397589.8277871686, 10.26483655524162]), ('53114*', [-1397540.455450265, 11.256557893728486]), ('55224*', [-1397537.6069204186, 10.265140578703047]), ('51314*', [-1397479.5292965968, 11.268948137576677]), ('15314*', [-1397479.5292965968, 11.268948137576677]), ('41324*', [-1397469.7508976902, 11.256557578695377]), ('14324*', [-1397469.7508976902, 11.256557578695377]), ('5324*', [-1397469.75089769, 10.256557578695377]), ('45234*', [-1397460.63586059, 10.252855434969943]), ('34524*', [-1397445.1318457425, 10.248795734083544]), ('44514*', [-1397442.6141304718, 10.264835070903667]), ('44114*', [-1397391.5729285758, 11.256557893728486]), ('55134*', [-1397339.8634445914, 10.264831888268116]), ('41414*', [-1397331.3930579107, 11.268948137576677]), ('14414*', [-1397331.3930579107, 11.268948137576677]), ('5414*', [-1397331.3930579105, 10.268948137576677]), ('13424*', [-1397320.8683760008, 11.256557578695377]), ('4424*', [-1397320.8683760006, 10.256557578695377]), ('31424*', [-1397320.8683760006, 11.256557578695377]), ('25524*', [-1397294.745460197, 10.248795734083544]), ('35514*', [-1397292.9815661418, 10.264835070903667]), ('52254*', [-1397268.2543891643, 10.256557578695377]), ('35114*', [-1397241.9403642465, 11.256557893728486]), ('53154*', [-1397186.9973519444, 10.268948137576675]), ('13514*', [-1397182.510536221, 11.268948137576677]), ('4514*', [-1397182.5105362209, 10.268948137576677]), ('31514*', [-1397182.5105362209, 11.268948137576677]), ('21524*', [-1397171.2358116712, 11.256557578695377]), ('12524*', [-1397171.2358116712, 11.256557578695377]), ('3524*', [-1397171.235811671, 10.256557578695377]), ('22315*', [-1397144.804375248, 10.34087573964588]), ('15134*', [-1397134.5831073353, 11.268944961148303]), ('43254*', [-1397119.3718674749, 10.256557578695377]), ('22225*', [-1397113.9751960568, 10.259769940617986]), ('55542*', [-1397072.8177785191, 14.21151981593748]), ('44154*', [-1397038.1148302546, 10.268948137576675]), ('514*', [-1396986.446379831, 10.268944961148302]), ('1414*', [-1396986.446379831, 11.268944961148302]), ('34254*', [-1396969.7393031453, 10.256557578695377]), ('55452*', [-1396968.381544995, 14.206389452191276]), ('35154*', [-1396888.4822659255, 10.268948137576675]), ('42335*', [-1396872.2949630867, 9.563954901300571]), ('154*', [-1396843.1182089685, 10.26894324206173]), ('51354*', [-1396842.70988742, 10.268944961148302]), ('4154*', [-1396837.56336686, 10.268944961148303]), ('13154*', [-1396837.5633668597, 11.268944961148303]), ('54552*', [-1396822.2148623548, 14.2063890693482]), ('25254*', [-1396819.3529176002, 10.256557578695377]), ('15354*', [-1396696.9767953677, 10.268942257572757]), ('554*', [-1396694.9819702825, 9.268943242061733]), ('1454*', [-1396694.9819702825, 10.268943242061733]), ('41454*', [-1396693.826874449, 10.268944961148302]), ('45552*', [-1396673.3323406654, 14.2063890693482]), ('33345*', [-1396640.1469275851, 9.340177597919457]), ('21235*', [-1396591.7262252143, 10.340447045656763]), ('5454*', [-1396548.8405566819, 9.268942257572757]), ('14454*', [-1396548.8405566819, 10.268942257572757]), ('31554*', [-1396544.1938163638, 10.268944961148302]), ('54454*', [-1396501.8045380642, 9.264434059049426]), ('24345*', [-1396489.7605420405, 9.340177597919457]), ('23125*', [-1396465.29659502, 10.340336204163425]), ('23215*', [-1396452.40067803, 10.340401571548117]), ('32445*', [-1396401.8627348691, 9.34060482569283]), ('4554*', [-1396399.9580349922, 9.268942257572757]), ('13554*', [-1396399.958034992, 10.268942257572757]), ('53554*', [-1396354.9272596918, 9.264433020700213]), ('45454*', [-1396352.9220163748, 9.264434059049426]), ('11335*', [-1396295.2432072596, 10.34040004265401]), ('2335*', [-1396295.2432072593, 9.34040004265401]), ('44554*', [-1396206.0447380024, 9.264433020700213]), ('21145*', [-1396108.2815685817, 10.340851535667014]), ('22545*', [-1396104.1497148548, 9.34060482569283]), ('11245*', [-1396068.4862553722, 10.340851535667015]), ('2245*', [-1396068.486255372, 9.340851535667015]), ('35554*', [-1396056.4121736733, 9.264433020700213]), ('52524*', [-1395374.3502343264, 10.339836394201303]), ('51534*', [-1394919.161770642, 10.35813546876504]), ('15534*', [-1394779.0529441317, 10.358111758033958]), ('55354*', [-1394432.1967472052, 9.354003223902144]), ('33435*', [-1394337.0097643866, 9.411994980247256]), ('24435*', [-1394186.6233788417, 9.411994980247256]), ('42345*', [-1394095.5319590855, 9.422119606590757]), ('23535*', [-1394037.825315586, 9.411976226371543]), ('32125*', [-1393967.3792317812, 10.421853630008183]), ('32215*', [-1393943.0521020317, 10.422344383713629]), ('31135*', [-1393783.141554489, 10.42234292917528]), ('12235*', [-1393756.678060608, 10.42234292917528]), ('3235*', [-1393756.6780606078, 9.42234292917528]), ('23315*', [-1392976.5444781508, 10.250875209816899]), ('55534*', [-1392754.408102262, 9.429434419461789]), ('23225*', [-1392633.9434782807, 10.247145021479652]), ('22235*', [-1392613.5346624262, 10.251109172908855]), ('32135*', [-1392542.941562692, 10.258957352402968]), ('23135*', [-1392537.5364670428, 10.246759261051606]), ('22145*', [-1392503.4372383216, 10.250853235921484]), ('43335*', [-1392246.9496761253, 9.48718777837109]), ('33445*', [-1392187.440053329, 9.25057868710398]), ('21245*', [-1392148.8309643501, 10.250851535667014]), ('34335*', [-1392097.3171117958, 9.48718777837109]), ('24445*', [-1392037.053667784, 9.25057868710398]), ('15152*', [-1391974.14206854, 18.20809383395393]), ('25335*', [-1391946.9307262504, 9.48718777837109]), ('23545*', [-1391887.9518619953, 9.250577649081068]), ('2345*', [-1391849.9571809277, 9.250849926302044]), ('11345*', [-1391849.9571809277, 10.250849926302044]), ('42435*', [-1391748.4747556273, 9.495549945695954]), ('55242*', [-1391614.3430021177, 16.19868811945491]), ('54252*', [-1391458.3116151236, 16.19868811945491]), ('32535*', [-1391452.25404085, 9.495549945695954]), ('55152*', [-1391382.5328876197, 16.21688961392354]), ('45252*', [-1391309.4290934335, 16.19868811945491]), ('51552*', [-1391156.037304469, 16.209071748004003]), ('15552*', [-1391013.1904738485, 16.209069597115707]), ('55324*', [-1390882.8990063618, 10.248799295525998]), ('54424*', [-1390736.418967858, 10.248798412583756]), ('53524*', [-1390589.2327540077, 10.248795734083544]), ('45424*', [-1390587.5364461686, 10.248798412583756]), ('55414*', [-1390584.320876237, 10.26483655524162]), ('51324*', [-1390464.8458797194, 11.256557578695377]), ('15324*', [-1390464.8458797191, 11.256557578695377]), ('55234*', [-1390455.7765323159, 10.252855434969943]), ('22325*', [-1390440.503743723, 10.24391229778327]), ('44524*', [-1390440.350232318, 10.248795734083544]), ('54514*', [-1390437.8451372532, 10.264835070903667]), ('54114*', [-1390387.0597825919, 11.256557893728486]), ('51414*', [-1390327.1815673136, 11.268948137576677]), ('15414*', [-1390327.1815673134, 11.268948137576677]), ('5424*', [-1390316.7096410336, 10.256557578695377]), ('41424*', [-1390316.7096410336, 11.256557578695377]), ('14424*', [-1390316.7096410336, 11.256557578695377]), ('35524*', [-1390290.7176679883, 10.248795734083544]), ('45514*', [-1390288.9626155633, 10.264835070903667]), ('45114*', [-1390238.1772609025, 11.256557893728486]), ('41514*', [-1390179.0453286278, 11.268948137576677]), ('5514*', [-1390179.0453286273, 10.268948137576677]), ('14514*', [-1390179.0453286273, 11.268948137576677]), ('4524*', [-1390167.827119344, 10.256557578695377]), ('31524*', [-1390167.8271193437, 11.256557578695377]), ('13524*', [-1390167.8271193434, 11.256557578695377]), ('32315*', [-1390141.528172158, 10.34087573964588]), ('53254*', [-1390116.2231464295, 10.256557578695377]), ('32225*', [-1390110.8535261666, 10.259769940617986]), ('54154*', [-1390035.3734152191, 10.268948137576675]), ('1514*', [-1389983.9639561528, 11.268944961148302]), ('44254*', [-1389967.3406247396, 10.256557578695377]), ('45154*', [-1389886.4908935293, 10.268948137576675]), ('52335*', [-1389870.384730564, 9.563954901300571]), ('22415*', [-1389843.815152144, 10.34087573964588]), ('14154*', [-1389835.827228648, 11.268944961148303]), ('5154*', [-1389835.8272286477, 10.268944961148303]), ('35254*', [-1389817.7080604106, 10.256557578695377]), ('1554*', [-1389693.9605302883, 10.268943242061733]), ('51454*', [-1389692.8112244452, 10.268944961148302]), ('55552*', [-1389672.4194208013, 14.2063890693482]), ('43345*', [-1389639.4003516915, 9.340177597919457]), ('31235*', [-1389591.2223611362, 10.340447045656763]), ('15454*', [-1389548.5516597144, 10.268942257572757]), ('41554*', [-1389543.9282114746, 10.268944961148302]), ('34345*', [-1389489.7677873624, 9.340177597919457]), ('33125*', [-1389465.4264674147, 10.340336204163425]), ('33215*', [-1389452.5951920196, 10.340401571548117]), ('21335*', [-1389440.8354793473, 10.340447045656763]), ('42445*', [-1389402.3105734778, 9.34060482569283]), ('5554*', [-1389400.4154210286, 9.268942257572757]), ('14554*', [-1389400.4154210286, 10.268942257572757]), ('55454*', [-1389353.6151734125, 9.264434059049426]), ('25345*', [-1389339.3814018173, 9.340177597919457]), ('24125*', [-1389315.0400818698, 10.340336204163425]), ('24215*', [-1389302.2088064745, 10.340401571548117]), ('3335*', [-1389296.2254829558, 9.34040004265401]), ('12335*', [-1389296.2254829556, 10.34040004265401]), ('54554*', [-1389207.4741266358, 9.264433020700213]), ('2435*', [-1389145.8390974107, 9.34040004265401]), ('11435*', [-1389145.8390974107, 10.34040004265401]), ('31145*', [-1389110.2010012502, 10.340851535667014]), ('32545*', [-1389106.0898587003, 9.34060482569283]), ('3245*', [-1389070.6051645512, 9.340851535667015]), ('12245*', [-1389070.605164551, 10.340851535667015]), ('45554*', [-1389058.5916049462, 9.264433020700213]), ('2115*', [-1388825.8198005653, 10.340851535667014]), ('11115*', [-1388825.819800565, 11.340851535667014]), ('225*', [-1388772.892144537, 9.340851535667015]), ('1125*', [-1388772.892144537, 10.340851535667015]), ('43435*', [-1387347.8078083755, 9.411994980247256]), ('34435*', [-1387198.1752440464, 9.411994980247256]), ('52345*', [-1387107.5404257607, 9.422119606590757]), ('33535*', [-1387050.123040441, 9.411976226371543]), ('25435*', [-1387047.7888585008, 9.411994980247256]), ('42125*', [-1386980.0300720618, 10.421853630008183]), ('42215*', [-1386955.82488358, 10.422344383713629]), ('24535*', [-1386899.736654896, 9.411976226371543]), ('41135*', [-1386796.7158977143, 10.42234292917528]), ('4235*', [-1386770.3850537608, 9.42234292917528]), ('13235*', [-1386770.3850537606, 10.42234292917528]), ('23325*', [-1386022.2424782214, 10.234182114007266]), ('33315*', [-1385994.1619399511, 10.250875209816899]), ('24315*', [-1385843.7755544058, 10.250875209816899]), ('23415*', [-1385694.369478388, 10.250875447942791]), ('33225*', [-1385653.2782491397, 10.247145021479652]), ('23235*', [-1385642.0536047611, 10.238821861526453]), ('32235*', [-1385632.971733759, 10.251109172908855]), ('42135*', [-1385562.7324863765, 10.258957352402968]), ('33135*', [-1385557.3544841092, 10.246759261051606]), ('22245*', [-1385526.0347508253, 10.242190142492289]), ('32145*', [-1385523.4261799166, 10.250853235921484]), ('24225*', [-1385502.8918635945, 10.247145021479652]), ('24135*', [-1385406.9680985636, 10.246759261051606]), ('23145*', [-1385373.0397943717, 10.250853235921484]), ('53335*', [-1385268.2242777573, 9.48718777837109]), ('2215*', [-1385225.7122706838, 10.250853235921484]), ('21115*', [-1385225.7122706838, 11.25085323592148]), ('11215*', [-1385225.7122706838, 11.250853235921484]), ('43445*', [-1385209.0129506881, 9.25057868710398]), ('31245*', [-1385170.597392196, 10.250851535667014]), ('44335*', [-1385119.3417560675, 9.48718777837109]), ('34445*', [-1385059.380386359, 9.25057868710398]), ('21345*', [-1385020.210510407, 10.250851535667014]), ('35335*', [-1384969.709191738, 9.48718777837109]), ('33545*', [-1384911.025962749, 9.250577649081068]), ('25445*', [-1384908.994000814, 9.25057868710398]), ('11125*', [-1384873.55558498, 11.250851535667014]), ('2125*', [-1384873.5555849795, 10.250851535667014]), ('3345*', [-1384873.2217324101, 9.250849926302044]), ('12345*', [-1384873.2217324101, 10.250849926302044]), ('52435*', [-1384772.2479941489, 9.495549945695954]), ('24545*', [-1384760.6395772041, 9.250577649081068]), ('2445*', [-1384722.835346865, 9.250849926302044]), ('11445*', [-1384722.835346865, 10.250849926302044]), ('1135*', [-1384573.7339997417, 10.250848891530724]), ('235*', [-1384573.7339997415, 9.250848891530724]), ('42535*', [-1384477.512104335, 9.495549945695954]), ('55252*', [-1384335.40307595, 16.19868811945491]), ('55424*', [-1383617.1289609815, 10.248798412583756]), ('32325*', [-1383470.833269205, 10.24391229778327]), ('54524*', [-1383470.6805272847, 10.248795734083544]), ('51424*', [-1383347.6596922409, 11.256557578695377]), ('15424*', [-1383347.6596922406, 11.256557578695377]), ('45524*', [-1383321.798005595, 10.248795734083544]), ('55514*', [-1383320.0517504807, 10.264835070903667]), ('55114*', [-1383269.5209606038, 11.256557893728486]), ('51514*', [-1383210.685430859, 11.268948137576677]), ('15514*', [-1383210.6854308585, 11.268948137576677]), ('14524*', [-1383199.5234535548, 11.256557578695377]), ('5524*', [-1383199.5234535546, 10.256557578695377]), ('41524*', [-1383199.5234535546, 11.256557578695377]), ('42315*', [-1383173.3563314958, 10.34087573964588]), ('22425*', [-1383173.1202491915, 10.243481884374804]), ('42225*', [-1383142.8354440962, 10.259769940617986]), ('54254*', [-1383000.041910114, 10.256557578695377]), ('55154*', [-1382919.597443266, 10.268948137576675]), ('32415*', [-1382877.1356167188, 10.34087573964588]), ('15154*', [-1382869.1877331913, 11.268944961148303]), ('45254*', [-1382851.1593884244, 10.256557578695377]), ('53345*', [-1382673.7454583093, 9.340177597919457]), ('41235*', [-1382625.8089629624, 10.340447045656763]), ('22515*', [-1382579.4225967047, 10.34087573964588]), ('51554*', [-1382578.7518782, 10.268944961148302]), ('44345*', [-1382524.8629366201, 9.340177597919457]), ('43125*', [-1382500.64362907, 10.340336204163425]), ('43215*', [-1382487.8766712497, 10.340401571548117]), ('31335*', [-1382476.1759048763, 10.340447045656763]), ('52445*', [-1382437.8441075208, 9.34060482569283]), ('15554*', [-1382435.9584546424, 10.268942257572757]), ('35345*', [-1382375.2303722906, 9.340177597919457]), ('34125*', [-1382351.0110647408, 10.340336204163425]), ('34215*', [-1382338.2441069204, 10.340401571548117]), ('13335*', [-1382332.290775187, 10.34040004265401]), ('4335*', [-1382332.2907751868, 9.34040004265401]), ('21435*', [-1382325.7890230876, 10.340447045656763]), ('25125*', [-1382200.6246791957, 10.340336204163425]), ('25215*', [-1382187.8577213753, 10.340401571548117]), ('12435*', [-1382182.6582108578, 10.34040004265401]), ('3435*', [-1382182.6582108575, 9.34040004265401]), ('41145*', [-1382147.198752896, 10.340851535667014]), ('42545*', [-1382143.1082177064, 9.34060482569283]), ('4245*', [-1382107.8013928183, 9.340851535667015]), ('13245*', [-1382107.8013928183, 10.340851535667015]), ('55554*', [-1382095.8480519368, 9.264433020700213]), ('11535*', [-1382032.2718253126, 10.34040004265401]), ('2535*', [-1382032.2718253122, 9.34040004265401]), ('3115*', [-1381864.2430308647, 10.340851535667014]), ('12115*', [-1381864.2430308647, 11.340851535667014]), ('325*', [-1381811.5806780413, 9.340851535667015]), ('1225*', [-1381811.5806780413, 10.340851535667015]), ('23245*', [-1381352.2481199089, 9.431786888394813]), ('53435*', [-1380393.6396667426, 9.411994980247256]), ('44435*', [-1380244.7571450532, 9.411994980247256]), ('43535*', [-1380097.4470624304, 9.411976226371543]), ('35435*', [-1380095.1245807237, 9.411994980247256]), ('52125*', [-1380027.7054394633, 10.421853630008183]), ('52215*', [-1380003.6215810115, 10.422344383713629]), ('34535*', [-1379947.8144981011, 9.411976226371543]), ('51135*', [-1379845.3101389438, 10.42234292917528]), ('5235*', [-1379819.111280002, 9.42234292917528]), ('14235*', [-1379819.111280002, 10.42234292917528]), ('25535*', [-1379797.4281125562, 9.411976226371543]), ('33325*', [-1379074.7188161758, 10.234182114007266]), ('43315*', [-1379046.7790333694, 10.250875209816899]), ('24325*', [-1378924.3324306307, 10.234182114007266]), ('34315*', [-1378897.1464690398, 10.250875209816899]), ('23425*', [-1378774.883437445, 10.233793616694676]), ('33415*', [-1378748.4893003753, 10.250875447942791]), ('25315*', [-1378746.760083495, 10.250875209816899]), ('43225*', [-1378707.6040434977, 10.247145021479652]), ('33235*', [-1378696.435663355, 10.238821861526453]), ('42235*', [-1378687.399315802, 10.251109172908855]), ('52135*', [-1378617.512147065, 10.258957352402968]), ('43135*', [-1378612.1611023722, 10.246759261051606]), ('24415*', [-1378598.1029148302, 10.250875447942791]), ('32245*', [-1378580.9983612213, 10.242190142492289]), ('42145*', [-1378578.4028659386, 10.250853235921484]), ('34225*', [-1378557.9714791684, 10.247145021479652]), ('24235*', [-1378546.04927781, 10.238821861526453]), ('34135*', [-1378462.5285380427, 10.246759261051606]), ('23515*', [-1378448.6591944476, 10.250874071567042]), ('33145*', [-1378428.7703016095, 10.250853235921484]), ('25225*', [-1378407.5850936237, 10.247145021479652]), ('22115*', [-1378354.486177568, 11.241780679419708]), ('25135*', [-1378312.1421524978, 10.246759261051606]), ('21125*', [-1378283.9423213832, 11.241780511648393]), ('3215*', [-1378282.1812663998, 10.250853235921484]), ('31115*', [-1378282.1812663998, 11.25085323592148]), ('12215*', [-1378282.1812663998, 11.250853235921484]), ('21215*', [-1378282.1812663996, 11.250853235921484]), ('24145*', [-1378278.3839160644, 10.250853235921484]), ('2225*', [-1378270.6105020542, 10.241780511648392]), ('11225*', [-1378270.6105020542, 11.241780511648392]), ('53445*', [-1378265.5656527958, 9.25057868710398]), ('41245*', [-1378227.342654707, 10.250851535667014]), ('54335*', [-1378176.3439406778, 9.48718777837109]), ('11315*', [-1378131.7948808551, 11.250853235921484]), ('2315*', [-1378131.794880855, 10.250853235921484]), ('44445*', [-1378116.6831311064, 9.25057868710398]), ('31345*', [-1378077.7095966213, 9.431419270493203]), ('45335*', [-1378027.4614189882, 9.48718777837109]), ('43545*', [-1377969.072343375, 9.250577649081068]), ('35445*', [-1377967.0505667771, 9.25057868710398]), ('3125*', [-1377931.789788231, 10.250851535667014]), ('12125*', [-1377931.789788231, 11.250851535667014]), ('4345*', [-1377931.4576091187, 9.250849926302044]), ('13345*', [-1377931.4576091187, 10.250849926302044]), ('21445*', [-1377927.3227148324, 10.250851535667014]), ('34545*', [-1377819.439779046, 9.250577649081068]), ('3445*', [-1377781.8250447894, 9.250849926302044]), ('12445*', [-1377781.8250447894, 10.250849926302044]), ('11135*', [-1377781.4029064423, 11.250851535667014]), ('2135*', [-1377781.402906442, 10.250851535667014]), ('25545*', [-1377669.0533935009, 9.250577649081068]), ('335*', [-1377633.4710775458, 9.250848891530724]), ('1235*', [-1377633.4710775458, 10.250848891530724]), ('11545*', [-1377631.4386592444, 10.250849926302044]), ('2545*', [-1377631.4386592442, 9.250849926302044]), ('52535*', [-1377537.731500442, 9.495549945695954]), ('1145*', [-1377483.0846920009, 10.250848891530724]), ('245*', [-1377483.0846920004, 9.250848891530724]), ('42325*', [-1376536.0987062864, 10.24391229778327]), ('55524*', [-1376387.8104913086, 10.248795734083544]), ('51524*', [-1376266.1488481504, 11.256557578695377]), ('15524*', [-1376266.1488481504, 11.256557578695377]), ('52315*', [-1376240.1128904365, 10.34087573964588]), ('32425*', [-1376239.8779915099, 10.243481884374804]), ('52225*', [-1376209.744990905, 10.259769940617986]), ('42415*', [-1375945.3770006227, 10.34087573964588]), ('22525*', [-1375942.1649714957, 10.243481884374804]), ('55254*', [-1375919.5309798063, 10.256557578695377]), ('51235*', [-1375695.310137525, 10.340447045656763]), ('31435*', [-1375652.5392357504, 9.504513884865922]), ('32515*', [-1375649.1562858457, 10.34087573964588]), ('54345*', [-1375594.8701094887, 9.340177597919457]), ('53125*', [-1375570.7722027407, 10.340336204163425]), ('53215*', [-1375558.0692400998, 10.340401571548117]), ('41335*', [-1375546.4271245538, 10.340447045656763]), ('45345*', [-1375445.9875877989, 9.340177597919457]), ('44125*', [-1375421.8896810513, 10.340336204163425]), ('44215*', [-1375409.1867184099, 10.340401571548117]), ('14335*', [-1375403.263228126, 10.34040004265401]), ('5335*', [-1375403.2632281259, 9.34040004265401]), ('35125*', [-1375272.2571167217, 10.340336204163425]), ('35215*', [-1375259.5541540806, 10.340401571548117]), ('4435*', [-1375254.3807064365, 9.34040004265401]), ('13435*', [-1375254.3807064365, 10.34040004265401]), ('21535*', [-1375246.407184679, 10.340447045656763]), ('51145*', [-1375219.0989912378, 10.340851535667014]), ('52545*', [-1375215.0289601134, 9.34060482569283]), ('5245*', [-1375179.8991129056, 9.340851535667015]), ('14245*', [-1375179.8991129056, 10.340851535667015]), ('3535*', [-1375104.748142107, 9.34040004265401]), ('12535*', [-1375104.748142107, 10.34040004265401]), ('13115*', [-1374937.5616025596, 11.340851535667014]), ('4115*', [-1374937.5616025594, 10.340851535667014]), ('425*', [-1374885.1632230917, 9.340851535667015]), ('1325*', [-1374885.1632230917, 10.340851535667015]), ('33245*', [-1374428.1330982957, 9.431786888394813]), ('24245*', [-1374277.7467127503, 9.345705823466957]), ('23115*', [-1374036.1143705375, 10.432004322872597]), ('21225*', [-1374019.178228843, 10.427055791651608]), ('11325*', [-1373956.1871161733, 10.43200415510128]), ('2325*', [-1373956.1871161729, 9.43200415510128]), ('54435*', [-1373326.1934916058, 9.411994980247256]), ('31445*', [-1373259.6644995592, 9.35035484750079]), ('53535*', [-1373179.6218100372, 9.411976226371543]), ('45435*', [-1373177.310969916, 9.411994980247256]), ('44535*', [-1373030.739288347, 9.411976226371543]), ('15235*', [-1372902.681203223, 10.42234292917528]), ('35535*', [-1372881.106724018, 9.411976226371543]), ('43325*', [-1372162.0200534393, 10.234182114007266]), ('53315*', [-1372134.2203205514, 10.250875209816899]), ('34325*', [-1372012.3874891098, 10.234182114007266]), ('44315*', [-1371985.3377988616, 10.250875209816899]), ('33425*', [-1371863.6876184023, 10.233793616694676]), ('25325*', [-1371862.0011035649, 10.234182114007266]), ('43415*', [-1371837.4257836044, 10.250875447942791]), ('35315*', [-1371835.7052345325, 10.250875209816899]), ('53225*', [-1371796.7454666484, 10.247145021479652]), ('43235*', [-1371785.6330687136, 10.238821861526453]), ('52235*', [-1371776.642016421, 10.251109172908855]), ('24425*', [-1371713.3012328572, 10.233793616694676]), ('53135*', [-1371701.7809392682, 10.246759261051606]), ('34415*', [-1371687.793219275, 10.250875447942791]), ('42245*', [-1371670.774403317, 10.242190142492289]), ('52145*', [-1371668.1919181177, 10.250853235921484]), ('44225*', [-1371647.8629449587, 10.247145021479652]), ('34235*', [-1371636.000504384, 10.238821861526453]), ('23525*', [-1371563.9816085487, 10.233768152961899]), ('44135*', [-1371552.8984175783, 10.246759261051606]), ('33515*', [-1371539.09859494, 10.250874071567042]), ('25415*', [-1371537.40683373, 10.250875447942791]), ('43145*', [-1371519.3093964283, 10.250853235921484]), ('35225*', [-1371498.23038063, 10.247145021479652]), ('25235*', [-1371485.6141188394, 10.238821861526453]), ('32115*', [-1371445.3976262307, 11.241780679419708]), ('35135*', [-1371403.2658532492, 10.246759261051606]), ('24515*', [-1371388.7122093947, 10.250874071567042]), ('31125*', [-1371375.2073755614, 10.427459511259777]), ('4215*', [-1371373.455147977, 10.250853235921484]), ('41115*', [-1371373.455147977, 11.25085323592148]), ('13215*', [-1371373.455147977, 11.250853235921484]), ('31215*', [-1371373.4551479768, 11.250853235921484]), ('34145*', [-1371369.676832099, 10.250853235921484]), ('3225*', [-1371361.9423828153, 10.241780511648392]), ('12225*', [-1371361.9423828153, 11.241780511648392]), ('51245*', [-1371318.891418274, 10.250851535667014]), ('3315*', [-1371223.822583648, 10.250853235921484]), ('21315*', [-1371223.822583648, 11.250853235921484]), ('12315*', [-1371223.822583648, 11.250853235921484]), ('25145*', [-1371219.2904465538, 10.250853235921484]), ('54445*', [-1371208.7865824956, 9.25057868710398]), ('41345*', [-1371170.0084053031, 9.431419270493203]), ('55335*', [-1371120.0120998207, 9.48718777837109]), ('11415*', [-1371073.436198103, 11.250853235921484]), ('2415*', [-1371073.4361981025, 10.250853235921484]), ('53545*', [-1371061.9157031216, 9.250577649081068]), ('45445*', [-1371059.904060806, 9.25057868710398]), ('4125*', [-1371024.8200291297, 10.250851535667014]), ('13125*', [-1371024.8200291297, 11.250851535667014]), ('14345*', [-1371024.4895150862, 10.250849926302044]), ('5345*', [-1371024.489515086, 9.250849926302044]), ('44545*', [-1370913.0331814322, 9.250577649081068]), ('4445*', [-1370875.6069933965, 9.250849926302044]), ('13445*', [-1370875.6069933965, 10.250849926302044]), ('3135*', [-1370875.1869710444, 9.431641312657998]), ('12135*', [-1370875.1869710442, 11.250851535667014]), ('21545*', [-1370869.9884654286, 10.250851535667014]), ('35545*', [-1370763.4006171029, 9.250577649081068]), ('435*', [-1370727.996659744, 9.250848891530724]), ('1335*', [-1370727.996659744, 10.250848891530724]), ('12545*', [-1370725.9744290672, 10.250849926302044]), ('3545*', [-1370725.974429067, 9.250849926302044]), ('11145*', [-1370724.8000892554, 11.250851535667014]), ('2145*', [-1370724.8000892552, 10.250851535667014]), ('345*', [-1370578.3640954148, 9.250848891530724]), ('1245*', [-1370578.3640954145, 10.250848891530724]), ('25*', [-1370427.9777098696, 9.250848891530724]), ('115*', [-1370427.9777098696, 10.250848891530724]), ('52325*', [-1369636.1249365127, 10.24391229778327]), ('42425*', [-1369341.3890467, 10.243481884374804]), ('52415*', [-1369048.3642605513, 10.34087573964588]), ('32525*', [-1369045.1683319227, 10.243481884374804]), ('41435*', [-1368756.9943633934, 9.504513884865922]), ('42515*', [-1368753.628370738, 10.34087573964588]), ('51335*', [-1368651.4141458278, 10.340447045656763]), ('55345*', [-1368551.478068568, 9.340177597919457]), ('54125*', [-1368527.5009540934, 10.340336204163425]), ('54215*', [-1368514.8616658512, 10.340401571548117]), ('15335*', [-1368508.9678674347, 10.34040004265401]), ('45125*', [-1368378.6184324038, 10.340336204163425]), ('45215*', [-1368365.9791441616, 10.340401571548117]), ('5435*', [-1368360.8316287491, 9.34040004265401]), ('14435*', [-1368360.8316287491, 10.34040004265401]), ('31535*', [-1368352.8980747713, 9.504513884865922]), ('15245*', [-1368286.7233788902, 10.340851535667015]), ('4535*', [-1368211.9491070593, 9.34040004265401]), ('13535*', [-1368211.949107059, 10.34040004265401]), ('5115*', [-1368045.600600556, 10.340851535667014]), ('14115*', [-1368045.600600556, 11.340851535667014]), ('525*', [-1367993.464871261, 9.340851535667015]), ('1425*', [-1367993.464871261, 10.340851535667015]), ('43245*', [-1367538.7256387097, 9.431786888394813]), ('34245*', [-1367389.0930743802, 9.345705823466957]), ('21325*', [-1367254.5211477522, 10.346078155996896]), ('25245*', [-1367238.7066888355, 9.345705823466957]), ('33115*', [-1367148.6719294793, 10.432004322872597]), ('31225*', [-1367131.8206812607, 10.337665628922643]), ('3325*', [-1367069.1453155046, 9.43200415510128]), ('12325*', [-1367069.1453155044, 10.43200415510128]), ('24115*', [-1366998.2855439342, 10.432004322872597]), ('31315*', [-1366973.7850591931, 10.346573212342694]), ('2425*', [-1366918.7589299595, 9.34592147340805]), ('11425*', [-1366918.7589299595, 10.34592147340805]), ('41445*', [-1366376.1140623125, 9.35035484750079]), ('55435*', [-1366294.1733349168, 9.411994980247256]), ('54535*', [-1366148.3363531204, 9.411976226371543]), ('45535*', [-1365999.4538314308, 9.411976226371543]), ('31545*', [-1365942.8598449328, 9.35035484750079]), ('12145*', [-1365894.060930099, 10.350600147843148]), ('3145*', [-1365894.0609300989, 9.350600147843148]), ('53325*', [-1365283.9716280142, 10.234182114007266]), ('44325*', [-1365135.089106324, 10.234182114007266]), ('54315*', [-1365108.1750043498, 10.250875209816899]), ('43425*', [-1364987.134603071, 10.233793616694676]), ('35325*', [-1364985.4565419948, 10.234182114007266]), ('53415*', [-1364961.0044073714, 10.250875447942791]), ('45315*', [-1364959.2924826602, 10.250875209816899]), ('53235*', [-1364909.4713067214, 10.238821861526453]), ('34425*', [-1364837.5020387417, 10.233793616694676]), ('44415*', [-1364812.121885682, 10.250875447942791]), ('52245*', [-1364795.188377609, 10.242190142492289]), ('54225*', [-1364772.3917643768, 10.247145021479652]), ('44235*', [-1364760.5887850318, 10.238821861526453]), ('33525*', [-1364688.9308884412, 10.233768152961899]), ('25425*', [-1364687.1156531966, 10.233793616694676]), ('54135*', [-1364677.9032526624, 10.246759261051606]), ('43515*', [-1364664.172602503, 10.250874071567042]), ('35415*', [-1364662.4893213524, 10.250875447942791]), ('53145*', [-1364644.4825985937, 10.250853235921484]), ('45225*', [-1364623.5092426874, 10.247145021479652]), ('35235*', [-1364610.9562207027, 10.238821861526453]), ('42115*', [-1364570.9413157923, 11.241780679419708]), ('24525*', [-1364538.544502896, 10.233768152961899]), ('45135*', [-1364529.020730973, 10.246759261051606]), ('34515*', [-1364514.5400381738, 10.250874071567042]), ('41125*', [-1364501.1028981695, 10.427459511259777]), ('5215*', [-1364499.3594537359, 10.250853235921484]), ('51115*', [-1364499.3594537359, 11.25085323592148]), ('41215*', [-1364499.3594537359, 10.427591753404544]), ('14215*', [-1364499.3594537359, 11.250853235921484]), ('44145*', [-1364495.6000769038, 10.250853235921484]), ('4225*', [-1364487.9043970339, 10.241780511648392]), ('13225*', [-1364487.9043970336, 11.241780511648392]), ('25515*', [-1364364.153652629, 10.250874071567042]), ('4315*', [-1364350.4769320467, 10.250853235921484]), ('13315*', [-1364350.4769320465, 11.250853235921484]), ('35145*', [-1364345.9675125747, 10.250853235921484]), ('51345*', [-1364296.932500655, 9.431419270493203]), ('3415*', [-1364200.8443677172, 10.250853235921484]), ('21415*', [-1364200.8443677172, 11.250853235921484]), ('12415*', [-1364200.844367717, 11.250853235921484]), ('55445*', [-1364187.380061109, 9.25057868710398]), ('5125*', [-1364152.471890349, 10.250851535667014]), ('14125*', [-1364152.4718903487, 11.250851535667014]), ('15345*', [-1364152.1430330274, 10.250849926302044]), ('2515*', [-1364050.457982172, 10.250853235921484]), ('11515*', [-1364050.457982172, 11.250853235921484]), ('54545*', [-1364041.2453812556, 9.250577649081068]), ('14445*', [-1364004.0067943416, 10.250849926302044]), ('5445*', [-1364004.0067943411, 9.250849926302044]), ('13135*', [-1364003.588877378, 10.431641312657998]), ('4135*', [-1364003.5888773778, 9.431641312657998]), ('45545*', [-1363892.362859566, 9.250577649081068]), ('535*', [-1363857.1363667706, 9.250848891530724]), ('1435*', [-1363857.1363667704, 10.250848891530724]), ('13545*', [-1363855.1242726517, 10.250849926302044]), ('4545*', [-1363855.1242726515, 9.250849926302044]), ('445*', [-1363708.2538450805, 9.250848891530724]), ('1345*', [-1363708.2538450805, 10.250848891530724]), ('1115*', [-1363703.5689375033, 11.250851535667014]), ('215*', [-1363703.568937503, 10.250851535667014]), ('35*', [-1363558.6212807517, 9.250848891530724]), ('125*', [-1363558.6212807512, 10.250848891530724]), ('52425*', [-1362477.4792115947, 10.243481884374804]), ('42525*', [-1362182.743321781, 10.243481884374804]), ('51435*', [-1361896.0138433932, 9.504513884865922]), ('52515*', [-1361892.6647229872, 10.34087573964588]), ('55125*', [-1361519.5345455548, 10.340336204163425]), ('55215*', [-1361506.9586125398, 10.340401571548117]), ('15435*', [-1361501.8368993723, 10.34040004265401]), ('41535*', [-1361493.9431128325, 9.504513884865922]), ('14535*', [-1361353.700660686, 10.34040004265401]), ('5535*', [-1361353.7006606858, 9.34040004265401]), ('15115*', [-1361188.1859865338, 11.340851535667014]), ('1525*', [-1361136.311590861, 10.340851535667015]), ('53245*', [-1360683.851767313, 9.431786888394813]), ('44245*', [-1360534.9692456229, 9.345705823466957]), ('31325*', [-1360401.0718692408, 10.253042237555192]), ('35245*', [-1360385.336681294, 9.345705823466957]), ('43115*', [-1360295.753226832, 10.432004322872597]), ('41225*', [-1360278.986446555, 10.337665628922643]), ('21425*', [-1360250.684987452, 10.346078155996896]), ('4325*', [-1360216.6252450114, 9.43200415510128]), ('13325*', [-1360216.6252450114, 10.43200415510128]), ('34115*', [-1360146.1206625027, 10.432004322872597]), ('41315*', [-1360121.7429879815, 10.346573212342694]), ('3425*', [-1360066.992680682, 9.34592147340805]), ('12425*', [-1360066.992680682, 10.34592147340805]), ('25115*', [-1359995.7342769578, 10.345919328607959]), ('11525*', [-1359916.6062951372, 10.34592147340805]), ('2525*', [-1359916.606295137, 9.34592147340805]), ('31415*', [-1359688.4887706023, 10.346573212342694]), ('51445*', [-1359527.067854563, 9.35035484750079]), ('55535*', [-1359152.295656778, 9.411976226371543]), ('41545*', [-1359095.9853511967, 9.35035484750079]), ('4145*', [-1359047.4310439925, 9.350600147843148]), ('13145*', [-1359047.4310439925, 10.350600147843148]), ('315*', [-1358614.1768266128, 9.350600147843148]), ('1215*', [-1358614.1768266128, 10.350600147843148]), ('54325*', [-1358292.263614219, 10.234182114007266]), ('53425*', [-1358145.0507422183, 10.233793616694676]), ('45325*', [-1358143.3810925293, 10.234182114007266]), ('55315*', [-1358117.3481821872, 10.250875209816899]), ('44425*', [-1357996.1682205286, 10.233793616694676]), ('54415*', [-1357970.9152870825, 10.250875447942791]), ('54235*', [-1357919.6404993401, 10.238821861526453]), ('43525*', [-1357848.3417924633, 10.233768152961899]), ('35425*', [-1357846.5356561993, 10.233793616694676]), ('53515*', [-1357823.7076089906, 10.250874071567042]), ('45415*', [-1357822.0327653927, 10.250875447942791]), ('55225*', [-1357783.2480768233, 10.247145021479652]), ('45235*', [-1357770.757977651, 10.238821861526453]), ('52115*', [-1357730.9436499681, 11.241780679419708]), ('34525*', [-1357698.709228134, 10.233768152961899]), ('55135*', [-1357689.2331947167, 10.246759261051606]), ('44515*', [-1357674.825087301, 10.250874071567042]), ('51125*', [-1357661.4553018059, 10.427459511259777]), ('51215*', [-1357659.7205964972, 10.427591753404544]), ('15215*', [-1357659.7205964972, 11.250853235921484]), ('54145*', [-1357655.980063778, 10.250853235921484]), ('5225*', [-1357648.3229589874, 10.241780511648392]), ('14225*', [-1357648.3229589874, 11.241780511648392]), ('25525*', [-1357548.3228425889, 10.233768152961899]), ('35515*', [-1357525.1925229717, 10.250874071567042]), ('14315*', [-1357511.5843578118, 11.250853235921484]), ('5315*', [-1357511.5843578114, 10.250853235921484]), ('45145*', [-1357507.0975420885, 10.250853235921484]), ('4415*', [-1357362.7018361217, 10.250853235921484]), ('13415*', [-1357362.7018361217, 11.250853235921484]), ('15125*', [-1357314.5718288387, 11.250851535667014]), ('12515*', [-1357213.0692717927, 11.250853235921484]), ('3515*', [-1357213.0692717924, 10.250853235921484]), ('21515*', [-1357213.0692717922, 11.250853235921484]), ('15445*', [-1357166.850923462, 10.250849926302044]), ('5135*', [-1357166.4351013338, 9.431641312657998]), ('14135*', [-1357166.4351013338, 10.431641312657998]), ('55545*', [-1357055.7666109316, 9.250577649081068]), ('1535*', [-1357020.716693147, 10.250848891530724]), ('5545*', [-1357018.7146847763, 9.250849926302044]), ('14545*', [-1357018.7146847763, 10.250849926302044]), ('545*', [-1356872.5804544606, 9.250848891530724]), ('1445*', [-1356872.5804544606, 10.250848891530724]), ('135*', [-1356723.6979327714, 10.250848891530724]), ('45*', [-1356723.697932771, 9.250848891530724]), ('52525*', [-1355354.7166486038, 10.243481884374804]), ('51535*', [-1354669.3690940232, 9.504513884865922]), ('15535*', [-1354529.8296159885, 10.34040004265401]), ('54245*', [-1353715.2021436363, 9.345705823466957]), ('41325*', [-1353581.9759362747, 10.253042237555192]), ('45245*', [-1353566.3196219471, 9.345705823466957]), ('53115*', [-1353477.1852101847, 10.432004322872597]), ('51225*', [-1353460.5024744482, 10.337665628922643]), ('31425*', [-1353432.3428781892, 10.248469739381108]), ('5325*', [-1353398.4538623497, 9.43200415510128]), ('14325*', [-1353398.4538623497, 10.43200415510128]), ('44115*', [-1353328.302688495, 10.432004322872597]), ('51315*', [-1353304.047208601, 10.346573212342694]), ('21525*', [-1353281.9559964002, 10.346078155996896]), ('4425*', [-1353249.5713406599, 9.34592147340805]), ('13425*', [-1353249.5713406599, 10.34592147340805]), ('35115*', [-1353178.670124166, 10.345919328607959]), ('3525*', [-1353099.9387763308, 9.34592147340805]), ('12525*', [-1353099.9387763306, 10.34592147340805]), ('41415*', [-1352872.964705235, 10.346573212342694]), ('31515*', [-1352439.7104878551, 10.346573212342694]), ('51545*', [-1352283.4312464837, 9.35035484750079]), ('14145*', [-1352235.120320798, 10.350600147843148]), ('5145*', [-1352235.1203207977, 9.350600147843148]), ('1315*', [-1351804.0378174316, 10.350600147843148]), ('415*', [-1351804.0378174314, 9.350600147843148]), ('55325*', [-1351335.601976575, 10.234182114007266]), ('54425*', [-1351189.1270183544, 10.233793616694676]), ('53525*', [-1351042.0415795567, 10.233768152961899]), ('45425*', [-1351040.2444966647, 10.233793616694676]), ('55415*', [-1351015.864428139, 10.250875447942791]), ('55235*', [-1350964.8466584955, 10.238821861526453]), ('44525*', [-1350893.159057867, 10.233768152961899]), ('54515*', [-1350869.3946377921, 10.250874071567042]), ('15225*', [-1350843.0253530636, 11.241780511648392]), ('35525*', [-1350743.526493538, 10.233768152961899]), ('45515*', [-1350720.5121161027, 10.250874071567042]), ('15315*', [-1350706.9721627259, 11.250853235921484]), ('55145*', [-1350702.507837449, 10.250853235921484]), ('5415*', [-1350558.8359240398, 10.250853235921484]), ('14415*', [-1350558.8359240398, 11.250853235921484]), ('4515*', [-1350409.9534023502, 10.250853235921484]), ('13515*', [-1350409.9534023502, 11.250853235921484]), ('15135*', [-1350363.552988604, 10.431641312657998]), ('15545*', [-1350216.573029925, 10.250849926302044]), ('1545*', [-1350071.1713066297, 10.250848891530724]), ('5*', [-1349923.035067944, 9.250848891530723]), ('145*', [-1349923.0350679439, 10.250848891530724]), ('51325*', [-1346797.0611505483, 10.253042237555192]), ('55245*', [-1346781.4833144809, 9.345705823466957]), ('41425*', [-1346648.1781375776, 10.248469739381108]), ('15325*', [-1346614.4589925613, 10.43200415510128]), ('54115*', [-1346544.6594558775, 10.432004322872597]), ('31525*', [-1346498.5450794916, 10.248469739381108]), ('5425*', [-1346466.3227538753, 9.34592147340805]), ('14425*', [-1346466.3227538753, 10.34592147340805]), ('45115*', [-1346395.7769341883, 10.345919328607959]), ('4525*', [-1346317.4402321856, 9.34592147340805]), ('13525*', [-1346317.4402321856, 10.34592147340805]), ('51415*', [-1346091.6038828965, 10.346573212342694]), ('41515*', [-1345660.5213795295, 10.346573212342694]), ('15145*', [-1345456.9567335523, 10.350600147843148]), ('515*', [-1345028.0350583463, 9.350600147843148]), ('1415*', [-1345028.0350583463, 10.350600147843148]), ('55425*', [-1344268.0702996373, 10.233793616694676]), ('54525*', [-1344121.7221358528, 10.233768152961899]), ('45525*', [-1343972.8396141627, 10.233768152961899]), ('55515*', [-1343949.940597742, 10.250874071567042]), ('15415*', [-1343789.0748177604, 11.250853235921484]), ('14515*', [-1343640.9385790746, 11.250853235921484]), ('5515*', [-1343640.9385790743, 10.250853235921484]), ('15*', [-1343156.4609534435, 10.250848891530723]), ('51425*', [-1339898.0194494075, 10.248469739381108]), ('41525*', [-1339749.136436436, 10.248469739381108]), ('15425*', [-1339717.0756272534, 10.34592147340805]), ('55115*', [-1339646.8834229244, 10.345919328607959]), ('5525*', [-1339568.9393885673, 9.34592147340805]), ('14525*', [-1339568.9393885673, 10.34592147340805]), ('51515*', [-1338915.3133829765, 10.346573212342694]), ('1515*', [-1338285.997439257, 10.350600147843148]), ('55525*', [-1337236.0912286025, 10.233768152961899]), ('15515*', [-1336905.8538683266, 11.250853235921484]), ('51525*', [-1333033.559628697, 10.248469739381108]), ('15525*', [-1332854.2658298637, 10.34592147340805])]

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

    # distributions.append(uniform(mdp))
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
    distributions.append(initialDistribution)
    # initialDistribution = initialDistributionCombo

    # margins = [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
    # margins = np.arange(0, 0.1001, 0.005)
    # margins = np.arange(0, 0.0501, 0.005)
    # margins = np.arange(0.055, 0.1001, 0.005)
    # margins = np.arange(0.01, 0.0251, 0.005)
    margins = [0.04]
    # margins = [0.015]
    repeats = 1
    results = []
    for margin in margins:
        print("\n\n  Running margin",margin,"\n\n")

        running_time_avg = 0
        error = -1
        trimmed = 0
        
        for i in range(repeats):
            running_time, error, trimmed = runChains(
                grid, mdp, discount, discount_checkin, start_state, target_state,
                checkin_periods=[1, 2, 3, 4],
                chain_length=4,
                do_filter = True,
                margin = margin,
                distName = 'uniform',
                startName = '',
                distributions = distributions, 
                initialDistribution = initialDistribution,
                bounding_box = bounding_box, 
                TRUTH = TRUTH_C4L4, 
                TRUTH_COSTS = TRUTH_COSTS_C4L4,
                drawIntermediate=True)

            running_time_avg += running_time
        running_time_avg /= repeats

        quality = (1 - error) * 100
        results.append((margin, running_time_avg, quality, trimmed))
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