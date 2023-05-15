
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

from schedule import Schedule, ScheduleBounds

import colorsys
import math

import os

import time
import math

from lp import linearProgrammingSolve
from figure import drawParetoFront, loadDataChains, loadTruth

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


def run(grid, mdp, discount, start_state, checkin_period, doBranchAndBound, 
        drawPolicy=True, drawIterations=True, outputPrefix="", doLinearProg=False, 
        bnbGreedy=-1, doSimilarityCluster=False, simClusterParams=None, outputDir="output"):
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
            draw(grid, compMDP, values, policy, True, False, f"{outputDir}/policy-"+outputPrefix+str(checkin_period)+("-vi" if not doLinearProg else "-lp"))
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
                drawBNBIteration(grid, compMDPs[i], ratios, upperBounds, lowerBounds, pruned, i, f"{outputDir}/policy-"+outputPrefix+str(checkin_period)+"-"+suffix+"-"+str(i+1))

        if drawPolicy:
            draw(grid, compMDP, values, policy, True, False, f"{outputDir}/policy-"+outputPrefix+str(checkin_period)+"-"+suffix+"-f")

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

def visualizeActionSimilarity(mdp, diffs, state, midfix="", outputDir="output"):
    print("State:", state)
    cost_diffs, transition_diffs, total_diffs = diffs[state]

    short_names = []
    for action in mdp.actions:
        short_name = ""
        for a in action:
            short_name += ("0" if a == "NO-OP" else a[0])
        short_names.append(short_name)

    makeTable(short_names, cost_diffs)
    plt.savefig(f'{outputDir}/diff{midfix}-cost.png', bbox_inches='tight')

    makeTable(short_names, transition_diffs)
    plt.savefig(f'{outputDir}/diff{midfix}-transition.png', bbox_inches='tight')

    makeTable(short_names, total_diffs)
    plt.savefig(f'{outputDir}/diff{midfix}-total.png', bbox_inches='tight')

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
        

def runTwoCadence(checkin1, checkin2, outputDir):

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

    draw(grid, bestBlendedMDP, bestValues, bestPolicy, True, False, f"{outputDir}/policy-comp-"+str(checkin1)+"v"+str(checkin2)+"-lp")

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

def policyFromValues(mdp, values, discount, restricted_action_set = None):
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


def getStartParetoValues(mdp, chains, initialDistribution, is_lower_bound):
    dists = [initialDistribution]

    costs = []
    indices = []
    for chain in chains:
        name = ""
        for checkin in chain[0]:
            name += str(checkin)
        name += "*"

        points = chainPoints(chain, is_lower_bound)
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

def chainPoints(chain, is_lower_bound):
    execution_pi_star = chain[1][0][0]
    checkins_pi_star = chain[1][0][1]
    
    checkins_pi_c = chain[1][1][0]
    execution_pi_c = chain[1][1][1]
    
    # return [[values, hitting_checkins]]
    if is_lower_bound:
        return [[execution_pi_star, checkins_pi_star], [execution_pi_star, checkins_pi_c], [execution_pi_c, checkins_pi_c]]
    else:
        points = [[execution_pi_star, checkins_pi_star], [execution_pi_c, checkins_pi_c]]
        for i in range(len(chain[1][2])):
            execution_pi_midpoint = chain[1][2][i][0]
            checkins_pi_midpoint = chain[1][2][i][1]
            points.append([execution_pi_midpoint, checkins_pi_midpoint])

        return points
        # return [[execution_pi_star, checkins_pi_star], [execution_pi_c, checkins_pi_c]]
    # return [[values, hitting_checkins], [values, hitting_checkins_greedy], [values_greedy, hitting_checkins_greedy]]

def step_filter(mdp, new_chains, all_chains, distributions, margin, bounding_box):

    # costs = []
    # indices = []

    # for i in range(len(all_chains)):
    #     chain = all_chains[i]

    #     idx = []

    #     points = chainPoints(chain, is_lower_bound)
        
    #     for j in range(len(points)):
    #         point = points[j]
    #         cost = getStateDistributionParetoValues(mdp, point, distributions)
    #         idx.append(len(costs))
    #         costs.append(cost)

    #     indices.append(idx)
    upper = []
    for i in range(len(all_chains)):
        sched = all_chains[i]
        sched.project_bounds(lambda point: getStateDistributionParetoValues(mdp, point, distributions))
        upper += sched.proj_upper_bound
        
    #costs = [getStateDistributionParetoValues(mdp, chain, distributions) for chain in all_chains]
    # is_efficient = calculateParetoFrontC(costs)
    # is_efficient_chains = []
    #is_efficient = calculateParetoFrontSched(all_chains)
    is_efficient_upper = calculateParetoFrontC(upper)
    front_upper = np.array([upper[i] for i in range(len(upper)) if is_efficient_upper[i]])
    
    is_efficient = calculateParetoFrontSchedUpper(all_chains, front_upper)
    
    # front = np.array([costs[i] for i in range(len(costs)) if is_efficient[i]])
    # front = np.array([all_chains[i] for i in range(len(all_chains)) if is_efficient[i]])

    #filtered_all_chains = [all_chains[i] for i in range(len(all_chains)) if is_efficient[i]]
    filtered_all_chains = []
    for i in range(len(all_chains)):
        sched = all_chains[i]

        # efficient = False

        # for idx in indices[i]:
        #     if is_efficient[idx]:
        #         efficient = True
        #         filtered_all_chains.append(chain)
        #         break
        # is_efficient_chains.append(efficient)
        efficient = is_efficient[i]
        
        if efficient:
            filtered_all_chains.append(sched)
        elif margin > 0:
            # for idx in indices[i]:
            #     cost = np.array(costs[idx])
            for lower in sched.proj_lower_bound:
                dist = calculateDistance(lower, front_upper, bounding_box)
                if dist <= margin:
                    filtered_all_chains.append(sched)
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
    

def drawParetoStep(mdp, schedules, initialDistribution, TRUTH, TRUTH_COSTS, plotName, title, stepLen, bounding_box, outputDir):

    plotName += "-step" + str(stepLen)
    title += " Length " + str(stepLen)

    # start_state_costs, indices = getStartParetoValues(mdp, chains, initialDistribution, is_lower_bound=True)
    # start_state_costs_upper, _ = getStartParetoValues(mdp, chains_upper, initialDistribution, is_lower_bound=False)
        
    # is_efficient = calculateParetoFront(start_state_costs)

    # is_efficient_upper = calculateParetoFront(start_state_costs_upper)
    # front_upper = [start_state_costs_upper[i] for i in range(len(start_state_costs_upper)) if is_efficient_upper[i]]
    # front_upper.sort(key = lambda point: point[1][0])

    sched_bounds, is_efficient, front_lower, front_upper = getData(mdp, schedules, initialDistribution)

    error = 0 if TRUTH is None else calculateError((front_lower, front_upper), TRUTH, bounding_box)
    print("Error from true Pareto:",error)

    saveDataChains(sched_bounds, is_efficient, front_lower, front_upper, TRUTH, TRUTH_COSTS, "pareto-" + plotName, outputDir)
    drawParetoFront(sched_bounds, is_efficient, front_lower, front_upper, TRUTH, TRUTH_COSTS, "pareto-" + plotName, title, bounding_box, prints=False, outputDir=outputDir)


def mixedPolicy(values1, values2, compMDP1, compMDP2, alpha, discount):
    values_blend = {state: alpha * values1[state] + (1-alpha) * values2[state] for state in compMDP1.states}
    blendedMDP = blendMDPCosts(compMDP1, compMDP2, alpha) 
    policy_blend = policyFromValues(blendedMDP, values_blend, discount)

    return policy_blend


# def mixedPolicy2(values1, values2, compMDP1, compMDP2, alpha, discount):
#     values_blend = {state: alpha * values1[state] + (1-alpha) * values2[state] for state in compMDP1.states}
#     blendedMDP = blendMDPCosts(compMDP1, compMDP2, alpha) 
#     policy_blend = policyFromValues(blendedMDP, values_blend, discount)
#     #policy_blend = policyFromValues(compMDP, values_blend)

#     return policy_blend


def createChainTail(grid, mdp, discount, discount_checkin, target_state, compMDPs, greedyCompMDPs, k, midpoints):
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

    # values_blend = {state: alpha * values[state] + beta * values_greedy[state] for state in mdp.states}
    # blendedMDP = blendMDPCosts(compMDP, greedyMDP, alpha, beta) 
    # policy_blend = policyFromValues(blendedMDP, values_blend)

    # eval_blend_exec = policyEvaluation(compMDP, policy_blend, discount_t)
    # eval_blend_check = policyEvaluation(greedyMDP, policy_blend, discount_c_t)
    midpoint_evals = []

    for midpoint_alpha in midpoints:
        policy_blend = mixedPolicy(values, values_greedy, compMDP, greedyMDP, midpoint_alpha, discount_t) # TODO what if discounts are different?
        # policy_blend = mixedPolicy2(values, eval_greedy, compMDP, greedyMDP, midpoint_alpha)

        eval_blend_exec = policyEvaluation(compMDP, policy_blend, discount_t)
        eval_blend_check = policyEvaluation(greedyMDP, policy_blend, discount_c_t)

        midpoint_evals.append((eval_blend_exec, eval_blend_check))

    #hitting_checkins_greedy = expectedMarkovHittingTime(mdp, markovProbsFromPolicy(greedyMDP, policy_greedy), target_state, 1)

    # print(k)
    # # print("valu1",values)
    # print("valu",[values_greedy[mdp.states[i]] for i in range(len(mdp.states))])
    # print("eval",eval_greedy)
    # print("hitt", hitting_checkins_greedy)

    # draw(grid, greedyMDP, values_greedy, policy_greedy, True, False, "output/policy-chain"+str(k)+"-lp")

    #chain = [[k], values, [policy], (hitting_time, hitting_checkins)]
    #chain = [[k], [[values, hitting_checkins], [values_greedy, eval_greedy]]]
    #chain = [[k], [[values, eval_normal], [values_greedy, eval_greedy], midpoint_evals]]
    sched = Schedule(strides=[k], pi_exec_data=(values, eval_normal), pi_checkin_data=(eval_greedy, values_greedy), pi_mid_data=midpoint_evals)
    return sched

def extendChain(discount, discount_checkin, compMDPs, greedyCompMDPs, sched, k, midpoints):
    compMDP = compMDPs[k]
    greedyMDP = greedyCompMDPs[k]

    chain_checkins = list(sched.strides)
    chain_checkins.insert(0, k)

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

    old_values = sched.pi_exec_data[0]
    new_values = runOneValueIterationPass(old_values, discount_t, compMDP)
    policy = policyFromValues(compMDP, new_values, discount_t)
    
    # prev_hitting = tail_values[0][1]
    # new_hitting = extendMarkovHittingTime(mdp, markovProbsFromPolicy(compMDP, policy), target_state, 1, prev_hitting)
    old_eval = sched.pi_exec_data[1]
    new_eval = extendPolicyEvaluation(greedyMDP, policy, old_eval, discount_c_t)


    old_values_greedy = sched.pi_checkin_data[1]
    new_values_greedy = runOneValueIterationPass(old_values_greedy, discount_c_t, greedyMDP)
    policy_greedy = policyFromValues(greedyMDP, new_values_greedy, discount_c_t)

    #new_eval_greedy = extendPolicyEvaluation(greedyMDP, policy_greedy, tail_values[1][1], discount_c_t)
    old_eval_greedy = sched.pi_checkin_data[0]
    new_eval_greedy = extendPolicyEvaluation(compMDP, policy_greedy, old_eval_greedy, discount_t)
    
    # prev_hitting_greedy = tail_values[1][2]
    # new_hitting_greedy = extendMarkovHittingTime(mdp, markovProbsFromPolicy(greedyMDP, policy_greedy), target_state, 1, prev_hitting_greedy)


    # values_blend = {state: alpha * new_values[state] + beta * new_values_greedy[state] for state in mdp.states}
    # blendedMDP = blendMDPCosts(compMDP, greedyMDP, alpha, beta) 
    # policy_blend = policyFromValues(blendedMDP, values_blend)

    # eval_blend_exec = extendPolicyEvaluation(compMDP, policy_blend, tail_values[2][0], discount_t)
    # eval_blend_check = extendPolicyEvaluation(greedyMDP, policy_blend, tail_values[2][1], discount_c_t)

    midpoint_evals = sched.pi_mid_data
    new_midpoint_evals = []
    for m_ind in range(len(midpoints)):
        midpoint_alpha = midpoints[m_ind]
        evals = midpoint_evals[m_ind]

        policy_blend = mixedPolicy(new_values, new_values_greedy, compMDP, greedyMDP, midpoint_alpha, discount_t) # TODO what if discounts are different?
        # policy_blend = mixedPolicy2(new_values, new_eval_greedy, compMDP, greedyMDP, midpoint_alpha)

        eval_blend_exec = extendPolicyEvaluation(compMDP, policy_blend, evals[0], discount_t)
        eval_blend_check = extendPolicyEvaluation(greedyMDP, policy_blend, evals[1], discount_c_t)

        new_midpoint_evals.append((eval_blend_exec, eval_blend_check))
    
    # new_chain = [chain_checkins, [[new_values, new_hitting], [new_values_greedy, new_eval_greedy, new_hitting_greedy]]]
    # new_chain = [chain_checkins, [[new_values, new_eval], [new_values_greedy, new_eval_greedy], new_midpoint_evals]]
    new_sched = Schedule(
        strides=chain_checkins, 
        pi_exec_data=(new_values, new_eval), 
        pi_checkin_data=(new_eval_greedy, new_values_greedy), 
        pi_mid_data=new_midpoint_evals)
    
    return new_sched


def runExtensionStage(mdp, stage, chains_list, all_chains, compMDPs, greedyCompMDPs, discount, discount_checkin, checkin_periods, do_filter, distributions, margin, bounding_box, midpoints):
    chains = []
    previous_chains = chains_list[stage - 1]

    for tail in previous_chains:
        for k in checkin_periods:
            if stage == 1 and k == tail.strides[0]:
                continue # don't duplicate recurring tail value (e.g. 23* and 233*)
            
            new_chain = extendChain(discount, discount_checkin, compMDPs, greedyCompMDPs, tail, k, midpoints)
            chains.append(new_chain)
            all_chains.append(new_chain)
    
    if do_filter:
        filtered_chains, filtered_all_chains = step_filter(mdp, chains, all_chains, distributions, margin, bounding_box)
        #print("Filtered from",len(chains),"to",len(filtered_chains),"new chains and",len(all_chains),"to",len(filtered_all_chains),"total.")
        og_len = len(all_chains) - len(chains)
        new_len_min_add = len(filtered_all_chains) - len(filtered_chains)
        removed = og_len - new_len_min_add
        
        # print("Considering new chains: " + chains_to_str(chains))
        print("Added",len(filtered_chains),"out of",len(chains),"new schedules and removed",removed,"out of",og_len,"previous schedules.")
        all_chains = filtered_all_chains

        chains_list.append(filtered_chains)
    else:
        chains_list.append(chains)

    return all_chains


def calculateChainValues(grid, mdp, discount, discount_checkin, start_state, target_state, 
                         checkin_periods, chain_length, do_filter, distributions, initialDistribution, margin, 
                         bounding_box, drawIntermediate, TRUTH, TRUTH_COSTS, name, title, midpoints, outputDir):
    all_compMDPs = createCompositeMDPs(mdp, discount, checkin_periods[-1])
    compMDPs = {k: all_compMDPs[k - 1] for k in checkin_periods}

    # greedy_mdp = convertToGreedyMDP(grid, mdp)
    # all_greedy_compMDPs = createCompositeMDPs(greedy_mdp, discount_checkin, checkin_periods[-1])
    # greedyCompMDPs = {k: all_greedy_compMDPs[k-1] for k in checkin_periods}
    greedyCompMDPs = {k: convertCompToCheckinMDP(grid, compMDPs[k], k, discount_checkin) for k in checkin_periods}

    # for k in checkin_periods:
    #     print(k,greedyCompMDPs[k].rewards[greedyCompMDPs[k].states[0]][greedyCompMDPs[k].actions[0]])

    chains_list = []
    all_chains = []

    # chains_list_upper = []
    # all_chains_upper = []

    chains_list.append([])
    # chains_list_upper.append([])

    l = 1
    
    for k in checkin_periods:
        chain = createChainTail(grid, mdp, discount, discount_checkin, target_state, compMDPs, greedyCompMDPs, k, midpoints)
        chains_list[0].append(chain)
        all_chains.append(chain)

        # chains_list_upper[0].append(chain)
        # all_chains_upper.append(chain)

    # if True:
    # print(getStartParetoValues(mdp, all_chains_upper, initialDistribution, is_lower_bound=False))
    #     exit()


    if drawIntermediate:
        drawParetoStep(mdp, all_chains, initialDistribution, TRUTH, TRUTH_COSTS, name, title, l, bounding_box, outputDir)


    print("--------")
    print(len(all_chains),"current schedules")
    # print(len(all_chains_upper),"current upper bound chains")
    # print("Current chains: " + chains_to_str(all_chains))

    for i in range(1, chain_length):
        l += 1

        all_chains = runExtensionStage(mdp, i, chains_list, all_chains, compMDPs, greedyCompMDPs, discount, discount_checkin, checkin_periods, do_filter, distributions, margin, bounding_box, midpoints)

        if drawIntermediate:
            drawParetoStep(mdp, all_chains, initialDistribution, TRUTH, TRUTH_COSTS, name, title, l, bounding_box, outputDir)

        print("--------")
        print(len(all_chains),"current schedules")
        # print("Current chains: " + chains_to_str(all_chains))

    start_state_index = mdp.states.index(start_state)

    # chains = sorted(chains, key=lambda chain: chain[1][start_state], reverse=True)

    # start_state_costs, indices = getStartParetoValues(mdp, all_chains, initialDistribution)
    # return start_state_costs, indices
    return all_chains

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

def is_eff(schedules, i):
    sched = schedules[i]
    for j in range(len(schedules)):
        if j != i and not sched.not_dominated_by(schedules[j]):
            return False
    return True

# def calculateParetoFrontSched(schedules):
#     is_efficient = [is_eff(schedules, i) for i in range(len(schedules))]
#     return is_efficient

def calculateParetoFrontSchedUpper(schedules, upper):
    is_efficient = [schedules[i].not_dominated_by(upper) for i in range(len(schedules))]
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

def calculateError(fronts, true_fronts, bounding_box):
    
    front_lower, front_upper = fronts
    truth_optimistic_front, truth_realizable_front = true_fronts

    area_front_lower = areaUnderFront(front_lower, bounding_box)
    area_front_upper = areaUnderFront(front_upper, bounding_box)

    area_true_lower = areaUnderFront(truth_optimistic_front, bounding_box)
    area_true_upper = areaUnderFront(truth_realizable_front, bounding_box)

    #return abs(area - area_true) / area_true
    error = (abs(area_front_upper - area_true_upper) + abs(area_front_lower - area_true_lower)) / (area_true_upper + area_true_lower)
    return error

def areaUnderFront(front, bounding_box):
    # chains_filtered = [chains[i][1] for i in range(len(chains)) if is_efficient[i]]
    # chains_filtered.sort(key = lambda chain: chain[0])

    # true = [t[1] for t in true_front]

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

    front_costs = [point[1] for point in front]
    front_normalized = np.divide(np.array(front_costs) - mins, ranges)
    
    return areaUnderPareto(front_normalized)

def saveDataChains(sched_bounds, is_efficient, front_lower, front_upper, TRUTH, TRUTH_COSTS, name, outputDir):
    sched_data = [sched.to_arr() for sched in sched_bounds]
    data = {'Schedules': sched_data, 'Efficient': is_efficient, 'Optimistic Front': front_lower, 'Realizable Front': front_upper}
    # if TRUTH is not None:
    #     data['Truth'] = TRUTH
    # if TRUTH_COSTS is not None:
    #     data['Truth Costs'] = TRUTH_COSTS
    jsonStr = json.dumps(data, indent=4)
    
    with open(f'{outputDir}/data/{name}.json', "w") as file:
        file.write(jsonStr)

# def loadDataChains(filename):
#     with open(f'output/data/{filename}.json', "r") as file:
#         jsonStr = file.read()
#         obj = json.loads(jsonStr)
#         return (obj['Points'], obj['Indices'], obj['Efficient'])

# def drawChainsParetoFront(chains, indices, is_efficient, true_front, true_costs, name, title, bounding_box, prints, x_offset=0, x_scale=1, loffsets={}):
#     plt.style.use('seaborn-whitegrid')

#     arrows = True

#     font = FontProperties()
#     font.set_family('serif')
#     font.set_name('Times New Roman')
#     font.set_size(20)
#     # rc('font',**{'family':'serif','serif':['Times'],'size':20})
#     plt.rcParams['font.family'] = 'serif'
#     plt.rcParams['font.serif'] = ['Times']
#     plt.rcParams['font.size'] = 20
#     plt.rcParams["text.usetex"] = True
#     # plt.rcParams['font.weight'] = 'bold'
    
#     chains_filtered = []
#     chains_dominated = []
#     for i in range(len(chains)):
#         if is_efficient[i]:
#             chains_filtered.append(chains[i])
#         else:
#             chains_dominated.append(chains[i])

#     n = 0
#     is_efficient_chains = []
#     for i in range(len(indices)):
#         idx = indices[i][1]

#         efficient = False
#         for j in idx:
#             if is_efficient[j]:
#                 efficient = True
#                 n += 1
#                 break
#         is_efficient_chains.append(efficient)

#     print(n,"vs",len(indices))

#     chains_filtered.sort(key = lambda chain: chain[1][0])

#     if prints:
#         print("Non-dominated chains:")
#         for chain in chains_filtered:
#             print("  ", chain[0])
#     # x_f = [chain[1] for chain in chains_filtered]
#     # y_f = [chain[2] for chain in chains_filtered]
#     # labels_f = [chain[0] for chain in chains_filtered]

#     if prints:
#         print(len(chains_dominated),"dominated chains out of",len(chains),"|",len(chains_filtered),"non-dominated")

#     # costs = [chain[1] for chain in chains_filtered]
#     if prints:
#         print("Pareto front:",chains_filtered)
    
#     fig, ax = plt.subplots()
#     # ax.scatter(x, y, c=["red" if is_efficient[i] else "black" for i in range(len(chains))])
#     # ax.scatter(x_f, y_f, c="red")

#     if true_costs is not None:
#         scatter(ax, true_costs, doLabel=False, color="gainsboro", lcolor="gray", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)

#     if true_front is not None:
#         manhattan_lines(ax, true_front, color="green", bounding_box=bounding_box, x_offset=x_offset, x_scale=x_scale)
#         scatter(ax, true_front, doLabel=False, color="green", lcolor="green", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)

    
#     # scatter(ax, chains_dominated, doLabel=True, color="orange", lcolor="gray", arrows=arrows, x_offset=x_offset, x_scale=x_scale)
    
#     # scatter(ax, chains_dominated, doLabel=False, color="orange", lcolor="gray", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)

#     for i in range(len(is_efficient_chains)):
#         points = []
#         for j in indices[i][1]:
#             points.append(chains[j])
#         if is_efficient_chains[i]:
#             box(ax, points, color="red", bounding_box=bounding_box, x_offset=x_offset, x_scale=x_scale)
#             scatter(ax, points, doLabel=True, color="red", lcolor="gray", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)
#         else:
#             print("bad",indices[i][0], points[0])
#             box(ax, points, color="orange", bounding_box=bounding_box, x_offset=x_offset, x_scale=x_scale)
#             scatter(ax, points, doLabel=False, color="orange", lcolor="gray", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)

    

#     manhattan_lines(ax, chains_filtered, color="red", bounding_box=bounding_box, x_offset=x_offset, x_scale=x_scale)
#     scatter(ax, chains_filtered, doLabel=True, color="red", lcolor="black", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)
    
#     # for i in range(len(chains)):
#     #     plt.plot(x[i], y[i])

#     # plt.xlabel("Execution Cost", fontproperties=font, fontweight='bold')
#     # plt.ylabel("Checkin Cost", fontproperties=font, fontweight='bold')
#     plt.xlabel(r"\textbf{Execution Cost}", fontproperties=font, fontweight='bold')
#     plt.ylabel(r"\textbf{Checkin Cost}", fontproperties=font, fontweight='bold')
#     #plt.title(title)

#     plt.xlim((bounding_box[0] + x_offset) * x_scale)
#     plt.ylim(bounding_box[1])

#     plt.gcf().set_size_inches(10, 7)
#     plt.savefig(f'output/{name}.pdf', format="pdf", bbox_inches='tight', pad_inches=0.2, dpi=300)
#     # plt.savefig(f'output/{name}.png', bbox_inches='tight', pad_inches=0.5, dpi=300)
#     # plt.savefig(f'output/pareto-{name}.svg', bbox_inches='tight', pad_inches=0.5, dpi=300, format="svg")
#     # plt.show()



def drawChainsParetoFrontSuperimposed(stuffs, true_front, true_costs, name, bounding_box, x_offset=0, x_scale=1, loffsets={}, outputDir="output"):
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
    plt.savefig(f'{outputDir}/{name}.pdf', format="pdf", bbox_inches='tight', pad_inches=0.2, dpi=300)
    # plt.savefig(f'output/{name}.png', bbox_inches='tight', pad_inches=0.5, dpi=300)
    # plt.savefig(f'output/pareto-{name}.svg', bbox_inches='tight', pad_inches=0.5, dpi=300, format="svg")
    # plt.show()


def drawCompares(data, outputDir="output"):
    plt.style.use('seaborn-whitegrid')

    fig, ax = plt.subplots()
    
    scatter(ax, data, doLabel=True, color="red", lcolor="black")

    plt.xlabel("Evaluation Time (s)")
    plt.ylabel("Error (%)")

    plt.gcf().set_size_inches(10, 7)
    plt.savefig(f'{outputDir}/pareto-compare.png', bbox_inches='tight', pad_inches=0.5, dpi=300)
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


def getData(mdp, schedules, initialDistribution):
    dists = [initialDistribution]
    
    upper = []
    lower = []
    sched_bounds = []
    for sched in schedules:
        sched.project_bounds(lambda point: getStateDistributionParetoValues(mdp, point, dists))

        name = sched.to_str()
        for b in sched.proj_upper_bound:
            upper.append([name, b])
        for b in sched.proj_lower_bound:
            lower.append([name, b])

        sched_bounds.append(sched.get_proj_bounds())

    is_efficient_upper = calculateParetoFront(upper)
    front_upper_nolbl = np.array([upper[i][1] for i in range(len(upper)) if is_efficient_upper[i]])
    front_upper = [upper[i] for i in range(len(upper)) if is_efficient_upper[i]]

    is_efficient_lower = calculateParetoFront(lower)
    front_lower = [lower[i] for i in range(len(lower)) if is_efficient_lower[i]]

    is_efficient = calculateParetoFrontSchedUpper(schedules, front_upper_nolbl)

    front_lower.sort(key = lambda point: point[1][0])
    front_upper.sort(key = lambda point: point[1][0])

    return sched_bounds, is_efficient, front_lower, front_upper


def runChains(grid, mdp, discount, discount_checkin, start_state, target_state, 
    checkin_periods, chain_length, do_filter, margin, distName, startName, 
    distributions, initialDistribution, bounding_box, TRUTH, TRUTH_COSTS, drawIntermediate, midpoints, outputDir="output"):
        
    midpoints.sort(reverse=True)

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

    schedules = calculateChainValues(grid, mdp, discount, discount_checkin, start_state, target_state, 
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
        title=title,
        midpoints=midpoints, 
        outputDir=outputDir)

    numRemaining = len(schedules)# / 3 #because 3 points in each L
    numWouldBeTotal = pow(len(checkin_periods), chain_length)
    numPruned = numWouldBeTotal - numRemaining
    fractionTrimmed = numPruned / numWouldBeTotal * 100

    # is_efficient = calculateParetoFront(start_state_costs)

    # is_efficient_upper = calculateParetoFront(start_state_costs_upper)
    # front_upper = [start_state_costs_upper[i] for i in range(len(start_state_costs_upper)) if is_efficient_upper[i]]
    # front_upper.sort(key = lambda point: point[1][0])
    sched_bounds, is_efficient, front_lower, front_upper = getData(mdp, schedules, initialDistribution)
    
    c_end = time.time()
    running_time = c_end - c_start
    print("Chain evaluation time:", running_time)
    print("Trimmed:",numPruned,"/",numWouldBeTotal,"(" + str(int(fractionTrimmed)) + "%)")

    error = 0 if TRUTH is None else calculateError((front_lower, front_upper), TRUTH, bounding_box)
    print("Error from true Pareto:",error)

    saveDataChains(sched_bounds, is_efficient, front_lower, front_upper, TRUTH, TRUTH_COSTS, "pareto-" + name, outputDir)
    drawParetoFront(sched_bounds, is_efficient, front_lower, front_upper, TRUTH, TRUTH_COSTS, "pareto-" + name, title, bounding_box, prints=False, outputDir=outputDir)

    # print("All costs:",start_state_costs)

    return running_time, error, fractionTrimmed


def getAdjustedAlphaValue(alpha, scaling_factor):
    if alpha >= 1:
        return 1
    # 0.5/0.5:
    #(0.5x) = 0.5 (1-x) * scaling_factor
    #(0.5 / scaling_factor + 0.5) x = 0.5
    #x = 0.5 / (0.5 / scaling_factor + 0.5)

    # 0.25/0.75:
    #(0.75x) = 0.25 (1-x) * scaling_factor
    #(0.75 / scaling_factor + 0.25) x = 0.25
    #x = 0.25 / (0.75 / scaling_factor + 0.75)

    beta = 1 - alpha

    scaled_alpha = alpha / (beta / scaling_factor + beta)
    return scaled_alpha


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

    # composed_cost = 0
    # for i in range(checkin_period):
    #     composed_cost += pow(discount, i) * cost_per_action
    composed_cost = cost_per_stride

    for state in compMDP.rewards:
        (x, y) = state
        state_type = grid[y][x]

        checkinMDP.rewards[state] = {}

        for action in compMDP.rewards[state]:
            checkinMDP.rewards[state][action] = 0 if state_type == TYPE_GOAL else (-composed_cost)
        
    return checkinMDP

def blendMDPCosts(mdp1, mdp2, alpha):

    blend = MDP([], [], {}, {}, [])

    blend.states = mdp1.states.copy()
    blend.terminals = mdp1.terminals.copy()
    blend.actions = mdp1.actions.copy()
    blend.transitions = mdp1.transitions.copy()

    for state in mdp1.rewards:
        (x, y) = state

        blend.rewards[state] = {}

        for action in mdp1.rewards[state]:
            blend.rewards[state][action] = alpha * mdp1.rewards[state][action] + (1 - alpha) * mdp2.rewards[state][action]
        
    return blend