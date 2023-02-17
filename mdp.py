
from tabnanny import check
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation

from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

import pandas as pd

from networkx.drawing.nx_agraph import graphviz_layout, to_agraph

import colorsys
import math

import os

import time
import math

from lp import linearProgrammingSolve

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

    return values[start_state], elapsed


def runFig2Ratio(wallMin, wallMax, increment = 1, _discount = math.sqrt(0.99)):
    results = []
    for numWalls in range(wallMin, wallMax+increment, increment):
        grid, mdp, discount, start_state = paper2An(numWalls, _discount)

        pref = "paperFig2-" + str(numWalls) + "w-"
        value2, elapsed2 = run(grid, mdp, discount, start_state, checkin_period=2, doBranchAndBound=False, doLinearProg=True, drawPolicy=False, drawIterations=False, outputPrefix=pref)
        value3, elapsed3 = run(grid, mdp, discount, start_state, checkin_period=3, doBranchAndBound=False, doLinearProg=True, drawPolicy=False, drawIterations=False, outputPrefix=pref)

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
            value, elapsed = run(grid, mdp, discount, start_state, checkin_period, doBranchAndBound=False, doLinearProg=True) # LP
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

def getAllStateParetoValues(mdp, chain):
    pareto_values = []
    for i in range(len(mdp.states)):
        state = mdp.states[i]

        values = chain[1]
        hitting = chain[3]
    
        hitting_time = hitting[0][i]
        hitting_checkins = hitting[1][i]

        checkin_cost = hitting_checkins
        execution_cost = - values[state]

        pareto_values.append(checkin_cost)
        pareto_values.append(execution_cost)
    return pareto_values

def getStateDistributionParetoValues(mdp, chain, distributions):
    pareto_values = []
    for distribution in distributions:
        dist_checkin_cost = 0
        dist_execution_cost = 0

        for i in range(len(mdp.states)):
            state = mdp.states[i]

            values = chain[1]
            hitting = chain[3]
        
            hitting_time = hitting[0][i]
            hitting_checkins = hitting[1][i]

            checkin_cost = hitting_checkins
            execution_cost = - values[state]

            dist_checkin_cost += distribution[i] * checkin_cost
            dist_execution_cost += distribution[i] * execution_cost

        pareto_values.append(dist_execution_cost)
        pareto_values.append(dist_checkin_cost)
    return pareto_values

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

def step_filter(new_chains, all_chains, distributions, margin, bounding_box):

    costs = [getStateDistributionParetoValues(mdp, chain, distributions) for chain in all_chains]
    is_efficient = calculateParetoFrontC(costs)

    filtered_all_chains = [all_chains[i] for i in range(len(all_chains)) if is_efficient[i]]
    front = np.array([costs[i] for i in range(len(all_chains)) if is_efficient[i]])
    
    if margin > 0 and len(front) >= 1:
        for i in range(len(all_chains)):
            if not is_efficient[i]:
                chain = all_chains[i]
                cost = np.array(costs[i])
                dist = calculateDistance(cost, front, bounding_box)
                if dist <= margin:
                    filtered_all_chains.append(chain)

    filtered_new_chains = [chain for chain in new_chains if chain in filtered_all_chains] # can do this faster with index math

    return filtered_new_chains, filtered_all_chains


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


def calculateChainValues(grid, mdp, discount, start_state, target_state, checkin_periods, chain_length, do_filter, distributions, margin, bounding_box):
    all_compMDPs = createCompositeMDPs(mdp, discount, checkin_periods[-1])
    compMDPs = {k: all_compMDPs[k - 1] for k in checkin_periods}

    chains_list = []
    all_chains = []

    chains = []
    for k in checkin_periods:
        discount_t = pow(discount, k)
        compMDP = compMDPs[k]

        policy, values = linearProgrammingSolve(grid, compMDP, discount_t)
        
        markov = markovProbsFromPolicy(compMDP, policy)
        hitting_time = expectedMarkovHittingTime(mdp, markov, target_state, k)
        hitting_checkins = expectedMarkovHittingTime(mdp, markov, target_state, 1)

        chain = ([k], values, [policy], (hitting_time, hitting_checkins))
        chains.append(chain)
        all_chains.append(chain)
    chains_list.append(chains)

    print("--------")
    print("Current chains: " + chains_to_str(all_chains))

    for i in range(1, chain_length):
        previous_chains = chains_list[i - 1]
        chains = []

        for tail in previous_chains:
            for k in checkin_periods:
                if i == 1 and k == tail[0][0]:
                    continue # don't duplicate recurring tail value (e.g. 23* and 233*)

                compMDP = compMDPs[k]

                chain = list(tail[0])
                chain.insert(0, k)

                tail_values = tail[1]
                discount_t = pow(discount, k)

                new_values = runOneValueIterationPass(tail_values, discount_t, compMDP)

                policies = list(tail[2])
                policy = policyFromValues(compMDP, tail_values)
                policies.insert(0, policy)

                markov = markovProbsFromPolicy(compMDP, policy)
                prev_hitting_time = tail[3][0]
                prev_hitting_checkins = tail[3][1]
                hitting_time = extendMarkovHittingTime(mdp, markov, target_state, k, prev_hitting_time)
                hitting_checkins = extendMarkovHittingTime(mdp, markov, target_state, 1, prev_hitting_checkins)
                
                new_chain = (chain, new_values, policies, (hitting_time, hitting_checkins))
                chains.append(new_chain)
                all_chains.append(new_chain)
        
        if do_filter:
            filtered_chains, filtered_all_chains = step_filter(chains, all_chains, distributions, margin, bounding_box)
            #print("Filtered from",len(chains),"to",len(filtered_chains),"new chains and",len(all_chains),"to",len(filtered_all_chains),"total.")
            og_len = len(all_chains) - len(chains)
            new_len_min_add = len(filtered_all_chains) - len(filtered_chains)
            removed = og_len - new_len_min_add
            
            print("Considering new chains: " + chains_to_str(chains))
            print("Added",len(filtered_chains),"out of",len(chains),"new chains and removed",removed,"out of",og_len,"previous chains.")
            all_chains = filtered_all_chains

            chains_list.append(filtered_chains)
        else:
            chains_list.append(chains)

        print("--------")
        print("Current chains: " + chains_to_str(all_chains))

    start_state_index = mdp.states.index(start_state)

    chains = all_chains
    chains = sorted(chains, key=lambda chain: chain[1][start_state], reverse=True)

    costs = []
    start_state_costs = []

    for chain in chains:
        name = ""
        for checkin in chain[0]:
            name += str(checkin)
        name += "*"

        values = chain[1]
        hitting = chain[3]

        hitting_time = hitting[0][start_state_index]
        hitting_checkins = hitting[1][start_state_index]

        checkin_cost = hitting_checkins
        execution_cost = - values[start_state]

        # pareto_values = getAllStateParetoValues(mdp, chain)
        pareto_values = getStateDistributionParetoValues(mdp, chain, distributions)

        # print(name + ":", values[start_state], "| Hitting time:", hitting_time, "| Hitting checkins:", hitting_checkins, "| Execution cost:", execution_cost, "| Checkin cost:", checkin_cost)
        print(name + ":", values[start_state], "| Execution cost:", execution_cost, "| Checkin cost:", checkin_cost)
        # costs.append((name, execution_cost, checkin_cost))
        costs.append((name, pareto_values))
        start_state_costs.append((name, [execution_cost, checkin_cost]))
        
    return costs, start_state_costs

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

def scatter(ax, chains, doLabel, color, lcolor):
    # x = [chain[1][start_state_index * 2 + 1] for chain in chains]
    # y = [chain[1][start_state_index * 2] for chain in chains]
    x = [chain[1][0] for chain in chains]
    y = [chain[1][1] for chain in chains]
    labels = [chain[0] for chain in chains]
    
    ax.scatter(x, y, c=color)

    if doLabel:
        for i in range(len(labels)):
            ax.annotate(labels[i], (x[i], y[i]), color=lcolor)

def lines(ax, chains, color):
    x = [chain[1][0] for chain in chains]
    y = [chain[1][1] for chain in chains]
    
    ax.plot(x, y, c=color)

def manhattan_lines(ax, chains, color):
    x = []
    y = []

    for i in range(len(chains) - 1):
        point = chains[i][1]
        next_point = chains[i+1][1]
        
        x.append(point[0])
        y.append(point[1])

        x.append(next_point[0])
        y.append(point[1])
    
    ax.plot(x, y, c=color)

def calculateParetoFront(chains):
    return calculateParetoFrontC([chain[1] for chain in chains])

def calculateParetoFrontC(costs):
    costs = np.array(costs)

    is_efficient = np.ones(len(costs), dtype = bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i]>c, axis=1)) and np.all(np.any(costs[i+1:]>c, axis=1))

    return is_efficient

def areaUnderPareto(pareto_front):

    area = 0

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

def drawChainsParetoFront(chains, is_efficient, true_front, true_costs, name):
    plt.style.use('seaborn-whitegrid')

    chains_filtered = []
    chains_dominated = []
    for i in range(len(chains)):
        if is_efficient[i]:
            chains_filtered.append(chains[i])
        else:
            chains_dominated.append(chains[i])

    chains_filtered.sort(key = lambda chain: chain[1][0])

    print("Non-dominated chains:")
    for chain in chains_filtered:
        print("  ", chain[0])
    # x_f = [chain[1] for chain in chains_filtered]
    # y_f = [chain[2] for chain in chains_filtered]
    # labels_f = [chain[0] for chain in chains_filtered]

    print(len(chains_dominated),"dominated chains out of",len(chains),"|",len(chains_filtered),"non-dominated")

    # costs = [chain[1] for chain in chains_filtered]
    print("Pareto front:",chains_filtered)
    
    fig, ax = plt.subplots()
    # ax.scatter(x, y, c=["red" if is_efficient[i] else "black" for i in range(len(chains))])
    # ax.scatter(x_f, y_f, c="red")

    scatter(ax, true_costs, doLabel=False, color="gainsboro", lcolor="gray")

    manhattan_lines(ax, true_front, color="green")
    scatter(ax, true_front, doLabel=True, color="green", lcolor="green")
    
    scatter(ax, chains_dominated, doLabel=True, color="orange", lcolor="gray")
    
    manhattan_lines(ax, chains_filtered, color="red")
    scatter(ax, chains_filtered, doLabel=True, color="red", lcolor="black")
    
    # for i in range(len(chains)):
    #     plt.plot(x[i], y[i])

    plt.xlabel("Execution Cost")
    plt.ylabel("Checkin Cost")

    plt.gcf().set_size_inches(10, 7)
    plt.savefig(f'output/pareto-{name}.png', bbox_inches='tight', pad_inches=0.5, dpi=300)
    plt.show()



def drawCompares(data):
    plt.style.use('seaborn-whitegrid')

    fig, ax = plt.subplots()
    
    scatter(ax, data, doLabel=True, color="red", lcolor="black")

    plt.xlabel("Evaluation Time (s)")
    plt.ylabel("Error (%)")

    plt.gcf().set_size_inches(10, 7)
    plt.savefig(f'output/pareto-compare.png', bbox_inches='tight', pad_inches=0.5, dpi=300)
    plt.show()





# start = time.time()

# grid, mdp, discount, start_state = paper2An(3)#splitterGrid(rows = 50, discount=0.99)#paper2An(3)#, 0.9999)

grid, mdp, discount, start_state = corridorTwoCadence(n1=3, n2=6, cadence1=2, cadence2=3)

# end = time.time()
# print("MDP creation time:", end - start)

# checkin_period = 8

#run(grid, mdp, discount, start_state, checkin_period, doBranchAndBound=False, doLinearProg=False) # VI
# run(grid, mdp, discount, start_state, checkin_period, doBranchAndBound=True, doLinearProg=False) # BNB
# run(grid, mdp, discount, start_state, checkin_period, doBranchAndBound=False, doLinearProg=True) # LP

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





TRUTH_234 = [('223*', [-1471895.5491260004, 10.36207210840559]), ('23*', [-1466111.360834025, 10.272072943363703]), ('2343*', [-1459989.7576398463, 10.188573289663607]), ('2234*', [-1442990.0823133066, 9.440067985664028]), ('2334*', [-1437776.5013845253, 9.358111758033958]), ('234*', [-1432535.7535183069, 9.268943242061733]), ('24*', [-1425204.691571236, 9.26894324206173])]
TRUTH = [('1121*', [-1490162.871172391, 25.553586157256976]), ('221*', [-1490162.8711723909, 24.553586157256976]), ('2221*', [-1489466.9071525347, 23.64358615725698]), ('2231*', [-1488603.1401359926, 22.738537023683413]), ('2331*', [-1482744.2415537834, 22.468598326365132]), ('2341*', [-1475378.7537851958, 22.370432645136855]), ('2113*', [-1471922.145938075, 11.362072108405588]), ('223*', [-1471895.5491260004, 10.36207210840559]), ('23*', [-1466111.360834025, 10.272072943363703]), ('2343*', [-1459989.7576398463, 10.188573289663607]), ('2234*', [-1442990.0823133066, 9.440067985664028]), ('2334*', [-1437776.5013845253, 9.358111758033958]), ('234*', [-1432535.7535183069, 9.268943242061733]), ('24*', [-1425204.691571236, 9.26894324206173])]
TRUTH_12345 = [('1121*', [-1490162.871172391, 25.553586157256976]), ('221*', [-1490162.8711723909, 24.553586157256976]), ('2221*', [-1489466.9071525347, 23.64358615725698]), ('2231*', [-1488603.1401359926, 22.738537023683413]), ('2331*', [-1482744.2415537834, 22.468598326365132]), ('2341*', [-1475378.7537851958, 22.370432645136855]), ('2113*', [-1471922.145938075, 11.362072108405588]), ('223*', [-1471895.5491260004, 10.36207210840559]), ('23*', [-1466111.360834025, 10.272072943363703]), ('2343*', [-1459989.7576398463, 10.188573289663607]), ('2353*', [-1452372.0628502036, 10.188552523383924]), ('2234*', [-1442990.0823133066, 9.440067985664028]), ('2334*', [-1437776.5013845253, 9.358111758033958]), ('234*', [-1432535.7535183069, 9.268943242061733]), ('24*', [-1425204.691571236, 9.26894324206173]), ('2354*', [-1425055.5828949779, 9.268942257572757]), ('2345*', [-1391849.9571809277, 9.250849926302044]), ('235*', [-1384573.7339997415, 9.250848891530724]), ('5*', [-1349923.035067944, 9.250848891530723])]

TRUTH_COSTS = [('1121*', [-1490162.871172391, 25.553586157256976]), ('221*', [-1490162.8711723909, 24.553586157256976]), ('1*', [-1490162.8711723906, 26.553586157256984]), ('21*', [-1490162.8711723906, 25.553586157256984]), ('2221*', [-1489466.9071525347, 23.64358615725698]), ('2231*', [-1488603.1401359926, 22.738537023683413]), ('2121*', [-1483225.5813779982, 25.463586157256977]), ('231*', [-1482999.0148700937, 24.454578742677814]), ('1131*', [-1482999.0148700937, 25.454578742677814]), ('2321*', [-1482964.3910972718, 23.459078742677814]), ('2331*', [-1482744.2415537834, 22.468598326365132]), ('321*', [-1482693.3360592595, 24.553586157256976]), ('1221*', [-1482693.3360592595, 25.553586157256976]), ('31*', [-1482693.3360592593, 25.553586157256984]), ('121*', [-1482693.3360592593, 26.553586157256984]), ('3221*', [-1482000.8606028247, 23.64358615725698]), ('2241*', [-1481178.8955120144, 22.649247056417572]), ('3231*', [-1481141.423272778, 22.738537023683413]), ('3121*', [-1475790.8198662712, 25.463586157256977]), ('2131*', [-1475640.4329844825, 25.463586157256977]), ('1231*', [-1475565.3890372368, 25.454578742677814]), ('331*', [-1475565.3890372366, 24.454578742677814]), ('3321*', [-1475530.9388182536, 23.459078742677814]), ('241*', [-1475415.002651691, 24.454578742677814]), ('1141*', [-1475415.002651691, 25.454578742677814]), ('2421*', [-1475380.552432708, 23.459078742677814]), ('2341*', [-1475378.7537851958, 22.370432645136855]), ('3331*', [-1475311.892788198, 22.468598326365132]), ('41*', [-1475261.2424606672, 25.553586157256984]), ('421*', [-1475261.2424606672, 24.553586157256976]), ('1321*', [-1475261.242460667, 25.553586157256976]), ('131*', [-1475261.2424606667, 26.553586157256984]), ('2431*', [-1475161.5064026525, 22.468598326365132]), ('4221*', [-1474572.2380810096, 23.64358615725698]), ('3241*', [-1473754.393141913, 22.649247056417572]), ('4231*', [-1473717.1087346328, 22.738537023683413]), ('2113*', [-1471922.145938075, 11.362072108405588]), ('1123*', [-1471895.5491260008, 11.36207210840559]), ('223*', [-1471895.5491260004, 10.36207210840559]), ('4121*', [-1468393.3255642185, 25.463586157256977]), ('3131*', [-1468243.692506133, 25.463586157256977]), ('431*', [-1468169.024721393, 24.454578742677814]), ('1331*', [-1468169.0247213927, 25.454578742677814]), ('2213*', [-1468153.2436541691, 11.196475818657829]), ('4321*', [-1468134.7471862994, 23.459078742677814]), ('2141*', [-1468093.3056243437, 25.463586157256977]), ('341*', [-1468019.392157063, 24.454578742677814]), ('1241*', [-1468019.392157063, 25.454578742677814]), ('3421*', [-1467985.1146219694, 23.459078742677814]), ('3341*', [-1467983.3249902911, 22.370432645136855]), ('4331*', [-1467916.7991382459, 22.468598326365132]), ('141*', [-1467866.402698667, 26.553586157256984]), ('1421*', [-1467866.402698667, 25.553586157256976]), ('2441*', [-1467832.938604746, 22.370432645136855]), ('3431*', [-1467767.1665739163, 22.468598326365132]), ('2123*', [-1466408.705859376, 11.272072108405588]), ('4241*', [-1466367.106556894, 22.649247056417572]), ('113*', [-1466111.3608340255, 11.272072943363703]), ('23*', [-1466111.360834025, 10.272072943363703]), ('2243*', [-1465616.228584849, 10.286110505296184]), ('3113*', [-1464544.0436074014, 11.362072108405588]), ('323*', [-1464517.5801135206, 10.36207210840559]), ('1223*', [-1464517.5801135201, 11.36207210840559]), ('2223*', [-1461877.3528783931, 11.128724189611201]), ('4131*', [-1460884.0286546377, 25.463586157256977]), ('1431*', [-1460809.7351468646, 25.454578742677814]), ('3213*', [-1460794.0331832329, 11.196475818657829]), ('3141*', [-1460734.3955965517, 25.463586157256977]), ('441*', [-1460660.852625174, 24.454578742677814]), ('1341*', [-1460660.852625174, 25.454578742677814]), ('2313*', [-1460643.6467976875, 11.196475818657829]), ('4421*', [-1460626.746908381, 23.459078742677814]), ('4341*', [-1460624.9662473442, 22.370432645136855]), ('3441*', [-1460475.3336830148, 22.370432645136855]), ('4431*', [-1460409.8913386262, 22.468598326365132]), ('2343*', [-1459989.7576398463, 10.188573289663607]), ('3123*', [-1459058.2399938556, 11.272072108405588]), ('1113*', [-1458907.853112067, 12.272072108405588]), ('213*', [-1458907.8531120669, 11.272072108405588]), ('3*', [-1458762.385429146, 10.272072943363701]), ('123*', [-1458762.3854291455, 11.272072943363703]), ('1143*', [-1458611.9990436006, 11.272072943363703]), ('243*', [-1458611.9990436004, 10.272072943363703]), ('3243*', [-1458269.735061509, 10.286110505296184]), ('4113*', [-1457202.9244786943, 11.362072108405588]), ('423*', [-1457176.593634741, 10.36207210840559]), ('1323*', [-1457176.593634741, 11.36207210840559]), ('2212*', [-1454562.4917619782, 16.388660680798203]), ('3223*', [-1454549.6007046825, 11.128724189611201]), ('2323*', [-1454498.5290640278, 11.11071946063285]), ('4213*', [-1453471.7112176276, 11.196475818657829]), ('4141*', [-1453412.3725681007, 25.463586157256977]), ('1441*', [-1453339.1982354922, 25.454578742677814]), ('3313*', [-1453322.078653298, 11.196475818657829]), ('2413*', [-1453171.6922677532, 11.196475818657829]), ('4441*', [-1453154.6092186985, 22.370432645136855]), ('2232*', [-1453140.235002809, 15.483121320673867]), ('3343*', [-1452671.4671559858, 10.188573289663607]), ('2443*', [-1452521.0807704409, 10.188573289663607]), ('4123*', [-1451744.618800782, 11.272072108405588]), ('313*', [-1451594.9857426966, 11.272072108405588]), ('1213*', [-1451594.9857426966, 12.272072108405588]), ('13*', [-1451450.2472256853, 11.272072943363701]), ('43*', [-1451450.2472256853, 10.272072943363703]), ('2143*', [-1451444.5988609076, 11.272072108405588]), ('343*', [-1451300.614661356, 10.272072943363703]), ('1243*', [-1451300.6146613555, 11.272072943363703]), ('4243*', [-1450960.0662990008, 10.286110505296184]), ('1423*', [-1449872.4043123855, 11.36207210840559]), ('2112*', [-1448350.9237481316, 17.29809383395393]), ('112*', [-1448337.591928803, 17.29809383395393]), ('2*', [-1448337.5919288027, 16.29809383395393]), ('2312*', [-1448107.2364449599, 16.302645101108837]), ('2332*', [-1447962.523354243, 15.312092969513923]), ('3212*', [-1447271.4057896696, 16.388660680798203]), ('4223*', [-1447258.5793496093, 11.128724189611201]), ('3323*', [-1447207.7637087659, 11.11071946063285]), ('2423*', [-1447057.377323221, 11.11071946063285]), ('2242*', [-1446487.0383991455, 15.389392814878663]), ('4313*', [-1446037.2103297107, 11.196475818657829]), ('3413*', [-1445887.5777653817, 11.196475818657829]), ('3232*', [-1445856.2781819564, 15.483121320673867]), ('4343*', [-1445389.8600634478, 10.188573289663607]), ('3443*', [-1445240.2274991188, 10.188573289663607]), ('413*', [-1444318.7745809462, 11.272072108405588]), ('1313*', [-1444318.7745809462, 12.272072108405588]), ('143*', [-1444174.7615748546, 11.272072943363703]), ('3143*', [-1444169.1415228604, 11.272072108405588]), ('443*', [-1444025.8790531647, 10.272072943363703]), ('1343*', [-1444025.8790531647, 11.272072943363703]), ('2234*', [-1442990.0823133066, 9.440067985664028]), ('212*', [-1442265.5665031679, 17.20809383395393]), ('1112*', [-1442265.5665031674, 18.20809383395393]), ('1132*', [-1442037.609696898, 17.20359117794156]), ('232*', [-1442037.6096968977, 16.20359117794156]), ('2342*', [-1441420.148637307, 15.209072657030335]), ('3112*', [-1441090.9736511593, 17.29809383395393]), ('32*', [-1441077.7086584133, 16.29809383395393]), ('12*', [-1441077.708658413, 17.29809383395393]), ('3312*', [-1440848.507845921, 16.302645101108837]), ('3332*', [-1440704.5201386728, 15.312092969513923]), ('2412*', [-1440698.1214603756, 16.302645101108837]), ('2432*', [-1440554.1337531274, 15.312092969513923]), ('4212*', [-1440016.8668443586, 16.388660680798203]), ('4323*', [-1439953.5437733876, 11.11071946063285]), ('3423*', [-1439803.9112090587, 11.11071946063285]), ('3242*', [-1439236.4311447109, 15.389392814878663]), ('4413*', [-1438639.9753450758, 11.196475818657829]), ('4232*', [-1438608.8326527807, 15.483121320673867]), ('4443*', [-1437995.8699627367, 10.188573289663607]), ('2334*', [-1437776.5013845253, 9.358111758033958]), ('2114*', [-1437570.9978407982, 10.358944961148303]), ('1124*', [-1437531.2025275892, 10.358944961148303]), ('224*', [-1437531.202527589, 9.358944961148303]), ('1413*', [-1437079.0358852695, 12.272072108405588]), ('4143*', [-1436930.1528722984, 11.272072108405588]), ('1443*', [-1436787.6085147425, 11.272072943363703]), ('3234*', [-1435757.003771188, 9.440067985664028]), ('312*', [-1435036.1196421143, 17.20809383395393]), ('1212*', [-1435036.1196421143, 18.20809383395393]), ('2132*', [-1434885.7327603253, 17.20809383395393]), ('332*', [-1434809.3054836725, 16.20359117794156]), ('1232*', [-1434809.3054836725, 17.20359117794156]), ('1142*', [-1434658.9190981272, 17.20359117794156]), ('242*', [-1434658.919098127, 16.20359117794156]), ('3342*', [-1434194.9394864773, 15.209072657030335]), ('2442*', [-1434044.5531009324, 15.209072657030335]), ('4112*', [-1433867.4145106508, 17.29809383395393]), ('132*', [-1433854.2160095149, 17.29809383395393]), ('42*', [-1433854.2160095146, 16.29809383395393]), ('4312*', [-1433626.1640805104, 16.302645101108837]), ('4332*', [-1433482.8981207006, 15.312092969513923]), ('3412*', [-1433476.531516181, 16.302645101108837]), ('3432*', [-1433333.2655563713, 15.312092969513923]), ('2214*', [-1433188.3575476066, 10.268948137576677]), ('2124*', [-1432834.634993155, 10.268944961148303]), ('4423*', [-1432586.8035499887, 11.11071946063285]), ('1134*', [-1432535.7535183073, 10.268943242061733]), ('234*', [-1432535.7535183069, 9.268943242061733]), ('4242*', [-1432022.168015154, 15.389392814878663]), ('3334*', [-1430569.5562447114, 9.358111758033958]), ('2434*', [-1430419.1698591663, 9.358111758033958]), ('3114*', [-1430365.082800422, 10.358944961148303]), ('324*', [-1430325.486963723, 9.358944961148303]), ('1224*', [-1430325.486963723, 10.358944961148303]), ('4234*', [-1428560.1814901738, 9.440067985664028]), ('1312*', [-1427842.9108381362, 18.20809383395393]), ('412*', [-1427842.910838136, 17.20809383395393]), ('3132*', [-1427693.2777800509, 17.20809383395393]), ('432*', [-1427617.233599929, 16.20359117794156]), ('1332*', [-1427617.2335999287, 17.20359117794156]), ('2142*', [-1427542.8908982612, 17.20809383395393]), ('342*', [-1427467.6010355994, 16.20359117794156]), ('1242*', [-1427467.601035599, 17.20359117794156]), ('4342*', [-1427005.947150934, 15.209072657030335]), ('3442*', [-1426856.314586605, 15.209072657030335]), ('142*', [-1426666.931571829, 17.29809383395393]), ('4412*', [-1426291.140245772, 16.302645101108837]), ('4432*', [-1426148.5924155961, 15.312092969513923]), ('3214*', [-1426004.4107673392, 10.268948137576677]), ('2224*', [-1425995.1914461541, 10.256557578695377]), ('2314*', [-1425854.0243817943, 10.268948137576677]), ('3124*', [-1425652.4612694387, 10.268944961148303]), ('2134*', [-1425502.0743876493, 10.268944961148303]), ('334*', [-1425355.077956781, 9.268943242061733]), ('1234*', [-1425355.0779567808, 10.268943242061733]), ('114*', [-1425204.6915712361, 10.26894324206173]), ('24*', [-1425204.691571236, 9.26894324206173]), ('4334*', [-1423398.7363706802, 9.358111758033958]), ('3434*', [-1423249.1038063508, 9.358111758033958]), ('4114*', [-1423195.2878623903, 10.358944961148303]), ('1324*', [-1423155.890502313, 10.358944961148303]), ('424*', [-1423155.8905023127, 9.358944961148303]), ('1412*', [-1420685.758445693, 18.20809383395393]), ('4132*', [-1420536.8754327223, 17.20809383395393]), ('1432*', [-1420461.2124288362, 17.20359117794156]), ('3142*', [-1420387.2423746365, 17.20809383395393]), ('442*', [-1420312.329907146, 16.20359117794156]), ('1342*', [-1420312.329907146, 17.20359117794156]), ('4442*', [-1419704.1075699234, 15.209072657030335]), ('4214*', [-1418856.4739721308, 10.268948137576677]), ('3224*', [-1418847.3008633729, 10.256557578695377]), ('3314*', [-1418706.8414078015, 10.268948137576677]), ('2324*', [-1418696.914477828, 10.256557578695377]), ('2414*', [-1418556.4550222564, 10.268948137576677]), ('4124*', [-1418506.2886432235, 10.268944961148303]), ('3134*', [-1418356.6555851377, 10.268944961148303]), ('1334*', [-1418210.3959831242, 10.268943242061733]), ('434*', [-1418210.3959831237, 9.268943242061733]), ('1114*', [-1418206.2687033494, 11.268944961148302]), ('214*', [-1418206.268703349, 10.268944961148302]), ('34*', [-1418060.7634187948, 9.26894324206173]), ('124*', [-1418060.7634187948, 10.26894324206173]), ('4434*', [-1416114.9781605748, 9.358111758033958]), ('1424*', [-1416022.2320940855, 10.358944961148303]), ('4142*', [-1413267.461989279, 17.20809383395393]), ('1442*', [-1413192.9250252433, 17.20359117794156]), ('4224*', [-1411735.2395316928, 10.256557578695377]), ('4314*', [-1411595.484137976, 10.268948137576677]), ('3324*', [-1411585.6069673637, 10.256557578695377]), ('3414*', [-1411445.851573647, 10.268948137576677]), ('2424*', [-1411435.2205818188, 10.256557578695377]), ('4134*', [-1411247.0536437728, 10.268944961148303]), ('1434*', [-1411101.5271772128, 10.268943242061733]), ('314*', [-1411097.4205856877, 10.268944961148302]), ('1214*', [-1411097.4205856875, 11.268944961148302]), ('4*', [-1410952.6446555236, 9.26894324206173]), ('134*', [-1410952.6446555236, 10.26894324206173]), ('4324*', [-1404509.9453330499, 10.256557578695377]), ('4414*', [-1404370.890472034, 10.268948137576677]), ('3424*', [-1404360.3127687208, 10.256557578695377]), ('414*', [-1404024.2060163158, 10.268944961148302]), ('1314*', [-1404024.2060163156, 11.268944961148302]), ('14*', [-1403880.155784607, 10.26894324206173]), ('4424*', [-1397320.8683760006, 10.256557578695377]), ('1414*', [-1396986.446379831, 11.268944961148302])]

bounding_box = np.array([[-1.5e6, -1.39e6], [0, 30]])

compares = [('Truth', [74.15495896339417, 0]), ('Dirac 0', [24.367187023162842, 0.3009274397414072]), ('Dirac 0.1', [25.151644229888916, 0.26517950151114317]), ('Dirac 0.2', [49.12092685699463, 0.16426632211119754]), ('Gaussian 0', [38.717007875442505, 0.14972084714714468]), ('Gaussian 0.05', [44.45709490776062, 0.0006497689800423529]), ('Gaussian 0.1', [59.4831109046936, 0]), ('Uniform 0', [31.752576112747192, 0.15037592784433365]), ('Uniform 0.05', [46.60546684265137, 0.15037592784433365]), ('Uniform 0.1', [39.49303913116455, 0.0])]

# drawCompares(compares)

# if True:
#     exit()

c_start = time.time()

start_state_index = mdp.states.index(start_state)

distributions = []
    
distStart = []
for i in range(len(mdp.states)):
    distStart.append(1 if i == start_state_index else 0)
distributions.append(distStart)

distributions.append(uniform(mdp))
# distributions.append(gaussian(mdp, center_state=start_state, sigma=4))
# distributions.append(gaussian(mdp, center_state=start_state, sigma=10))
# distributions.append(gaussian(mdp, center_state=target_state, sigma=4))

checkin_periods=[1, 2, 3, 4, 5]
chain_length=4
do_filter = True
margin = 0.0

name = "c"+str(checkin_periods[-1]) + "-l" + str(chain_length)
name += "-uniform"
if do_filter:
    name += "-filtered"
    if margin > 0:
        name += "-margin" + "{:.2f}".format(margin)
    else:
        name += "-margin0.0"

costs, start_state_costs = calculateChainValues(grid, mdp, discount, start_state, target_state, 
    checkin_periods=checkin_periods, 
    # execution_cost_factor=1, 
    # checkin_costs={2: 10, 3: 5, 4: 2}, 
    chain_length=chain_length,
    do_filter = do_filter, 
    distributions=distributions, 
    margin=margin, 
    bounding_box=bounding_box)

is_efficient = calculateParetoFront(start_state_costs)

c_end = time.time()
print("Chain evaluation time:", c_end - c_start)

error = calculateError(start_state_costs, is_efficient, TRUTH, bounding_box)
print("Error from true Pareto:",error)

drawChainsParetoFront(start_state_costs, is_efficient, TRUTH, TRUTH_COSTS, name)

# print("All costs:",start_state_costs)

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