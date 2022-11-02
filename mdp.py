
from tabnanny import check
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation

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

def extendCompositeMDP(mdp, discount, prevPeriodMDP):
    compMDP = MDP([], [], {}, {}, [])

    compMDP.states = mdp.states.copy()
    compMDP.terminals = mdp.terminals.copy()

    for action_sequence in prevPeriodMDP.actions:
        for action in mdp.actions:
            extended_action_sequence = action_sequence + (action,) # extend tuple
            compMDP.actions.append(extended_action_sequence)

    for state in prevPeriodMDP.transitions.keys():
        compMDP.transitions[state] = {}
        for prev_action_sequence in prevPeriodMDP.transitions[state].keys():
            for end_state in prevPeriodMDP.transitions[state][prev_action_sequence].keys():
                # looping through every state-actionsequence-state chain in the previous step MDP
                # now extend chain by one action by multiplying transition probability of previous chain end state to new end state through action

                for action in mdp.actions:
                    prob_chain = prevPeriodMDP.transitions[state][prev_action_sequence][end_state]

                    if end_state in mdp.transitions and action in mdp.transitions[end_state]:
                        for new_end_state in mdp.transitions[end_state][action].keys():
                            prob_additional = mdp.transitions[end_state][action][new_end_state]

                            extended_action_sequence = prev_action_sequence + (action,)

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
            prev_reward = prevPeriodMDP.rewards[state][prev_action_sequence]

            for action in mdp.actions:
                if action in mdp.rewards[end_state]:
                    # extend chain by one action
                    extended_action_sequence = prev_action_sequence + (action,)

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



def branchAndBound(grid, base_mdp, discount, checkin_period, threshold, max_iterations):

    compMDP = convertSingleStepMDP(base_mdp)
    pruned_action_set = {state: [action for action in compMDP.actions] for state in base_mdp.states}

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
            compMDP = extendCompositeMDP(base_mdp, discount, compMDP)
            # pruned_action_set = compMDP.actions

            for state in base_mdp.states:
                extended_action_set = []
                for prev_action_sequence in pruned_action_set[state]:
                    for action in base_mdp.actions:
                        extended_action_set.append(prev_action_sequence + (action,))
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
        policy, state_values, q_values = qValueIteration(grid, compMDP, discount_input, threshold, max_iterations, pruned_action_set)
        lowerBound = state_values

        upperBounds.append(upperBound)
        lowerBounds.append(q_values)

        pr = {}

        tot = 0
        for state in base_mdp.states:
            toPrune = []
            for action in pruned_action_set[state]:
                prefix = action[:t]
                # print(prefix, upperBound[state][prefix], lowerBound[state])
                if upperBound[state][prefix] < lowerBound[state]:
                    toPrune.append(prefix)

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


def paper2An(n):
    goalActionReward = 10000
    noopReward = 0#-1
    wallPenalty = -300000
    movePenalty = -1

    moveProb = 0.9
    discount = math.sqrt(0.99)

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


def splitterGrid():
    goalActionReward = 10000
    noopReward = 0#-1
    wallPenalty = -50000
    movePenalty = -1

    moveProb = 0.9
    discount = math.sqrt(0.9)

    grid = []
    
    rows = 8
    p1 = 2
    p2 = 3

    maxN = 3
    nL = 0
    nR = 0

    for i in range(rows):
        row = None
        if nL < maxN and (rows-i) % p1 == 1:
            nL += 1
            row = [0, 1, 0, 1, 0, 1]
        else:
            row = [0, 0, 0, 0, 0, 0]

        row.append(2 if i == 0 else 1)

        if nR < maxN and (rows-i) % p2 == 1:
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


def run(grid, mdp, discount, start_state, checkin_period, doBranchAndBound, drawPolicy=True, drawIterations=True, outputPrefix="", doLinearProg=False):
    policy = None
    values = None
    q_values = None

    start = time.time()
    elapsed = None

    if not doBranchAndBound:
        compMDP = createCompositeMDP(mdp, discount, checkin_period)
        print("Actions:",len(mdp.actions),"->",len(compMDP.actions))

        end1 = time.time()
        print("MDP composite time:", end1 - start)

        # policy, values = valueIteration(grid, compMDP, discount, 1e-20, int(1e4))#1e-20, int(1e4))
        discount_t = pow(discount, checkin_period)
        # print("final",checkin_period,len(compMDP.actions),discount_t,1e-20, int(1e4), (len(mdp.states) * len(compMDP.actions)))

        if doLinearProg:
            policy, values = linearProgrammingSolve(grid, compMDP, discount_t)
            end2 = time.time()
            print("MDP linear programming time:", end2 - end1)
        else:
            policy, values, q_values = qValueIteration(grid, compMDP, discount_t, 1e-20, int(1e4))#1e-20, int(1e4))
            print(policy)

            end2 = time.time()
            print("MDP value iteration time:", end2 - end1)
        
        print("MDP total time:", end2 - start)
        elapsed = end2 - start

        print("Start state value:",values[start_state])

        if drawPolicy:
            draw(grid, compMDP, values, policy, True, False, "output/policy-"+outputPrefix+str(checkin_period)+("-vi" if not doLinearProg else "-lp"))
    else:
        compMDP, policy, values, q_values, ratios, upperBounds, lowerBounds, pruned, compMDPs = branchAndBound(grid, mdp, discount, checkin_period, 1e-20, int(1e4))#1e-20, int(1e4))
        print(policy)
        
        end = time.time()
        print("MDP branch and bound time:", end - start)
        print("MDP total time:", end - start)
        elapsed = end - start

        print("Start state", start_state, "value:",values[start_state])

        if drawIterations:
            for i in range(0, checkin_period-1):
                drawBNBIteration(grid, compMDPs[i], ratios, upperBounds, lowerBounds, pruned, i, "output/policy-"+outputPrefix+str(checkin_period)+"-bnb-"+str(i+1))

        if drawPolicy:
            draw(grid, compMDP, values, policy, True, False, "output/policy-"+outputPrefix+str(checkin_period)+"-bnb-f")

    # if not os.path.exists("output/"):
    #     os.makedirs("output/")

    # draw(grid, compMDP, values, {}, False, True, "output/multi"+str(checkin_period))
    # draw(grid, compMDP, values, policy, True, False, "output/policy"+str(checkin_period))


    # s = compMDP.states[0]
    # for action in compMDP.transitions[s].keys():
    #     for end_state in compMDP.transitions[s][action].keys():
    #         print(s,action,"->",end_state,"is",compMDP.transitions[s][action][end_state])

    return values[start_state], elapsed


def runFig2Ratio(wallMin, wallMax, increment = 1):
    results = []
    for numWalls in range(wallMin, wallMax+increment, increment):
        grid, mdp, discount, start_state = paper2An(numWalls)

        pref = "paperFig2-" + str(numWalls) + "w-"
        value2, elapsed2 = run(grid, mdp, discount, start_state, checkin_period=2, doBranchAndBound=True, drawPolicy=True, drawIterations=False, outputPrefix=pref)
        value3, elapsed3 = run(grid, mdp, discount, start_state, checkin_period=3, doBranchAndBound=True, drawPolicy=True, drawIterations=False, outputPrefix=pref)

        results.append((numWalls, value2, value3))

    print("\n\n ===== RESULTS ===== \n\n")
    for r in results:
        print("Length " + str(3 + r[0] * 3 + 1) + " (" + str(r[0]) + " walls):")
        print("\tValue k=2:", r[1])
        print("\tValue k=3:", r[2])
        print("\tRatio k3/k2:", r[2]/r[1])
        print("")


def runCheckinSteps(checkinMin, checkinMax, increment = 1):
    grid, mdp, discount, start_state = paper2An(3)
    times = []

    for checkin_period in range(checkinMin, checkinMax+increment, increment):
        print("\n\n ==== CHECKIN PERIOD " + str(checkin_period)  + " ==== \n\n")
        time = 0
        for i in range(0, 1):
            value, elapsed = run(grid, mdp, discount, start_state, checkin_period, doBranchAndBound=False, doLinearProg=True) # LP
            time += elapsed
        time /= 3

        times.append(time)
        print("")
        print("Average",time)
        print(times)



# start = time.time()

# grid, mdp, discount, start_state = paper2An(3)

# end = time.time()
# print("MDP creation time:", end - start)

# checkin_period = 4

# #run(grid, mdp, discount, start_state, checkin_period, doBranchAndBound=False, doLinearProg=False) # VI
# #run(grid, mdp, discount, start_state, checkin_period, doBranchAndBound=True, doLinearProg=False) # BNB
# run(grid, mdp, discount, start_state, checkin_period, doBranchAndBound=False, doLinearProg=True) # LP

runCheckinSteps(1, 20)

#runFig2Ratio(30, 100, 10)