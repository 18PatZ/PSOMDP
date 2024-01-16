import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation

from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
from matplotlib.font_manager import FontProperties
from matplotlib import rc

import colorsys

from mdp_def import TYPE_STATE, TYPE_WALL, TYPE_GOAL, stateToStr, fourColor

def drawSchedulePolicy(grid, start_state, target_state, compMDPs, sched, name):

    policies = sched.opt_policies
    values = sched.pi_exec_data[0]
    # all_values = sched.opt_values
    sequence = sched.strides

    current_state = start_state
    stage = 0

    max_value = None
    min_value = None

    if len(values) > 0:
        min_value = min(values.values())
        max_value = max(values.values())

    G = nx.MultiDiGraph()

    for state in compMDPs[sequence[0]].states:
        label = "\\nS" if state == start_state else "\\nG" if state == target_state else ''
        G.add_node(state, label=label)

    #'''
    for y in range(len(grid)):
        for x in range(len(grid[y])):
            state = (x, y)
            state_type = grid[y][x]

            if state_type == TYPE_WALL:
                G.add_node(state, label='')
    #'''

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
            #color = "blue"
            color = "white"
            G.add_edge(current_state, end, prob=probability, color=color, fontcolor=color, headclip=False, tailclip=False)
            #G.add_edge(current_state, end, prob=probability, label=f"{action}: " + "{:.2f}".format(probability), color=color, fontcolor=color)
            
            current_state = end
            if stage < len(sequence) - 1:
                stage += 1
        else:
            break

    drawGraph(grid, values, min_value, max_value, G, name)

def drawGraph(grid, values, min_value, max_value, G, name):
    # Build plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    layout = {}

    ax.clear()

    borderColor = '#333333'
    #wallColor = '#6e6e6e:#333333'
    wallColor = '#6e6e6e'

    # G.graph['edge'] = {'arrowsize': '1.0', 'fontsize':'10', 'penwidth': '5'}
    # G.graph['graph'] = {'scale': '3', 'splines': 'true'}
    G.graph['edge'] = {'arrowsize': '1.0', 'fontsize':'10', 'penwidth': '10'}
    # G.graph['graph'] = {'scale': '3', 'splines': 'true'}
    G.graph['graph'] = {'scale': '3', 'splines': 'curved', 'outputorder': 'nodesfirst', 
        'overlap': 'scalexy', 'sep': '0',
        'pad': '0.75', 'bgcolor': wallColor}

    A = to_agraph(G)

    A.node_attr['style']='filled'

    for node in G.nodes():
        layout[node] = (node[0], -node[1])

        state_type = grid[node[1]][node[0]]

        n = A.get_node(node)
        n.attr['color'] = borderColor#fourColor(node)
        n.attr['shape'] = 'square'
        n.attr['fixedsize'] = True
        n.attr['height'] = n.attr['width'] = 2
        n.attr['penwidth'] = 3
        n.attr['fontsize'] = 50
        n.attr['fontcolor'] = 'white'
        n.attr['fontname'] = "Helvetica bold"
        

        color = None
        if state_type == TYPE_WALL:
            color = wallColor#"#6a0dad"
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

    m = 0.5
    for k,v in layout.items():
        A.get_node(k).attr['pos']='{},{}!'.format(v[0]*m,v[1]*m)

    #A.layout('dot')
    A.layout(prog='neato')
    A.draw(name + '.png')
    # A.draw(name + '.pdf')

# def drawGraph(grid, values, min_value, max_value, G, name):
#     # Build plot
#     fig, ax = plt.subplots(figsize=(8, 8))
    
#     layout = {}

#     ax.clear()
#     labels = {}
#     edge_labels = {}
#     color_map = []

#     # G.graph['edge'] = {'arrowsize': '1.0', 'fontsize':'10', 'penwidth': '5'}
#     # G.graph['graph'] = {'scale': '3', 'splines': 'true'}
#     G.graph['edge'] = {'arrowsize': '1.0', 'fontsize':'10', 'penwidth': '5'}
#     G.graph['graph'] = {'scale': '3', 'splines': 'true'}

#     A = to_agraph(G)

#     A.node_attr['style']='filled'

#     for node in G.nodes():
#         labels[node] = f"{stateToStr(node)}"

#         layout[node] = (node[0], -node[1])

#         state_type = grid[node[1]][node[0]]

#         n = A.get_node(node)
#         n.attr['color'] = fourColor(node)

#         if state_type != TYPE_WALL:
#             n.attr['xlabel'] = "{:.4f}".format(values[node])

#         color = None
#         if state_type == TYPE_WALL:
#             color = "#6a0dad"
#         elif min_value is None and state_type == TYPE_GOAL:
#             color = "#00FFFF"
#         elif min_value is None:
#             color = "#FFA500"
#         else:
#             value = values[node]
#             frac = (value - min_value) / (max_value - min_value)
#             hue = frac * 250.0 / 360.0 # red 0, blue 1

#             col = colorsys.hsv_to_rgb(hue, 1, 1)
#             col = (int(col[0] * 255), int(col[1] * 255), int(col[2] * 255))
#             color = '#%02x%02x%02x' % col

#         n.attr['fillcolor'] = color

#         color_map.append(color)

#     for s, e, d in G.edges(data=True):
#         edge_labels[(s, e)] = "{:.2f}".format(d['prob'])

#     # Set the title
#     #ax.set_title("MDP")

#     #plt.show()
#     m = 0.7#1.5
#     for k,v in layout.items():
#         A.get_node(k).attr['pos']='{},{}!'.format(v[0]*m,v[1]*m)

#     #A.layout('dot')
#     A.layout(prog='neato')
#     A.draw(name + '.png')#, prog="neato")
#     # A.draw(name + f'-{c_stage}' '.pdf')#, prog="neato")
#     # A.draw(name + '.pdf')#, prog="neato")
#     # c_stage += 1