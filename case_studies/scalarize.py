import sys
 
# adding above folder to the system path
sys.path.insert(0, '../')

from figure import *

import numpy as np
import math

name = "pareto-c3-l4-truth-recc_no-alpha_-step2"#"pareto-c3-l4-uniform-e_no-alpha_-filtered-margin0.000-step4"
schedules, is_efficient, optimistic_front, realizable_front = loadDataChains(name, outputDir="../output")

print("Total schedules:", len(schedules))
n = 0
d = 0
for i in range(len(is_efficient)):
    if is_efficient[i]:
        n += 1
        # print("N",schedules[i].name)
    else:
        d += 1
        # print("D",schedules[i].name)
print("Total non-dominated:", n)
print("Total dominated:", d)


def scalarize(sched, alpha):
    point = sched.upper_bound[0]
    checkin_cost = point[1]
    exec_cost = point[0]
    scalarized = alpha * exec_cost + (1 - alpha) * checkin_cost
    return (checkin_cost, scalarized)


font = setPlotStyle()
fig, ax = plt.subplots()




alpha = 0.76

xVals = {"D": [], "N": [], "rD": [], "rN": []}
yVals = {"D": [], "N": [], "rD": [], "rN": []}
labels = {"D": [], "N": [], "rD": [], "rN": []}

for i in range(len(is_efficient)):
    sched = schedules[i]
    x, y = scalarize(sched, alpha)

    key = "N" if is_efficient[i] else "D"
    if "(" in sched.name:
        key = "r" + key
    
    xVals[key].append(x)
    yVals[key].append(y)
    labels[key].append(sched.name)

size = 1
ax.scatter(xVals["D"], yVals["D"], c="orange", s=size, marker='.', linewidths=0)    
ax.scatter(xVals["N"], yVals["N"], c="green", s=size, marker='.', linewidths=0)

# ax.scatter(xVals["rD"], yVals["rD"], c="purple", s=size)    
# ax.scatter(xVals["rN"], yVals["rN"], c="blue", s=size)
ax.scatter(xVals["rD"], yVals["rD"], c="orange", s=size)    
ax.scatter(xVals["rN"], yVals["rN"], c="green", s=size)

for key in ["D", "N"]:
    offset = (0, -20)
    if key == "D":
        offset = (-30, 30)
    for i in range(len(labels[key])):
        l = labels[key][i]
        if len(l) == 2 or l == "332*":
            ax.annotate(translateLabel(l), 
            xy=(xVals[key][i], yVals[key][i]), xycoords='data', xytext=offset, textcoords='offset points',
            arrowprops=dict(arrowstyle="->", color="black"), 
            color="black",fontsize=8)

# scales = [
#     # (0.25, "red", "orange"),
#     (0.5, "green", "springgreen"), 
#     # (0.75, "blue", "skyblue")
# ]

# for scale in scales:

#     alpha = scale[0]

#     xVals = {"S": [], "rS": []}
#     yVals = {"S": [], "rS": []}
#     labels = {"S": [], "rS": []}

#     for i in range(len(is_efficient)):
#         sched = schedules[i]
#         x, y = scalarize(sched, alpha)

#         key = "S"
#         if "(" in sched.name:
#             key = "r" + key
        
#         xVals[key].append(x)
#         yVals[key].append(y)
#         labels[key].append(sched.name)

#     ax.scatter(xVals["rS"], yVals["rS"], c=scale[2], s=1)
#     ax.scatter(xVals["S"], yVals["S"], c=scale[1], s=1)  

#     for i in range(len(labels["S"])):
#         l = labels["S"][i]
#         ax.annotate(translateLabel(l), 
#         xy=(xVals["S"][i], yVals["S"][i]), xycoords='data', xytext=(15, -15), textcoords='offset points',
#         arrowprops=dict(arrowstyle="->", color="black"), 
#         color="black",fontsize=8)



name="scalarization_" + name
# bounding_box=np.array([[96, 103], [-625.75, -624.75]])#None
# bounding_box=np.array([[96, 103], [-468.1, -467.2]])
bounding_box=None#np.array([[96, 103], [-600, -400]])
x_offset=0
x_scale=1
outputDir="../output"

plt.xlabel(r"\textbf{Checkin Cost}", fontproperties=font, fontweight='bold')
plt.ylabel(r"\textbf{Scalarized Cost}", fontproperties=font, fontweight='bold')
#plt.title(title)

if bounding_box is not None:
    plt.xlim((bounding_box[0] + x_offset) * x_scale)
    plt.ylim(bounding_box[1])

#plt.gcf().set_size_inches(10, 10)
plt.gcf().set_size_inches(10, 7)
plt.subplots_adjust(top=0.99, right=0.99)
#plt.savefig(f'output/{name}.pdf', format="pdf",  pad_inches=0.2, dpi=600)
plt.savefig(f'{outputDir}/{name}.pdf', format="pdf",  pad_inches=0.0, dpi=600)
# plt.savefig(f'output/{name}.png', bbox_inches='tight', pad_inches=0.5, dpi=300)
# plt.show()

plt.close(fig)
