
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import rc
import json

def translateLabel(label):
    star_index = label.index('*')
    prefix = label[:star_index-1]
    tail = label[star_index-1]
    suffix = label[star_index+1:]
    label = prefix + "$\overline{" + tail + "}$" + suffix
    #label = label[:-2] + "$\overline{" + label[-2] + "}$"
    # label = label[:-2] + "$\dot{" + label[-2] + "}$"
    
    return label



def addXY(point, x, y, x_offset=0, x_scale=1):
    x.append((point[0] + x_offset) * x_scale)
    y.append(point[1])


def lines(ax, points, color, x_offset=0, x_scale=1):
    x = [(point[1][0] + x_offset) * x_scale for point in points]
    y = [point[1][1] for point in points]
    
    ax.plot(x, y, c=color)


def manhattan_lines(ax, points, color, bounding_box, x_offset=0, x_scale=1, linestyle=None, linethickness = 2.5, fillabove=True, fillcolor=None, fillalpha=1.0):
    x = []
    y = []

    xmax = bounding_box[0][1]
    ymax = bounding_box[1][1]
    
    if len(points) > 0:
        point = points[0][1]
        addXY((point[0], ymax), x, y, x_offset, x_scale)

    for i in range(len(points)):
        point = points[i][1]
        
        addXY(point, x, y, x_offset, x_scale)

        if i < len(points) - 1:
            next_point = points[i+1][1]

            addXY((next_point[0], point[1]), x, y, x_offset, x_scale)

    if len(points) > 0:
        point = points[-1][1]
        addXY((xmax + x_offset, point[1]), x, y, x_offset, x_scale)

    
    if color is not None:
        if linestyle is None:
            ax.plot(x, y, c=color, linewidth=linethickness)
        else:
            ax.plot(x, y, c=color, linestyle=linestyle, linewidth=linethickness)

    if fillabove:
        addXY((xmax, ymax), x, y, x_offset, x_scale)
        if fillcolor is None:
            fillcolor = color
        ax.fill(x, y, facecolor=fillcolor, alpha=fillalpha)


def drawL(ax, points, color, bounding_box, face_color, x_offset=0, x_scale=1):
    x = []
    y = []

    xmax = bounding_box[0][1]
    ymax = bounding_box[1][1]

    addXY([points[0][1][0], ymax], x, y, x_offset, x_scale)
    addXY(points[0][1], x, y, x_offset, x_scale)
    addXY([max(points[0][1][0], points[2][1][0]), max(points[0][1][1], points[2][1][1])], x, y, x_offset, x_scale)
    addXY(points[2][1], x, y, x_offset, x_scale)
    addXY([xmax, points[2][1][1]], x, y, x_offset, x_scale)
    addXY([xmax, ymax], x, y, x_offset, x_scale)

    #ax.plot(x, y, c=color, linestyle="solid", linewidth=1.33, alpha=1.0)
    ax.fill(x, y, facecolor=face_color, alpha=0.2)
    #ax.fill(x, y, facecolor=face_color, alpha=0.3, hatch = "////", edgecolor="white")




def drawLdominated(ax, points, color, bounding_box, face_color, x_offset=0, x_scale=1):
    x = []
    y = []

    xmax = bounding_box[0][1]
    ymax = bounding_box[1][1]

    addXY([points[0][1][0], ymax], x, y, x_offset, x_scale)
    addXY(points[0][1], x, y, x_offset, x_scale)
    addXY([max(points[0][1][0], points[2][1][0]), max(points[0][1][1], points[2][1][1])], x, y, x_offset, x_scale)
    addXY(points[2][1], x, y, x_offset, x_scale)
    addXY([xmax, points[2][1][1]], x, y, x_offset, x_scale)
    addXY([xmax, ymax], x, y, x_offset, x_scale)

    ax.plot(x, y, c=color, linestyle="solid", linewidth=0.33, alpha=0.05)
    ax.fill(x, y, facecolor=face_color, alpha=0.10)
    #ax.fill(x, y, facecolor=face_color, alpha=1.00, hatch = "////", edgecolor="white")



def scatter(ax, points, doLabel, color, lcolor, arrows=False, x_offset = 0, x_scale=1, loffsets={}, elide=False):
    x_full = [(point[1][0] + x_offset) * x_scale for point in points]
    y_full = [point[1][1] for point in points]
    labels_full = [point[0] for point in points]

    deltax = (max(x_full) - min(x_full)) / 1.25 # Since the labels extend horizontally, we penalze the horizontal more
    deltay = (max(y_full) - min(y_full)) 

    if elide:
        # elide some labels, when a bunch appear and they would overlap
        x = []
        y = []
        labels = []
        elide_dist = 0.002
        for i in range(len(labels_full)):
            nearest_d2 = elide_dist + 1.0
            for j in range(len(labels)):
                nearest_d2 = min(nearest_d2, ((1.0/deltax)*(x_full[i] - x[j]))**2 + ((1.0/deltay)*(y_full[i] - y[j]))**2 )
            if nearest_d2 >= elide_dist:
                # Should NOT drop i:
                x.append(x_full[i])
                y.append(y_full[i])
                labels.append(labels_full[i])
    else:
        x = x_full
        y = y_full
        labels = labels_full

    
    #ax.scatter(x, y, c=color, s=15)

    if doLabel:
        for i in range(len(labels)):
            l = labels[i]
            if not arrows:
                ax.annotate(translateLabel(l),
                    xy=(x[i], y[i]), xycoords='data',
                    xytext=(5, 5), textcoords='offset points',
                    color=lcolor)
            else:
                offset = (-45, -30)

                if l in loffsets:
                    offset = (offset[0] + loffsets[l][0], offset[1] + loffsets[l][1])

                ax.annotate(translateLabel(l), 
                    xy=(x[i], y[i]), xycoords='data',
                    xytext=offset, textcoords='offset points',
                    arrowprops=dict(arrowstyle="->", color=lcolor), 
                    color=lcolor,fontsize=12)


def drawParetoFront(points, indices, is_efficient, realizable_front, true_front, true_costs, name, title, bounding_box, prints, x_offset=0, x_scale=1, loffsets={}):
    plt.style.use('seaborn-whitegrid')

    if prints:
        print("\n-----------\nDrawing",name,"\n-----------\n")

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

    # points is a list of point tuples, each tuple is: (string name of schedule, [execution cost, checkin cost])
    # each schedule has 3 points (pi*, pi^c, and bottom corner of L)
    # all points from all schedules are together in 1D points array
    # indices array gives indices of schedule's points in points array for each schedule

    points_nondominated = []
    points_dominated = []
    for i in range(len(points)):
        if is_efficient[i]:
            points_nondominated.append(points[i])
        else:
            points_dominated.append(points[i])


    num_efficient_schedules = 0
    is_efficient_schedules = []
    for i in range(len(indices)):
        point_indices = indices[i][1]

        efficient = False
        for j in point_indices:
            if is_efficient[j]: # at least one of the 3 points are in the front, so the schedule is in the front
                efficient = True
                num_efficient_schedules += 1
                break
        is_efficient_schedules.append(efficient)

    points_nondominated.sort(key = lambda point: point[1][0])

    if prints:
        print("Non-dominated points:")
        for point in points_nondominated:
            print("  ", point[0])

    if prints:
        print(len(points_dominated),"dominated points out of",len(points),"|",len(points_nondominated),"non-dominated")
        print(len(indices)-num_efficient_schedules,"dominated schedules out of",len(indices),"|",num_efficient_schedules,"non-dominated")

    if prints:
        print("Pareto front:",points_nondominated)
        print("Realizable front:",realizable_front)
    
    fig, ax = plt.subplots()

    # draw truth (old)
    # if true_costs is not None:
    #     scatter(ax, true_costs, doLabel=False, color="gainsboro", lcolor="gray", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)

    # if true_front is not None:
    #     manhattan_lines(ax, true_front, color="green", bounding_box=bounding_box, x_offset=x_offset, x_scale=x_scale)
    #     scatter(ax, true_front, doLabel=False, color="green", lcolor="green", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)

    # scatter(ax, points_dominated, doLabel=False, color="orange", lcolor="gray", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)
    
    # draw optimistic front
    manhattan_lines(ax, points_nondominated, color="#e6c7c7", bounding_box=bounding_box, x_offset=x_offset, x_scale=x_scale)
    #scatter(ax, points_nondominated, doLabel=True, color="#aa3333", lcolor="black", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets, elide=True)

    # draw realizable front
    if realizable_front is not None:
        manhattan_lines(ax, realizable_front, color="#aa3333", bounding_box=bounding_box, x_offset=x_offset, x_scale=x_scale, fillcolor="#ededed")
        scatter(ax, realizable_front, doLabel=True, color="blue", lcolor="black", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets, elide=True)



    # earlier draws (usually) are in the back
    for i in range(len(is_efficient_schedules)):
        if not is_efficient_schedules[i]: # dominated schedule
            schedule_points = []
            for j in indices[i][1]:
                schedule_points.append(points[j])

            drawLdominated(ax, schedule_points, bounding_box=bounding_box, color="#222222", face_color="#aaaaaa", x_offset=x_offset, x_scale=x_scale)
            #scatter(ax, schedule_points, doLabel=False, color="orange", lcolor="gray", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)

    for i in range(len(is_efficient_schedules)):
        if is_efficient_schedules[i]: # non-dominated schedule
            schedule_points = []
            for j in indices[i][1]:
                schedule_points.append(points[j])
        
            #drawL(ax, schedule_points, bounding_box=bounding_box, color="#ff2222", face_color="#ff2222", x_offset=x_offset, x_scale=x_scale)
            #scatter(ax, schedule_points, doLabel=True, color="red", lcolor="gray", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)


    
    plt.xlabel(r"\textbf{Execution Cost}", fontproperties=font, fontweight='bold')
    plt.ylabel(r"\textbf{Checkin Cost}", fontproperties=font, fontweight='bold')
    #plt.title(title)


    plt.xlim((bounding_box[0] + x_offset) * x_scale)
    plt.ylim(bounding_box[1])

    plt.gcf().set_size_inches(10, 10)
    plt.savefig(f'output/{name}.pdf', format="pdf",  pad_inches=0.2, dpi=600)
    # plt.savefig(f'output/{name}.png', bbox_inches='tight', pad_inches=0.5, dpi=300)
    # plt.show()


def loadDataChains(filename):
    with open(f'output/data/{filename}.json', "r") as file:
        jsonStr = file.read()
        obj = json.loads(jsonStr)

        truth = obj['Truth'] if 'Truth' in obj else None
        truth_costs = obj['Truth Costs'] if 'Truth Costs' in obj else None
        realizable_front = obj['Realizable Front'] if 'Realizable Front' in obj else None

        return (obj['Points'], obj['Indices'], obj['Efficient'], realizable_front, truth, truth_costs)





if __name__ == "__main__":
    
    #bounding_box = np.array([[-1.5e6, -1.39e6], [0.0001, 30]])
    #bounding_box = np.array([[-1.55e6, -1.25e6], [0.0001, 30]])
    #bounding_box = np.array([[-1.50e6, -1.10e6+15], [0.0000+3.5, 25+1.5]])
    #bounding_box = np.array([[-1.50e6, -1.40e6+15], [0.0000+3.5, 25+1.5]])
    bounding_box = np.array([[-1.560e6, -1.10e6+15], [0.0000+3.5, 25+1.5]])

    x_offset = 1.56e6
    x_scale = 1/1000

    names = [
        "pareto-c4-l4-uniform-filtered-margin0.040-step1",
        "pareto-c4-l4-uniform-filtered-margin0.040-step2",
        "pareto-c4-l4-uniform-filtered-margin0.040-step3",
        "pareto-c4-l4-uniform-filtered-margin0.040-step4",
    ]

    label_offsets = {
        # "2321*": (-5, 0),
        # "43334*": (-10, 0),
        # "2232121*": (-10, 0),
    }

    for name in names:
        points, indices, is_efficient, realizable_front, truth, truth_costs = loadDataChains(name)
        drawParetoFront(points, indices, is_efficient, realizable_front, 
            true_front = None, #truth, 
            true_costs = None, #truth_costs, 
            name=name, title="", bounding_box=bounding_box, prints=True, x_offset=x_offset, x_scale=x_scale, loffsets=label_offsets)