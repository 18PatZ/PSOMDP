
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import rc
import json
from schedule import ScheduleBounds

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
    
    # addXY([max(points[0][1][0], points[2][1][0]), max(points[0][1][1], points[2][1][1])], x, y, x_offset, x_scale)
    # addXY(points[2][1], x, y, x_offset, x_scale)
    for i in range(0, len(points)-1):
        addXY([max(points[i][1][0], points[i+1][1][0]), max(points[i][1][1], points[i+1][1][1])], x, y, x_offset, x_scale)
        addXY(points[i+1][1], x, y, x_offset, x_scale)
    
    addXY([xmax, points[-1][1][1]], x, y, x_offset, x_scale)
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
            if nearest_d2 >= elide_dist or len(labels_full[i]) == 2 or labels_full[i] == "332*": # include 1, 2, 3 etc.
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


def setPlotStyle():
    plt.style.use('seaborn-whitegrid')

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

    return font


def finishPlot(font, name, bounding_box, x_offset, x_scale, outputDir="output"):
    plt.xlabel(r"\textbf{Execution Cost}", fontproperties=font, fontweight='bold')
    plt.ylabel(r"\textbf{Checkin Cost}", fontproperties=font, fontweight='bold')
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


def drawScalarizationLine(schedules, ax, anchor, alpha):
    # draw scalarization line
    pt = None
    for sched in schedules:
        if sched.name == anchor:
            pt = sched.upper_bound[0]
    if pt is not None:
        # scalarized = alpha * exec_cost + (1 - alpha) * checkin_cost
        # line is constant scalarization:
        #   C = ax + (1 - a) y
        #   y = (C - ax) / (1 - a)
        C = alpha * pt[0] + (1 - alpha) * pt[1]
        y = lambda x: (C - alpha * x) / (1 - alpha)
        pt0 = ((bounding_box[0][0] + x_offset) * x_scale, y(bounding_box[0][0]))
        pt1 = ((bounding_box[0][1] + x_offset) * x_scale, y(bounding_box[0][1]))

        ax.plot([pt0[0], pt1[0]], [pt0[1], pt1[1]], c="green", linewidth=0.5, alpha=0.5)


def drawParetoFront(schedules, is_efficient, optimistic_front, realizable_front, true_front, 
                    true_costs, name, title, bounding_box, prints, x_offset=0, x_scale=1, loffsets={}, outputDir="output"):
    
    if prints:
        print("\n-----------\nDrawing",name,"\n-----------\n")

    arrows = True
    
    font = setPlotStyle()

    # points is a list of point tuples, each tuple is: (string name of schedule, [execution cost, checkin cost])
    # each schedule has 3 points (pi*, pi^c, and bottom corner of L)
    # all points from all schedules are together in 1D points array
    # indices array gives indices of schedule's points in points array for each schedule

    scheds_nondominated = []
    scheds_dominated = []
    for i in range(len(schedules)):
        if is_efficient[i]:
            scheds_nondominated.append(schedules[i])
        else:
            scheds_dominated.append(schedules[i])
    


    # num_efficient_schedules = 0
    # is_efficient_schedules = []
    # for i in range(len(indices)):
    #     point_indices = indices[i][1]

    #     efficient = False
    #     for j in point_indices:
    #         if is_efficient[j]: # at least one of the 3 points are in the front, so the schedule is in the front
    #             efficient = True
    #             num_efficient_schedules += 1
    #             break
    #     is_efficient_schedules.append(efficient)

    # points_nondominated.sort(key = lambda point: point[1][0])
    
    if prints:
        print("Non-dominated schedules:")
        for sched in scheds_nondominated:
            print("  ", sched.name)

    if prints:
        print(len(scheds_dominated),"dominated schedules out of",len(schedules),"|",len(scheds_nondominated),"non-dominated")
        # print(len(indices)-num_efficient_schedules,"dominated schedules out of",len(indices),"|",num_efficient_schedules,"non-dominated")

    if prints:
        print("Pareto front:",scheds_nondominated)
        print("Optimistic front:",optimistic_front)
        print("Realizable front:",realizable_front)
    
    fig, ax = plt.subplots()


    

    # draw truth (old)
    # if true_costs is not None:
    #     scatter(ax, true_costs, doLabel=False, color="gainsboro", lcolor="gray", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)

    # if true_front is not None:
    #     manhattan_lines(ax, true_front, color="green", bounding_box=bounding_box, x_offset=x_offset, x_scale=x_scale)
    #     scatter(ax, true_front, doLabel=False, color="green", lcolor="green", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)

    if true_front is not None:
        truth_optimistic_front, truth_realizable_front = true_front
        if truth_optimistic_front is not None:
            manhattan_lines(ax, truth_optimistic_front, color="#c7e6d0", bounding_box=bounding_box, x_offset=x_offset, x_scale=x_scale)
        if truth_realizable_front is not None:
            manhattan_lines(ax, truth_realizable_front, color="#33ab55", bounding_box=bounding_box, x_offset=x_offset, x_scale=x_scale, fillcolor="#ededed", linethickness=0.5)
            # scatter(ax, realizable_front, doLabel=True, color="blue", lcolor="black", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets, elide=True)
            scatter(ax, truth_realizable_front, doLabel=True, color="blue", lcolor="gray", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets, elide=True)

    # scatter(ax, points_dominated, doLabel=False, color="orange", lcolor="gray", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)
    
    # draw optimistic front
    if optimistic_front is not None:
        manhattan_lines(ax, optimistic_front, color="#e6c7c7", bounding_box=bounding_box, x_offset=x_offset, x_scale=x_scale)
    #scatter(ax, points_nondominated, doLabel=True, color="#aa3333", lcolor="black", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets, elide=True)

    # draw realizable front
    if realizable_front is not None:
        pts = [pt[1] for pt in realizable_front]
        min = np.min(pts, axis=0)
        max = np.max(pts, axis=0)
        if prints:
            print("Realizable front actual bounds", [[min[0], max[0]], [min[1], max[1]]])

        manhattan_lines(ax, realizable_front, color="#aa3333", bounding_box=bounding_box, x_offset=x_offset, x_scale=x_scale, fillcolor="#ededed", linethickness=0.5)
        # manhattan_lines(ax, realizable_front, color="#aa3333", bounding_box=bounding_box, x_offset=x_offset, x_scale=x_scale, fillcolor="#c7e6d0")
        scatter(ax, realizable_front, doLabel=True, color="blue", lcolor="black", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets, elide=True)

    # if true_front is not None and truth_realizable_front is not None:
    #     manhattan_lines(ax, truth_realizable_front, color="#33ab55", bounding_box=bounding_box, x_offset=x_offset, x_scale=x_scale, fillcolor="#ededed")
    # manhattan_lines(ax, realizable_front, color="#aa3333", bounding_box=bounding_box, x_offset=x_offset, x_scale=x_scale, fillabove=False)

    # earlier draws (usually) are in the back
    for sched in scheds_dominated:
        schedule_points = []
        # schedule_points.append([sched.name, sched.upper_bound[0]])
        # for b in sched.lower_bound:
        #     schedule_points.append([sched.name, b])
        # schedule_points.append([sched.name, sched.upper_bound[-1]])
        for b in sched.upper_bound:
            schedule_points.append([sched.name, b])

        drawLdominated(ax, schedule_points, bounding_box=bounding_box, color="#222222", face_color="#aaaaaa", x_offset=x_offset, x_scale=x_scale)
        # scatter(ax, schedule_points, doLabel=True, color="orange", lcolor="gray", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets, elide=True)

    for sched in scheds_nondominated:
        schedule_points = []
            
        schedule_points.append([sched.name, sched.upper_bound[0]])
            
        if len(sched.lower_bound) > 0:
            for b in sched.lower_bound:
                schedule_points.append([sched.name, b])
            schedule_points.append([sched.name, sched.upper_bound[-1]])
    
        # drawLdominated(ax, schedule_points, bounding_box=bounding_box, color="#222222", face_color="#aaaaaa", x_offset=x_offset, x_scale=x_scale)
        #drawL(ax, schedule_points, bounding_box=bounding_box, color="#ff2222", face_color="#ff2222", x_offset=x_offset, x_scale=x_scale)
        #scatter(ax, schedule_points, doLabel=True, color="red", lcolor="gray", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)

    # alpha = 0.76
    # drawScalarizationLine(schedules, ax, anchor="(212)*", alpha=alpha)
    # drawScalarizationLine(schedules, ax, anchor="3*", alpha=alpha)
    # drawScalarizationLine(schedules, ax, anchor="1*", alpha=alpha)

    finishPlot(font, name, bounding_box, x_offset, x_scale, outputDir)
    plt.close(fig)


def drawParetoFrontSuperimposed(fronts, true_fronts, true_costs, colors, name, 
                                bounding_box, prints, x_offset=0, x_scale=1, loffsets={}, outputDir="output"):
    plt.style.use('seaborn-whitegrid')

    if prints:
        print("\n-----------\nDrawing",name,"\n-----------\n")

    arrows = True

    font = setPlotStyle()
    
    fig, ax = plt.subplots()

    # if true_front is not None:
    #     manhattan_lines(ax, true_front, color="green", bounding_box=bounding_box, x_offset=x_offset, x_scale=x_scale)
    #     scatter(ax, true_front, doLabel=False, color="green", lcolor="green", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)

    for i in range(len(fronts)):
        front_lower, front_upper = fronts[i]
        # true_front_lower, true_front_upper = true_fronts[i]
        (color_lower, color_upper) = colors[i]

        # if i == len(fronts)-1:
        #     scatter(ax, chains_dominated, doLabel=False, color="orange", lcolor="gray", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)

        if i == len(fronts)-1:
            manhattan_lines(ax, front_lower, color=color_lower, bounding_box=bounding_box, x_offset=x_offset, x_scale=x_scale)
        manhattan_lines(ax, front_upper, color=color_upper, bounding_box=bounding_box, x_offset=x_offset, x_scale=x_scale, fillcolor="#ededed")
        
        if i == len(fronts)-1:
            scatter(ax, front_upper, doLabel=True, color=color_upper, lcolor="black", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets, elide=True)
    
    
    # if true_costs is not None:
    #     truth_schedules, truth_nondominated, truth_dominated = true_costs
    #     for sched in truth_dominated:
    #         schedule_points = []
    #         for b in sched.upper_bound:
    #             schedule_points.append([sched.name, b])

    #         drawLdominated(ax, schedule_points, bounding_box=bounding_box, color="#222222", face_color="#aaaaaa", x_offset=x_offset, x_scale=x_scale)
    #         #scatter(ax, schedule_points, doLabel=False, color="orange", lcolor="gray", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)
    #     # scatter(ax, true_costs, doLabel=False, color="gainsboro", lcolor="gray", arrows=arrows, x_offset=x_offset, x_scale=x_scale, loffsets=loffsets)


    finishPlot(font, name, bounding_box, x_offset, x_scale, outputDir)
    plt.close(fig)



def loadDataChains(filename, outputDir="output"):
    with open(f'{outputDir}/data/{filename}.json', "r") as file:
        jsonStr = file.read()
        obj = json.loads(jsonStr)

        # truth = obj['Truth'] if 'Truth' in obj else None
        # truth_costs = obj['Truth Costs'] if 'Truth Costs' in obj else None
        realizable_front = obj['Realizable Front'] if 'Realizable Front' in obj else None
        optimistic_front = obj['Optimistic Front'] if 'Optimistic Front' in obj else None
        
        schedules = [ScheduleBounds(sched) for sched in obj['Schedules']]

        # realizable_front = [r for r in realizable_front if "2*" in r[0]]
        # schedules = [s for s in schedules if "2*" in s.name]

        return (schedules, obj['Efficient'], optimistic_front, realizable_front)#, truth, truth_costs)


def loadTruth(filename, outputDir="output"):
    if filename is None:
        return (None, None)

    truth_schedules, is_efficient, truth_optimistic_front, truth_realizable_front = loadDataChains(filename, outputDir)

    scheds_nondominated = []
    scheds_dominated = []
    for i in range(len(truth_schedules)):
        if is_efficient[i]:
            scheds_nondominated.append(truth_schedules[i])
        else:
            scheds_dominated.append(truth_schedules[i])

    return ((truth_optimistic_front, truth_realizable_front), (truth_schedules, scheds_nondominated, scheds_dominated))


if __name__ == "__main__":
    
    #bounding_box = np.array([[-1.5e6, -1.39e6], [0.0001, 30]])
    #bounding_box = np.array([[-1.55e6, -1.25e6], [0.0001, 30]])
    #bounding_box = np.array([[-1.50e6, -1.10e6+15], [0.0000+3.5, 25+1.5]])
    #bounding_box = np.array([[-1.50e6, -1.40e6+15], [0.0000+3.5, 25+1.5]])
    # bounding_box = np.array([[-1.560e6, -1.10e6+15], [0.0000+3.5, 25+1.5]])
    #bounding_box = np.array([[-1.5e6, -1e6], [0.0001, 30]])
    #bounding_box = np.array([[-1.56e6, -1e6], [0.0001, 30]])
    # bounding_box = np.array([[-1.56e6, -1.04e6], [5.0001, 28]])
    # bounding_box = np.array([[-1000, -900], [0.0001, 300]])
    #bounding_box = np.array([[-952.2795210898702, -949.436449420723], [97.07854114027103, 101.9800249998421]])
    # bounding_box = np.array([[-1000, -900], [0.0001, 300]])
    bounding_box = np.array([[-990, -880], [0.0001, 250]])
    
    # bounding_box = np.array([[-18, -15], [0.0001, 5]])
    # bounding_box = np.array([[-51, -44], [0.0001, 15]])

    x_offset = 1.56e6
    x_scale = 1/1000

    s = 3
    truth_name = f"pareto-c3-l8-truth-recc_no-alpha_-step{s}"#None#"pareto-c3-l4-truth_no-alpha_"

    names = [
        # "pareto-c4-l4-uniform_no-alpha_-filtered-margin0.040-step1",
        # "pareto-c4-l4-uniform_no-alpha_-filtered-margin0.040-step2",
        # "pareto-c4-l4-uniform_no-alpha_-filtered-margin0.040-step3",
        # "pareto-c4-l4-uniform_no-alpha_-filtered-margin0.040-step4",
        # "pareto-c4-l32-initial_10alpha_-filtered-margin0.000-step1",
        # "pareto-c4-l32-initial_10alpha_-filtered-margin0.000-step2",
        # "pareto-c4-l32-initial_10alpha_-filtered-margin0.000-step4",
        # "pareto-c4-l32-initial_10alpha_-filtered-margin0.000-step8",
        # "pareto-c4-l32-initial_10alpha_-filtered-margin0.000-step14",
        # "pareto-c4-l32-initial_10alpha_-filtered-margin0.000-step16",
        f"pareto-c3-l8-truth-depth-recc_no-alpha_-step{s}",
        #"pareto-c4-l4-truth_no-alpha_"
        # "pareto-c4-l4-truth_10alpha_"
    ]

    label_offsets = {
        # "2321*": (-5, 0),
        # "43334*": (-10, 0),
        # "2232121*": (-10, 0),
    }

    true_fronts, truth_schedules = loadTruth(truth_name)

    superimposed = False

    if not superimposed:
        for name in names:
            schedules, is_efficient, optimistic_front, realizable_front = loadDataChains(name)

            print("Total schedules:", len(schedules))
            n = 0
            d = 0
            for i in range(len(is_efficient)):
                if is_efficient[i]:
                    n += 1
                else:
                    d += 1
            print("Total non-dominated:", n)
            print("Total dominated:", d)

            scheds = set()
            unique = 0
            scheds2 = set()
            unique2 = 0
            unique3 = 0

            if optimistic_front is not None:
                for point in optimistic_front:
                    if point[0] not in scheds:
                        scheds.add(point[0])
                        unique += 1
                        unique3 += 1
            for point in realizable_front:
                if point[0] not in scheds2:
                    scheds2.add(point[0])
                    unique2 += 1
                if point[0] not in scheds:
                    scheds.add(point[0])
                    unique3 += 1

            print("Unique schedules in lower front:", unique)
            print("Unique schedules in upper front:", unique2)
            print("Unique schedules overall:", unique3)

            drawParetoFront(schedules, is_efficient, optimistic_front, realizable_front, 
                true_front = true_fronts, #truth, 
                true_costs = truth_schedules, #truth_costs, 
                name=name, title="", bounding_box=bounding_box, prints=True, x_offset=x_offset, x_scale=x_scale, loffsets=label_offsets)
    else:
        colors = [
            ("#e6c7c7", "#aa3333"), #red
            ("#e6e6c8", "#abab33"), #yellow
            ("#c7e6d0", "#33ab55"), #green
            ("#c8c9e6", "#3339ab") #blue
        ]
        
        outputName = "pareto-c4-l4-uniform_no-alpha_-filtered-margin0.040-steps"

        fronts = []
        for name in names:
            schedules, is_efficient, optimistic_front, realizable_front = loadDataChains(name)
            fronts.append((optimistic_front, realizable_front))

        drawParetoFrontSuperimposed(fronts, 
            true_fronts, truth_schedules, colors, 
            name=outputName, bounding_box=bounding_box, prints=False, x_offset=x_offset, x_scale=x_scale, loffsets=label_offsets)