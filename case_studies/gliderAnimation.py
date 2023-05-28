import sys
 
# adding above folder to the system path
sys.path.insert(0, '../')

from mdp import *
import numpy as np
import math
import random
import time

import cv2

from glider import *


def policyActionQuery(time, state, policy, checkin_period):
    macro_action = policy[state]
    time_within_period = time % checkin_period
    return macro_action[time_within_period], macro_action



def getPolicy(grid, mdp, start_state, discount, discount_checkin, checkin_period):
    # _, policy, _, compMDP = run(grid, mdp, discount, start_state, checkin_period=checkin_period, doBranchAndBound=False, 
    #                             drawPolicy=True, drawIterations=True, outputPrefix="", doLinearProg=True, 
    #                             bnbGreedy=-1, doSimilarityCluster=False, simClusterParams=None, outputDir="../output")
    
    # print(policy)
    policy = {(0, 0): ('DOUBLE-LEFT',), (1, 0): ('DOUBLE-LEFT',), (2, 0): ('DOUBLE-LEFT',), (3, 0): ('DOUBLE-LEFT',), (4, 0): ('RIGHT-LEFT',), (5, 0): ('RIGHT-DOUBLE',), (6, 0): ('RIGHT-DOUBLE',), (7, 0): ('RIGHT-DOUBLE',), (8, 0): ('RIGHT-DOUBLE',), (0, 1): ('DOUBLE-LEFT',), (1, 1): ('DOUBLE-LEFT',), (2, 1): ('DOUBLE-LEFT',), (3, 1): ('DOUBLE-LEFT',), (4, 1): ('FORWARD-LEFT',), (5, 1): ('RIGHT-DOUBLE',), (6, 1): ('RIGHT-DOUBLE',), (7, 1): ('RIGHT-DOUBLE',), (8, 1): ('RIGHT-DOUBLE',), (0, 2): ('DOUBLE-FORWARD',), (1, 2): ('DOUBLE-FORWARD',), (2, 2): ('DOUBLE-FORWARD',), (3, 2): ('DOUBLE-FORWARD',), (4, 2): ('DOUBLE-DOUBLE',), (5, 2): ('FORWARD-DOUBLE',), (6, 2): ('FORWARD-DOUBLE',), (7, 2): ('FORWARD-DOUBLE',), (8, 2): ('FORWARD-DOUBLE',), (0, 3): ('DOUBLE-RIGHT',), (1, 3): ('DOUBLE-RIGHT',), (2, 3): ('DOUBLE-RIGHT',), (3, 3): ('DOUBLE-RIGHT',), (4, 3): ('FORWARD-RIGHT',), (5, 3): ('LEFT-DOUBLE',), (6, 3): ('LEFT-DOUBLE',), (7, 3): ('LEFT-DOUBLE',), (8, 3): ('LEFT-DOUBLE',), (0, 4): ('DOUBLE-LEFT',), (1, 4): ('DOUBLE-LEFT',), (2, 4): ('DOUBLE-LEFT',), (3, 4): ('LEFT-RIGHT',), (4, 4): ('LEFT-RIGHT',), (5, 4): ('LEFT-RIGHT',), (6, 4): ('LEFT-DOUBLE',), (7, 4): ('LEFT-DOUBLE',), (8, 4): ('LEFT-DOUBLE',), (0, 5): ('DOUBLE-LEFT',), (1, 5): ('DOUBLE-LEFT',), (2, 5): ('DOUBLE-LEFT',), (3, 5): ('DOUBLE-LEFT',), (4, 5): ('FORWARD-LEFT',), (5, 5): ('RIGHT-DOUBLE',), (6, 5): ('RIGHT-DOUBLE',), (7, 5): ('RIGHT-DOUBLE',), (8, 5): ('RIGHT-DOUBLE',), (0, 6): ('DOUBLE-FORWARD',), (1, 6): ('DOUBLE-FORWARD',), (2, 6): ('DOUBLE-FORWARD',), (3, 6): ('DOUBLE-FORWARD',), (4, 6): ('DOUBLE-DOUBLE',), (5, 6): ('FORWARD-DOUBLE',), (6, 6): ('FORWARD-DOUBLE',), (7, 6): ('FORWARD-DOUBLE',), (8, 6): ('FORWARD-DOUBLE',), (0, 7): ('DOUBLE-RIGHT',), (1, 7): ('DOUBLE-RIGHT',), (2, 7): ('DOUBLE-RIGHT',), (3, 7): ('DOUBLE-RIGHT',), (4, 7): ('FORWARD-RIGHT',), (5, 7): ('LEFT-DOUBLE',), (6, 7): ('LEFT-DOUBLE',), (7, 7): ('LEFT-DOUBLE',), (8, 7): ('LEFT-DOUBLE',), (0, 8): ('DOUBLE-RIGHT',), (1, 8): ('DOUBLE-RIGHT',), (2, 8): ('DOUBLE-RIGHT',), (3, 8): ('DOUBLE-RIGHT',), (4, 8): ('LEFT-RIGHT',), (5, 8): ('LEFT-DOUBLE',), (6, 8): ('LEFT-DOUBLE',), (7, 8): ('LEFT-DOUBLE',), (8, 8): ('LEFT-DOUBLE',)}
    # policy = {(0, 0): ('DOUBLE-LEFT', 'DOUBLE-LEFT', 'DOUBLE-FORWARD'), (1, 0): ('DOUBLE-LEFT', 'DOUBLE-LEFT', 'DOUBLE-RIGHT'), (2, 0): ('DOUBLE-LEFT', 'DOUBLE-LEFT', 'DOUBLE-DOUBLE'), (3, 0): ('DOUBLE-LEFT', 'FORWARD-LEFT', 'DOUBLE-DOUBLE'), (4, 0): ('RIGHT-LEFT', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (5, 0): ('RIGHT-DOUBLE', 'FORWARD-LEFT', 'DOUBLE-DOUBLE'), (6, 0): ('RIGHT-DOUBLE', 'RIGHT-DOUBLE', 'DOUBLE-DOUBLE'), (7, 0): ('RIGHT-DOUBLE', 'RIGHT-DOUBLE', 'LEFT-DOUBLE'), (8, 0): ('RIGHT-DOUBLE', 'RIGHT-DOUBLE', 'FORWARD-DOUBLE'), (0, 1): ('DOUBLE-LEFT', 'DOUBLE-FORWARD', 'DOUBLE-FORWARD'), (1, 1): ('DOUBLE-LEFT', 'DOUBLE-FORWARD', 'DOUBLE-RIGHT'), (2, 1): ('DOUBLE-LEFT', 'DOUBLE-FORWARD', 'DOUBLE-DOUBLE'), (3, 1): ('DOUBLE-LEFT', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (4, 1): ('FORWARD-LEFT', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (5, 1): ('RIGHT-DOUBLE', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (6, 1): ('RIGHT-DOUBLE', 'FORWARD-DOUBLE', 'DOUBLE-DOUBLE'), (7, 1): ('RIGHT-DOUBLE', 'FORWARD-DOUBLE', 'LEFT-DOUBLE'), (8, 1): ('RIGHT-DOUBLE', 'FORWARD-DOUBLE', 'FORWARD-DOUBLE'), (0, 2): ('DOUBLE-FORWARD', 'DOUBLE-FORWARD', 'DOUBLE-FORWARD'), (1, 2): ('DOUBLE-FORWARD', 'DOUBLE-FORWARD', 'DOUBLE-RIGHT'), (2, 2): ('DOUBLE-FORWARD', 'DOUBLE-FORWARD', 'DOUBLE-DOUBLE'), (3, 2): ('DOUBLE-FORWARD', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (4, 2): ('DOUBLE-DOUBLE', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (5, 2): ('FORWARD-DOUBLE', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (6, 2): ('FORWARD-DOUBLE', 'FORWARD-DOUBLE', 'DOUBLE-DOUBLE'), (7, 2): ('FORWARD-DOUBLE', 'FORWARD-DOUBLE', 'LEFT-DOUBLE'), (8, 2): ('FORWARD-DOUBLE', 'FORWARD-DOUBLE', 'FORWARD-DOUBLE'), (0, 3): ('DOUBLE-RIGHT', 'DOUBLE-FORWARD', 'DOUBLE-FORWARD'), (1, 3): ('DOUBLE-RIGHT', 'DOUBLE-FORWARD', 'DOUBLE-RIGHT'), (2, 3): ('DOUBLE-RIGHT', 'DOUBLE-FORWARD', 'DOUBLE-DOUBLE'), (3, 3): ('DOUBLE-RIGHT', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (4, 3): ('FORWARD-RIGHT', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (5, 3): ('LEFT-DOUBLE', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (6, 3): ('LEFT-DOUBLE', 'FORWARD-DOUBLE', 'DOUBLE-DOUBLE'), (7, 3): ('LEFT-DOUBLE', 'FORWARD-DOUBLE', 'LEFT-DOUBLE'), (8, 3): ('LEFT-DOUBLE', 'FORWARD-DOUBLE', 'FORWARD-DOUBLE'), (0, 4): ('DOUBLE-LEFT', 'DOUBLE-LEFT', 'DOUBLE-FORWARD'), (1, 4): ('DOUBLE-LEFT', 'DOUBLE-LEFT', 'DOUBLE-LEFT'), (2, 4): ('DOUBLE-LEFT', 'DOUBLE-LEFT', 'DOUBLE-DOUBLE'), (3, 4): ('RIGHT-LEFT', 'DOUBLE-FORWARD', 'DOUBLE-DOUBLE'), (4, 4): ('RIGHT-LEFT', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (5, 4): ('RIGHT-LEFT', 'FORWARD-DOUBLE', 'DOUBLE-DOUBLE'), (6, 4): ('RIGHT-DOUBLE', 'RIGHT-DOUBLE', 'DOUBLE-DOUBLE'), (7, 4): ('RIGHT-DOUBLE', 'RIGHT-DOUBLE', 'RIGHT-DOUBLE'), (8, 4): ('RIGHT-DOUBLE', 'RIGHT-DOUBLE', 'FORWARD-DOUBLE'), (0, 5): ('DOUBLE-LEFT', 'DOUBLE-FORWARD', 'DOUBLE-FORWARD'), (1, 5): ('DOUBLE-LEFT', 'DOUBLE-FORWARD', 'DOUBLE-LEFT'), (2, 5): ('DOUBLE-LEFT', 'DOUBLE-FORWARD', 'DOUBLE-DOUBLE'), (3, 5): ('DOUBLE-LEFT', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (4, 5): ('FORWARD-LEFT', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (5, 5): ('RIGHT-DOUBLE', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (6, 5): ('RIGHT-DOUBLE', 'FORWARD-DOUBLE', 'DOUBLE-DOUBLE'), (7, 5): ('RIGHT-DOUBLE', 'FORWARD-DOUBLE', 'RIGHT-DOUBLE'), (8, 5): ('RIGHT-DOUBLE', 'FORWARD-DOUBLE', 'FORWARD-DOUBLE'), (0, 6): ('DOUBLE-FORWARD', 'DOUBLE-FORWARD', 'DOUBLE-FORWARD'), (1, 6): ('DOUBLE-FORWARD', 'DOUBLE-FORWARD', 'DOUBLE-LEFT'), (2, 6): ('DOUBLE-FORWARD', 'DOUBLE-FORWARD', 'DOUBLE-DOUBLE'), (3, 6): ('DOUBLE-FORWARD', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (4, 6): ('DOUBLE-DOUBLE', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (5, 6): ('FORWARD-DOUBLE', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (6, 6): ('FORWARD-DOUBLE', 'FORWARD-DOUBLE', 'DOUBLE-DOUBLE'), (7, 6): ('FORWARD-DOUBLE', 'FORWARD-DOUBLE', 'RIGHT-DOUBLE'), (8, 6): ('FORWARD-DOUBLE', 'FORWARD-DOUBLE', 'FORWARD-DOUBLE'), (0, 7): ('DOUBLE-RIGHT', 'DOUBLE-FORWARD', 'DOUBLE-FORWARD'), (1, 7): ('DOUBLE-RIGHT', 'DOUBLE-FORWARD', 'DOUBLE-LEFT'), (2, 7): ('DOUBLE-RIGHT', 'DOUBLE-FORWARD', 'DOUBLE-DOUBLE'), (3, 7): ('DOUBLE-RIGHT', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (4, 7): ('FORWARD-RIGHT', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (5, 7): ('LEFT-DOUBLE', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (6, 7): ('LEFT-DOUBLE', 'FORWARD-DOUBLE', 'DOUBLE-DOUBLE'), (7, 7): ('LEFT-DOUBLE', 'FORWARD-DOUBLE', 'RIGHT-DOUBLE'), (8, 7): ('LEFT-DOUBLE', 'FORWARD-DOUBLE', 'FORWARD-DOUBLE'), (0, 8): ('DOUBLE-RIGHT', 'DOUBLE-RIGHT', 'DOUBLE-FORWARD'), (1, 8): ('DOUBLE-RIGHT', 'DOUBLE-RIGHT', 'DOUBLE-LEFT'), (2, 8): ('DOUBLE-RIGHT', 'DOUBLE-RIGHT', 'DOUBLE-DOUBLE'), (3, 8): ('DOUBLE-RIGHT', 'FORWARD-RIGHT', 'DOUBLE-DOUBLE'), (4, 8): ('LEFT-RIGHT', 'DOUBLE-DOUBLE', 'DOUBLE-DOUBLE'), (5, 8): ('LEFT-DOUBLE', 'FORWARD-RIGHT', 'DOUBLE-DOUBLE'), (6, 8): ('LEFT-DOUBLE', 'LEFT-DOUBLE', 'DOUBLE-DOUBLE'), (7, 8): ('LEFT-DOUBLE', 'LEFT-DOUBLE', 'RIGHT-DOUBLE'), (8, 8): ('LEFT-DOUBLE', 'LEFT-DOUBLE', 'FORWARD-DOUBLE')}

    return lambda time, state: policyActionQuery(time, state, policy, checkin_period)

def addTup(tup1, tup2):
    return (tup1[0] + tup2[0], tup1[1] + tup2[1])

def arrToCvTup(a):
    return (int(a[0]), int(a[1]))

def sepToState(sepX, sepY, center_state):
    return (center_state[0] + sepX, center_state[1] + sepY)

def gridCoordsToImg(pos, glider_center, origin, gridSize):
    relativePos = pos - glider_center
    imagePos = origin + relativePos * gridSize
    return imagePos

def drawGlider(frame, pos, poses, endPos, glider_center, origin, gridSize, size, color):
    h, w, c = frame.shape

    rDim = np.array([size, size])
    gliderImagePos = gridCoordsToImg(pos, glider_center, origin, gridSize)

    start = gridCoordsToImg(poses[-1], glider_center, origin, gridSize)
    end = gridCoordsToImg(endPos, glider_center, origin, gridSize)
    frame = cv2.line(frame, arrToCvTup(start), arrToCvTup(end), color=(128, 128, 128), thickness=2)

    for i in range(len(poses)):
        start = gridCoordsToImg(poses[i], glider_center, origin, gridSize)
        end = gridCoordsToImg(poses[i+1] if i < len(poses)-1 else pos, glider_center, origin, gridSize)

        frame = cv2.line(frame, arrToCvTup(start), arrToCvTup(end), color=color, thickness=2)

    

    frame = cv2.rectangle(frame, arrToCvTup(gliderImagePos - rDim/2), arrToCvTup(gliderImagePos + rDim/2), color=color, thickness=-1)

    return frame

def drawGrid(frame, glider_center, origin, gridSize, markerSize, color):
    h, w, c = frame.shape

    gX = math.ceil(w / gridSize)
    gY = math.ceil(h / gridSize)
    
    for i in range(-gX, gX):
        for j in range(-int(gY), int(gY)):
            pos = np.array([i + int(glider_center[0]), j + int(glider_center[1])])

            imagePos = gridCoordsToImg(pos, glider_center, origin, gridSize)
            frame = cv2.line(frame, arrToCvTup(imagePos - np.array([0, markerSize/2])), arrToCvTup(imagePos + np.array([0, markerSize/2])), color=color, thickness=2)
            frame = cv2.line(frame, arrToCvTup(imagePos - np.array([markerSize/2, 0])), arrToCvTup(imagePos + np.array([markerSize/2, 0])), color=color, thickness=2)
    return frame


def drawFrame(frame, g1Pos, g2Pos, g1Poses, g2Poses, g1EndPos, g2EndPos):
    h, w, c = frame.shape

    origin = np.array([w/8, h/2])

    gridSize = h/8

    glider_center = (g1Pos + g2Pos) / 2

    drawGrid(frame, glider_center, origin, gridSize, markerSize=20, color=(128, 128, 128))

    drawGlider(frame, g1Pos, g1Poses, g1EndPos, glider_center, origin, gridSize, size=50, color=(0, 128, 255))
    drawGlider(frame, g2Pos, g2Poses, g2EndPos, glider_center, origin, gridSize, size=50, color=(0, 0, 255))

    return frame

def npa(tup):
    return np.array([tup[0], tup[1]])

def move(pos, action, moveProb):
    glider_actions = {
        "DOUBLE": (2, 0), 
        "FORWARD": (1, 0), 
        "LEFT": (1, -1), 
        "RIGHT": (1, 1)
    }

    displace = npa(glider_actions[action])
    r = random.random()
    if r <= (1-moveProb):
        driftDir = -1 if r <= (1-moveProb)/2 else 1
        displace += np.array([0, driftDir])

    return pos + displace




def animationLoop(grid, mdp, policy, checkin_period, center_state, initial_sep, moveProb, maxSeparation):
    
    g2Pos = np.array([0.0, 0.0])
    g1Pos = g2Pos + initial_sep

    g1Poses = []
    g2Poses = []

    while True:
        w = 2560
        h = 1440
        
        frame_canvas = np.zeros((h, w, 3), np.uint8)

        t = 0
        sep = g1Pos - g2Pos
        action, macro_action = policy(t, sepToState(sepX = int(sep[0]), sepY = int(sep[1]), center_state = center_state))
        
        print("Running time",t,"action",action)
        print(g1Pos, g2Pos, sepToState(sepX = int(sep[0]), sepY = int(sep[1]), center_state = center_state))

        frame_canvas = cv2.putText(frame_canvas, action, (50, 100), 
            cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255,255,255), lineType=cv2.LINE_AA, thickness=2)

        spl = action.split("-")
        g1_action = spl[0]
        g2_action = spl[1]

        g2EndPos = move(g2Pos, g2_action, moveProb)
        g1EndPos = move(g1Pos, g1_action, moveProb)

        g2EndPos_intended = move(g2Pos, g2_action, 1)
        g1EndPos_intended = move(g1Pos, g1_action, 1)
        
        end_sep = g1EndPos - g2EndPos
        
        end_sep = np.clip(end_sep, -maxSeparation, maxSeparation) # to keep within MDP grid, not necessarily realistic
        
        g1EndPos = end_sep + g2EndPos

        
        secPerStep = 1
        framesPerSec = 20
        framesPerStep = secPerStep * framesPerSec

        g1StartPos = g1Pos
        g2StartPos = g2Pos

        g1Poses.append(g1StartPos)
        g2Poses.append(g2StartPos)

        if len(g1Poses) > 4:
            del g1Poses[0]
            del g2Poses[0]

        for i in range(framesPerStep):
            start = time.time()
            frame = drawFrame(np.copy(frame_canvas), g1Pos, g2Pos, g1Poses, g2Poses, g1EndPos_intended, g2EndPos_intended)

            cv2.imshow("Paired Gliders", frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord(' '):
                key = cv2.waitKey(0)
            if key & 0xFF == ord('q'):
                exit()

            elapsed = (time.time() - start)
            # print(elapsed)
            secPerFrame = 1 / framesPerSec
            left = secPerFrame - elapsed
            if left > 0:
                time.sleep(left)

            # g1Pos += np.array([1.0, 1.0]) / 1000 * elapsed_millis
            # g2Pos += np.array([1.0, 1.0]) / 1000 * elapsed_millis

            g1Pos = g1StartPos + (g1EndPos - g1StartPos) / framesPerStep * i
            g2Pos = g2StartPos + (g2EndPos - g2StartPos) / framesPerStep * i

        g1Pos = g1EndPos
        g2Pos = g2EndPos
        t += 1


def runGliderAnimation():

    maxSeparation = 4
    moveProb = 0.9#0.9

    grid, mdp, start_state = gliderMDP(
        maxSeparation = maxSeparation, 
        desiredSeparation = 2, 
        moveProb = moveProb, 
        wallPenalty = -10, 
        movePenalty = 0, 
        collidePenalty = -100, 
        desiredStateReward=5)

    discount = math.sqrt(0.99)
    discount_checkin = discount

    centerInd = int(len(grid) / 2)
    center_state = (centerInd, centerInd)

    checkin_period = 1
    policy = getPolicy(grid, mdp, start_state, discount, discount_checkin, checkin_period)

    initial_sep = np.array([-1, -1])

    animationLoop(grid, mdp, policy, checkin_period, center_state, initial_sep, moveProb, maxSeparation)
    
    # print(center_state)
    # print(sepToState(0, 2, center_state))
    # print(policy(0, sepToState(0, 2, center_state)))



runGliderAnimation()

