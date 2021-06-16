import time
s = time.time()
import networkx as nx
import pandas as pd
import cv2
import numpy as np
from numpy.linalg import norm
import re
import math
import pytesseract
import matplotlib.pyplot as plt
import itertools
import sklearn.metrics
# print(f"Imports completed in {time.time() - s}s")

def load(somethinglol):
    img = cv2.imread(somethinglol)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    img = cv2.resize(img, (1000, 1000))
    return img

"""
How to detect question numbers:
1. Although some numbers are missed out, they're in sequence most of the time.
2. They are at the center of the distribution of all the numbers, very dense.
3. If you draw a line of best fit through them, they're all very close to that line.
4. That line of best fit shoult point straight downwards.
"""

def select_numbers(img):
    """
    Develop a score for each set of numbers. The score is based on:
    - Exactly how consecutive they are (maybe)
    - The displacement for each point from the best straight line drawn through the set of points
    - The angle the directed line makes with the x axis
    """
    def generateNumsets(G):
        """
        [Not Anymore] DFS-based enumeration of each possible numset
        This is just brute force right now, will find faster way to do it later.
        """
        # paths = []
        #
        # path = [0]
        # for edge in nx.dfs_edges(G, 0):
        #     if edge[0] == path[-1]:
        #         path.append(edge[1])
        #     else:
        #         paths.append(path)
        #         search_index = 2
        #         while search_index <= len(path):
        #             if edge[0] == path[-search_index]:
        #                 path = path[:-search_index + 1] + [edge[1]]
        #                 break
        #             search_index += 1
        #         else:
        #             raise Exception("Wrong path structure?", path, edge)
        # paths.append(path)
        # return paths

        """
        Trying to use itertools LMAO
        """
        # paths = []
        #
        # for path in itertools.combinations(G.nodes, 5):
        #     paths.append(path)
        # return paths

        """
        Generating paths using graph
        """
        paths = []
        n = len(G.nodes)
        for source in range(n):
            for target in range(source+1, n):
                paths.extend([path for path in nx.all_simple_paths(G, source=source, target=target)])
        return paths

        # return paths

    def score(numset):
        # print(numset, numbers)
        tot = 0
        n = len(numset)
        # consecutivity
        consecCounter = 0
        for i in range(n-1):
            n1 = numset[i]; n2 = numset[i+1]
            consecCounter += numbers[n2][0] - numbers[n1][0]
        consecCounter/=n

        tot += abs(1 - consecCounter)

        tot -= 2*n
        # print(tot)

        # check if it's within line of best fit
        # uses average shortest distances of the points from the line of best fit

        X = [numbers[i][1] for i in numset]
        Y = [numbers[i][2] for i in numset]
        coeffs = np.polyfit(Y, X, 1)
        Xfit = np.polyval(coeffs, Y)

        p1 = np.array([Xfit[0], Y[0]])
        p2 = np.array([Xfit[-1], Y[-1]])
        totdists = 0
        for i in range(n):
            p3 = np.array([X[i], Y[i]])
            dist = np.abs(np.cross(p2-p1, p1-p3)) / norm(p2-p1)
            totdists += dist

        tot += (totdists/n)

        # print(f"Numset: {numset}; Actual numbers: {[numbers[i][0] for i in numset]}; Score: {tot}")

        # see where that line is pointing
        tot += abs(math.atan(coeffs[0]))*1000

        return tot

    def treeify(numbers):
        n = len(numbers)
        adj_list = {x:[] for x in range(n)}
        for n1 in range(n):
            for n2 in range(n1+1, n):
                if numbers[n1][0] + 1 == numbers[n2][0]:
                    adj_list[n1].append(n2)
                elif numbers[n1][0] + 1 < numbers[n2][0]:
                    break
        return adj_list

    def unpunctuate(S):
        return re.sub(r'[^\w\s]', '', S)

    def numbersFromNumset(numset):
        return [numbers[x] for x in numset]

    s = time.time()
    words = pytesseract.image_to_data(img, output_type='data.frame').values.tolist()
    boxes = list(filter(lambda x: isinstance(x[11], str) and x[11].strip(), words))
    boxes = list(map(lambda b: [unpunctuate(b[11]), b[6], -b[7]], boxes))
    numbers = list(filter(lambda x: x[0].isnumeric(), boxes))
    numbers = list(map(lambda b: [int(b[0]), ] + b[1:], numbers))
    numbers.sort(key= lambda x: x[0])
    # print(f"Numbers obtained in {time.time() - s}s")
    # print(numbers)

    # rawNums = [x[0] for x in numbers]
    # plt.scatter(rawNums, np.zeros(len(rawNums)))
    # plt.show()

    # Getting some variables ready for graph visualisation
    pos = {i: (numbers[i][1], numbers[i][2]) for i in range(len(numbers))}

    labeldict = {i: f"{i},{numbers[i][0]}" for i in range(len(numbers))}

    # Creating the number graph
    G = nx.from_dict_of_lists(treeify(numbers))
    # nx.draw(G, labels=labeldict, pos=pos, with_labels=True, arrows=True)
    # plt.show()

    # print(score([0, 1, 3, 5, 6]))

    # Sets of numbers
    numsets = generateNumsets(G)
    scores = list(map(score, numsets))
    ranking = np.argsort(scores)
    n = len(ranking)
    # print(score([0, 1, 3, 5, 6]))
    # print("----------------------")
    # for i in range(5):
    #     print(f"Numset: {numsets[ranking[i]]}; Actual numbers: {numbersFromNumset(numsets[ranking[i]])}; Score: {score(numsets[ranking[i]])}")

    best = numsets[ranking[0]]

    numbers = [numbers[i][:2] + [-numbers[i][2], ] for i in range(len(numbers))]
    return [numbers[i] for i in best]

    # plt.show()
    # print(numsets)
    # for path in numsets:
    #     print(path)
    #     H = nx.from_edgelist([(path[i], path[i+1]) for i in range(len(path) - 1)])
    #     nx.draw(H, labels = labeldict, with_labels=True, pos=pos)
    #     plt.show()

def clusterCoords(img):
    # does not take care of multi-column question numbers
    qcoords = select_numbers(img)
    coords = []
    h, w = img.shape
    for i in range(len(qcoords)):
        question1, x1, y1 = qcoords[i]

        if i != len(qcoords)-1:
            question2, x2, y2 = qcoords[i+1]
        else:
            y2 = h

        coords.append([(x1)//2, (y1+y2)//2])

    return coords

def cropCoords(questionCoords, imgWidth, imgHeight):
    quads = []
    n = len(questionCoords)
    for i in range(n):
        q1 = questionCoords[i]
        if i == n - 1:
            q2 = [None, imgWidth, imgHeight]
        else:
            q2 = questionCoords[i+1]
        quads.append([q1[2], q2[2], 0, imgWidth])
    return quads


# print(pytesseract.image_to_data(load('numexample1.png')))
img = load('images/test.png')
questionCoords = select_numbers(img)
h, w = img.shape
imgNum = 0
for quad in cropCoords(questionCoords, w, h):
    temp = img[quad[0]:quad[1], quad[2]:quad[3]]
    cv2.imshow(str(imgNum), temp)
    imgNum += 1
cv2.waitKey(0)
