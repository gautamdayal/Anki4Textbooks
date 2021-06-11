import time
s = time.time()
import networkx as nx
import pandas as pd
import cv2
import re
import pytesseract
import matplotlib.pyplot as plt
print(f"Imports completed in {time.time() - s}s")

def load(somethinglol):
    img = cv2.imread(somethinglol)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def select_numbers(img):
    """
    Develop a score for each set of numbers. The score is based on:
    - Exactly how consecutive they are (maybe)
    - The displacement for each point from the best straight line drawn through the set of points
    - The angle the directed line makes with the x axis
    """
    def score(numset):
        return 0

    def generateNumsets(G):
        """
        DFS-based enumeration of each possible numset
        """
        paths = []

        path = [0]
        for edge in nx.dfs_edges(G, 0):
            if edge[0] == path[-1]:
                path.append(edge[1])
            else:
                paths.append(path)
                search_index = 2
                while search_index <= len(path):
                    if edge[0] == path[-search_index]:
                        path = path[:-search_index + 1] + [edge[1]]
                        break
                    search_index += 1
                else:
                    raise Exception("Wrong path structure?", path, edge)
        paths.append(path)
        return paths

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

    s = time.time()
    words = pytesseract.image_to_data(img, output_type='data.frame').values.tolist()
    boxes = list(filter(lambda x: isinstance(x[11], str) and x[11].strip(), words))
    boxes = list(map(lambda b: [unpunctuate(b[11]), b[6], -b[7]], boxes))
    numbers = list(filter(lambda x: x[0].isnumeric(), boxes))
    numbers = list(map(lambda b: [int(b[0]), ] + b[1:], numbers))
    numbers.sort(key= lambda x: x[0])
    print(f"Numbers obtained in {time.time() - s}s")
    print(numbers)

    pos = {i: (numbers[i][1], numbers[i][2]) for i in range(len(numbers))}
    labeldict = {i: numbers[i][0] for i in range(len(numbers))}
    G = nx.from_dict_of_lists(treeify(numbers))
    nx.draw(G, labels = labeldict, with_labels=True, pos=pos)
    plt.show()

    # numsets = generateNumsets(G)
    # for path in numsets:
    #     H = nx.from_edgelist([(path[i], path[i+1]) for i in range(len(path) - 1)])
    #     nx.draw(H, labels = labeldict, with_labels=True, pos=pos)
    #     plt.show()

# print(pytesseract.image_to_data(load('numexample1.png')))
print(select_numbers(load('images/numexample1.png')))
