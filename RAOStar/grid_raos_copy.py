#!/usr/bin/env python

# author: Sungkweon Hong
# email: sk5050@mit.edu
# Grid model as simple model to test rao star

import sys
from utils import import_models
import_models()
from grid_model import GRIDModel
from raostar import RAOStar
import graph_to_json
import time
import copy
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
# from iterative_raostar import *


def print_grid(size, constraint_states, start=(0,0)):
    print('_______' * size[0])
    for i in range(size[0]-1,-1,-1):
        print("|", end=" ")
        for j in range(size[1]):

            if (i,j) in constraint_states:
                print(" C |", end =" ")
            else:
                print(i,",",j,"|", end=" ")
        print(" "); print('_______' * size[0])






# Now you can give command line cc argument after filename
if __name__ == '__main__':
    # default chance constraint value
    cc = 0.02
    if len(sys.argv) > 1:
        cc = float(sys.argv[1])

   
    size = (5,5)
    constraint_states = [(0,1),(1,1),(3,3),(4,3),(0,4)]
    #constraint_states = []

    print_grid(size, constraint_states, start=(0,0))

    model = GRIDModel(size, constraint_states, prob_right_transition=0.98, prob_right_observation=0.98, DetermObs=True, goal=(4,4))

    algo = RAOStar(model, cc=cc, debugging=False, cc_type='o', fixed_horizon = 3, random_node_selection=False)
    # algo = RAOStar(model, cc=cc, debugging=False, cc_type='o', fixed_horizon = 3, random_node_selection=False, time_limit=60*45)

    b_init = {(0,0): 1.0}
    state = (0,0)
    P, G = algo.search(b_init)

    # print("Root risk : ",algo.graph.root.exec_risk)
    # print("Root value: ",algo.graph.root.value)

    #algo.extract_policy()

    print(P)

    # complete_flag = 1
    # for node in algo.opennodes:
    #     if node.terminal != True:
    #         complete_flag = 0
    #         break

    # print(complete_flag)
                

    # print(algo.incumbent_value_list)
    # print(algo.pruning_count)
