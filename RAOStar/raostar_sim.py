# author: Matt Deyo
'''
Simulate execution of RAO* policies, using the prob distributions from the models.
Goal is to compare how offline RAO* performs during execution vs iRAO*, this
includes the success rate (reaching goal state without crashing) and ultimate
utility of the path taken.
'''

import sys
from utils import import_models
import_models()
from r2d2model import R2D2Model
from raostar import RAOStar
import graph_to_json
from iterative_raostar import *
import random


def get_next_state(current_state, model, graph):
    # print(current_state)
    risk = model.state_risk(current_state.state)
    # print('risk', risk)
    if risk == 1.0:
        return "crash"
    action = current_state.best_action

    if not action:
        # print('no best action, done?')
        return 'done'

    children = graph.hyperedges[current_state][action]
    # print(action, children)
    # print(current_state.best_action.properties)
    chance = random.random()
    # print(chance)
    threshold = 0
    outcome = None
    outcome_index = 0
    found_outcome = False
    while not found_outcome:
        # print(outcome_index)
        # print(action.properties['prob'])
        # print('chance', chance)
        if chance <= threshold + action.properties['prob'][outcome_index]:
            found_outcome = True
            outcome = children[outcome_index]
        else:
            threshold += action.properties['prob'][outcome_index]
            outcome_index += 1

    # print('outcome', outcome)
    return outcome


# Now you can give command line cc argument after filename
if __name__ == '__main__':
    # default chance constraint value
    cc = 0.08
    if len(sys.argv) > 1:
        cc = float(sys.argv[1])

    num_sims = 100
    if len(sys.argv) > 2:
        num_sims = int(sys.argv[2])

    ice_blocks = [(1, 0), (1, 1)]
    model = R2D2Model(ice_blocks)
    algo = RAOStar(model, cc=cc, debugging=False)
    b_init = {(1, 0, 0): 1.0}
    P, G = algo.search(b_init)
    P = clean_policy(P)  # remove empty policies

    num_crashes = 0
    num_successes = 0

    # Got offline policy and graph, now need to simulate execution
    for i in range(num_sims):
        current_state = G.root

        for j in range(5):
            next_state = get_next_state(current_state, model, G)
            current_state = next_state
            if current_state == 'crash':
                num_crashes += 1
                break
            if current_state == 'done':
                num_successes += 1
                break

        print('***** done simulation *****', i)
    print('crashes', num_crashes)
    print('successes', num_successes)
    # raise ValueError()
    # model.print_model()
    # model.print_policy(P)
    # most_likely_policy(G, model)
    # g = graph_to_json.policy_to_json(G, chance_constraint, "results/r2d2_raos.json")
    # print(g)
