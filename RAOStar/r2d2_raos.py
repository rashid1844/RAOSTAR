#!/usr/bin/env python

# author: Yun Chang
# email: yunchang@mit.edu
# rao* on r2d2 model

import sys
from utils import import_models
import_models()
from r2d2model import R2D2Model
from raostar import RAOStar
import graph_to_json
from iterative_raostar import *

# Now you can give command line cc argument after filename
if __name__ == '__main__':
    # default chance constraint value
    cc = 0.08
    if len(sys.argv) > 1:
        cc = float(sys.argv[1])

    length = 3
    model = R2D2Model(length)
    algo = RAOStar(model, cc=cc, debugging=True, cc_type='o')

    b_init = {(1, 0, 0): 1.0}
    P, G = algo.search(b_init)
    print(P)
    P = clean_policy(P)

    model.print_model()
    model.print_policy(P)
    print(P)

    most_likely_policy(G, model)

    print(len(G.nodes), G.nodes)
    print(G.root)
    next = next_child(G, G.root)
    next = next_child(G, next)
    next = next_child(G, next)
    G.update_root_and_purge(next)
    print(len(G.nodes), G.nodes)
    # print(G.all_descendants(G.root))

    # g = graph_to_json.policy_to_json(G, chance_constraint, "results/r2d2_raos.json")
    # print(g)
