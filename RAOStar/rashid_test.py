from models.grid_model import GRIDModel
from models.network_model import NetworkModel
from models2.paint_model import paint_model
from models2.tiger_model import tiger_model
from raostar import RAOStar
import IPython
from collections import deque
import time

if __name__ == '__main__':
    Grid = True
    Network = False
    Tiger = False
    Paint = False

    if Grid:
        model = GRIDModel(size=(4, 4), constraint_states=[(1, 2),(1,3)], rewrd_states={(0, 3): 0, (3, 3): 0},
                          prob_right_transition=.8, prob_right_observation=.85, goal=(3,2), step_cost=1, walls=[(1, 1), (2, 2)])

        algo = RAOStar(model, cc=0.1, debugging=False, cc_type='o', fixed_horizon=6, random_node_selection=False)

        policy_current_belief = {(1, 0): 1.0}  # initial state is: current vehicle
        s_time = time.time()
        P, G = algo.search(policy_current_belief)
        print('expanded nodes', len(G.nodes))
        print('objective:', G.root.value)
        print('Time:', time.time()-s_time)
        IPython.embed()

    elif Network:
        model = NetworkModel()

        algo = RAOStar(model, cc=0.20, debugging=False, cc_type='o', fixed_horizon=6, random_node_selection=True)

        policy_current_belief = {0: 1.0}  # initial state is: current vehicle
        s_time = time.time()
        P, G = algo.search(policy_current_belief)
        print('expanded nodes', len(G.nodes))
        print('objective:', G.root.value)
        print('Time:', time.time()-s_time)
        IPython.embed()


    elif Tiger:

        model = tiger_model(optimization='maximize')

        algo = RAOStar(model, cc=0.1, debugging=False, cc_type='o', fixed_horizon=5, random_node_selection=False)

        policy_current_belief = {'L': .5, 'R': .5}  # initial state is: current vehicle

    else:
        model = paint_model(optimization='maximize')

        algo = RAOStar(model, cc=0.2, debugging=False, cc_type='o', fixed_horizon=5, random_node_selection=False)

        policy_current_belief = {'NFL-NBL-NPA': .5, 'FL-BL-NPA': .5}  # initial state is: current vehicle



    P, G = algo.search(policy_current_belief)
    print('expanded nodes', len(G.nodes))
    print('objective:', G.root.value)
    print('')
    IPython.embed()