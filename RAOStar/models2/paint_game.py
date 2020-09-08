import numpy as np
from ccpomdp_class_FPTAS_tiger import CCPOMDP
from paint_model import paint_model
import IPython
# state is the correct state
# observation depends on action
# for listen prob of 85% returns correct state
def paint_observ(state, action):  # return 1 if BL else 0
    if int(action) == 1:  # inspect action
        if np.random.choice([True, False], p=[.75, .25]):
            return state[3]
        else:
            return 0 if state[3] == 1 else 1
    else:
        return 0


def paint_test(node_names, actions_list):
    state = [1, 0, 0, 0] if np.random.choice([True, False]) else [0, 0, 0, 1]
    count = 0
    obs_count = 2

    current_node = 'r'
    obs_index = 0

    while not node_names[current_node].terminal and count < 50:
        try:
            best_action = node_names[current_node].best_action
        except:
            IPython.embed(header='paint')

        print(actions_list[int(best_action)])

        if int(best_action) == 0:  # paint
            if state[0] == 1:
                state = state if np.random.choice([True, False], p=[.1, .9]) else [0, 1, 0, 0]

            elif state[3] == 1:
                state = state if np.random.choice([True, False], p=[.1, .9]) else [0, 0, 1, 0]

        current_node += str(int(best_action))

        current_node += str(int(obs_index))

        observ = paint_observ(state=state, action=best_action)
        if observ == 1:
            print("Observ BL")
        else:
            print("Observ NBL")

        obs_index = 0

        if node_names[current_node].reset:
            print('Door NLF, NBL PA' if state[1] == 1 else 'Door BL or NPA')
            state = [1, 0, 0, 0] if np.random.choice([True, False]) else [0, 0, 0, 1]
            #current_node = 'r'
            obs_index = 0

        count += 1


if __name__ == '__main__':

    Paint_model = paint_model(time_horizon=2, step_horizon=2, cc=0.3, optimization='maximize')
    init_belief = np.array([0.5, 0., 0., 0.5])
    policy = CCPOMDP(initial_belief=init_belief, model=Paint_model)
    paint_test(node_names=policy.node_names, actions_list=Paint_model.action_list)
