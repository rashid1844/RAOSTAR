import numpy as np
from ccpomdp_class_FPTAS_tiger import CCPOMDP
from tiger_model import tiger_model

# state is the correct state
# observation depends on action
# for listen prob of 85% returns correct state
def tiger_observ(state, action):
    if int(action) == 2:
        if np.random.choice([True, False], p=[.85, .15]):
            return state
        else:
            return [state[1], state[0]]
    else:
        return [0.5, 0.5]


def tiger_test(node_names, actions_list):
    state = [1, 0] if np.random.choice([True, False]) else [0, 1]
    count = 0
    obs_count = 2

    current_node = 'r'

    while not node_names[current_node].terminal and count < 50:
        best_action = node_names[current_node].best_action
        print(actions_list[int(best_action)])

        current_node += str(int(best_action))
        observ = tiger_observ(state=state, action=best_action)
        if observ == [1, 0]:
            print("Listen Tiger Left")
        elif observ == [0, 1]:
            print("Listen Tiger Right")

        current_node += str(0) if observ == [1, 0] else str(1)

        if node_names[current_node].reset:
            print('Tiger Left' if state == [1, 0] else 'Tiger Right')
            state = [1, 0] if np.random.choice([True, False]) else [0, 1]
            #current_node = 'r'

        count += 1


if __name__ == '__main__':

    Tiger_model = tiger_model(time_horizon=5, step_horizon=5, cc=0.1, optimization='maximize')
    init_belief = np.array([0.5, 0.5])
    policy = CCPOMDP(initial_belief=init_belief, model=Tiger_model)
    tiger_test(node_names=policy.node_names, actions_list=Tiger_model.action_list)
