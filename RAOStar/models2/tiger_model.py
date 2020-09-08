import numpy as np

class tiger_model:
    def __init__(self, time_horizon=5, step_horizon=5, cc=0.1, optimization = 'Max'):
        #self.initial_belief = np.array([0.5, 0.5])  # np.array
        self.action_list = ['Open Left', 'Open Right', 'Listen']
        self.trans_prob = [[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1.0, 0.0], [0.0, 1.0]]]  # np.array
        self.obs_prob = [[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]], [[0.85, 0.15], [0.15, 0.85]]]  # np.array
        self.t_horizon = time_horizon
        self.s_horizon = step_horizon
        self.soft_horizon = step_horizon
        self.reward_func = np.array([[0, 10], [10, 0], [1, 1]])  # np.array
        self.risk_func = np.array([[1, 0], [0, 1], [0, 0]])  # np.array
        self.duration_func = np.array([1, 1, 1])
        self.cc = cc
        self.reset_actions = [True, True, False]
        self.optimization = optimization # maximize or minmize

        # calculates prior belief for a given action
        # and the previous belief: prior_b(s') = sum( trans_p(s'|s,a) * pre_b(s') ) 'for all s'
        # note: use action index, not action name

    def actions(self, state):
        return self.action_list

    def is_terminal(self, state):
        return False  # so it would go till horizon

    def state_transitions(self, state, action):
        '''
        act_idx = self.action_list.index(action)
        prior_belief = [0.0] * len(state)
        for s_prime, pre_prob in enumerate(state):
            for s in range(len(state)):
                prior_belief[s_prime] += self.trans_prob[act_idx][s][s_prime] * state[s]
        return [[(prior_belief), 1]]
        '''
        if 'Listen' in action:
            if state == 'L':
                return [('L', 1.), ('R', 0.)]
            elif state == 'R':
                return [('L', .0), ('R', 1.)]
        else:
            return [('L', .5), ('R', .5)]

    def observations(self, state, act):
        if 'Listen' in act:
            if state == 'L':
                return [('L', .85), ('R', .15)]
            elif state == 'R':
                return [('L', .15), ('R', .85)]
            else:
                raise RuntimeError('wrong obs input in tiger model')
        else:
            if state == 'L':
                return [('L', 1.), ('R', 0.)]
            elif state == 'R':
                return [('L', .0), ('R', 1.)]


    def state_risk(self, state, act):
        if str(act) == str(-1):
            return 0.0
        elif 'Open Left' in act and 'L' in state:
            return 1.

        elif 'Open Right' in act and 'R' in state:
            return 1.
        else:
            return 0.0


    def values(self, state, action):
        if 'Listen' in action:
            return 1
        elif 'Open Left' in action and 'R' in state:
            return 10

        elif 'Open Right' in action and 'L' in state:
            return 10
        else:
            return 0

    def heuristic(self, state):
        return 0

    def execution_risk_heuristic(self, state):
        return 0






