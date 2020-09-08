import numpy as np

class paint_model:
    def __init__(self, time_horizon=5, step_horizon=5, cc=0.1, optimization='Max'):
        #self.initial_belief = np.array([0.5, 0.5])  # np.array
        #self.action_list = np.array(['paint', 'inspect', 'ship', 'reject'])
        self.trans_prob = []  # trans_prob[action][state]=new_belief
        self.trans_prob.append([[0.1, 0.9, 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0.9, 0.1]])  # paint action
        self.trans_prob.append([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])  # inspect action
        self.trans_prob.append([[.5, 0., 0., .5], [.5, 0., 0., .5], [.5, 0., 0., .5], [.5, 0., 0., .5]])  # ship action
        self.trans_prob.append([[.5, 0., 0., .5], [.5, 0., 0., .5], [.5, 0., 0., .5], [.5, 0., 0., .5]])  # reject action
        self.trans_prob = np.array(self.trans_prob)

        self.obs_prob = []  # trans_prob[action][state]=new_belief
        self.obs_prob.append([[.25, .25, .25, .25], [.25, .25, .25, .25]])  # paint action
        self.obs_prob.append([[.25, .25, .25, .25], [1/12, 1/12, 1/12, .75]])  # inspect action
        self.obs_prob.append([[.25, .25, .25, .25], [.25, .25, .25, .25]])  # ship action
        self.obs_prob.append([[.25, .25, .25, .25], [.25, .25, .25, .25]])  # reject action
        #self.obs_prob = np.array(self.obs_prob)

        self.t_horizon = time_horizon
        self.s_horizon = step_horizon
        self.soft_horizon = step_horizon
        self.reward_func = np.array([[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 10., 0., 0.], [0., 0., 5., 5.]])  # reward only for ship or reject, rest is zero
        self.risk_func = np.array([[0., 0., 0., 0.], [0., 0., 0., 0.], [1/3, 0., 1/3, 1/3], [.5, .5, 0., 0.]])  # np.array
        self.duration_func = np.array([1, 1, 1, 1])
        self.cc = cc
        self.reset_actions = [True, False, False, True]
        self.optimization = optimization  # maximize or minimize

        #####################################################################################################
        self.action_list = ['paint', 'inspect', 'ship', 'reject']
        self.state_list = ['NFL-NBL-NPA', 'NFL-NBL-PA', 'FL-NBL-PA', 'FL-BL-NPA']



    def actions(self, state):
        return self.action_list

    def is_terminal(self, state):
        return False  # so it would go till horizon

    def state_transitions(self, state, action):
        if action == 'paint':
            if state == 'NFL-NBL-NPA':
                return [('NFL-NBL-NPA', .1), ('NFL-NBL-PA', .9)]
            elif state == 'FL-BL-NPA':
                return [('FL-BL-NPA', .1), ('FL-NBL-PA', .9)]
            else:
                return [(state, 1.)]

        elif action == 'inspect':
            return [(state, 1.)]

        else:  # for ship and reject action reset state
            return [('NFL-NBL-NPA', .5), ('FL-BL-NPA', .5)]


    def observations(self, state, action):
        if action == 'inspect':
            if state == 'FL-BL-NPA':
                return [('NFL-NBL-NPA', 1/12), ('NFL-NBL-PA', 1/12), ('FL-NBL-PA', 1/12), ('FL-BL-NPA', .75)]
            else:
                return [('NFL-NBL-NPA', .25), ('NFL-NBL-PA', .25), ('FL-NBL-PA', .25), ('FL-BL-NPA', .25)]
        elif action == 'paint':
            return [('NFL-NBL-NPA', 1/3), ('NFL-NBL-PA', 1/3), ('FL-NBL-PA', 1/3), ('FL-BL-NPA', .0)]

        else:  # for ship and reject
            return [('NFL-NBL-NPA', .5), ('FL-BL-NPA', .5)]


    def state_risk(self, state, action):
        if str(action) == str(-1):
            return 0.0

        elif action == 'ship' and state != 'NFL-NBL-PA':
            return 1.

        elif action == 'reject' and 'NFL' in state:
            return 1.
        else:
            return 0

    def values(self, state, action):
        if action == 'ship' and state == 'NFL-NBL-PA':
            return 10.

        elif action == 'reject' and 'NFL' not in state:
            return 1.

        else:
            return 0.

    def heuristic(self, state):
        return 0

    def execution_risk_heuristic(self, state):
        return 0










