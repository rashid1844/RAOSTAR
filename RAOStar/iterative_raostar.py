from utils import import_models
import_models()
from r2d2model import R2D2Model
from raostar import RAOStar
from belief import *

###############
# This one should result in a policy of RIGHT at (1,1)
# because cc = 0.09, and the 10% exec_risk at (1,1) is discounted
# by the stochastic behavior of RIGHT at (1,0)
###############
# ice_blocks = [(1, 0), (1, 1)]
# model = R2D2Model(ice_blocks)
# algo = RAOStar(model, cc=0.09)
# b_init = {(1, 0, 0): 1.0}
# P, G = algo.search(b_init)

# model.print_model()
# model.print_policy(P)


# G.root.print_node()
# print(G.hyperedges[G.root])
# for child in G.hyperedge_successors(G.root, G.root.best_action):
# print(child.name)
# print(G.root.best_action.properties)

def clean_policy(P):
    # Remove all the states from policy that did not have a best action
    P_notNone = {}
    for i in P:
        if P[i] != 'None':
            P_notNone[i] = P[i]

    return P_notNone


def next_child(G, state):
    next_child = None
    most_probable = 0
    for i, child in enumerate(G.hyperedge_successors(state, state.best_action)):
        prob_outcome = state.best_action.properties['prob'][i]
        if prob_outcome > most_probable:
            next_child = child
            most_probable = prob_outcome
    if state.best_action:
        print("   action: " + str(state.best_action.name.split("'")[1]))
        print("next state: " + str(next_child.state.state_print()))
    else:
        print('## Policy complete ##')
    return next_child


def next_child_agent_action(G, state, agent_action):
    next_child = None
    most_probable = 0
    print('current', state.state.state_print())
    for i, child in enumerate(G.hyperedge_successors(state, state.best_action)):
        if child.best_action:
            print(i, child, child.state.state_print(),
                  child.state.previous_action, child.best_action.name)
        if agent_action in child.state.previous_action:
            # print('found a match')
            next_child = child
            # break
    if state.best_action:
        print("   action: " + str(state.best_action.name.split("'")[1]))
        print("next state: " + str(next_child.state.state_print()))
    else:
        print('## Policy complete ##')
    return next_child


import copy


def most_likely_next_state(G, model, node):
    # print('current node:', node.state.state_print())
    next_state = next_child(G, node)
    # print('next node:', node)
    return next_state


def agent_move_next_state(G, model, node, agent_action):
    # print('current node:', node.state.state_print())
    next_state = next_child_agent_action(G, node, agent_action)
    # print('next node:', node)
    return next_state


def most_likely_policy(G, model):
    print('\n## Policy start ##')
    s = G.root
    print('first state: ' + str(s.state.state_print()))
    last_s = None
    while s:
        # print('loop', s)
        last_s = copy.copy(s)
        s = next_child(G, s)
    print('\nfinal state: ' + str(last_s.state.state_print()))

    final_state = last_s.state
    if isinstance(last_s.state, BeliefState):
        final_state = last_s.state.belief.keys()[0]
    if model.is_terminal(final_state):
        print('finished search!')
    else:
        print('Search Failed :(')


def best_action_children(G, P, print_out=False):
    children_names = []
    action = G.root.best_action
    prob_outcome = 0
    if print_out:
        print('# Outcomes from best action:')
    for i, child in enumerate(G.hyperedge_successors(G.root, G.root.best_action)):
        prob_outcome = action.properties['prob'][i]
        children_names.append((child.name, prob_outcome))
        if print_out:
            print('##   state: ' + child.name + ' prob: ' + str(prob_outcome) +
                  ' exec_risk_bound: ' + str(child.exec_risk_bound))
    return children_names


def match_state(list_of_nodes, state):
    for i, child in enumerate(list_of_nodes):
        child_state = child.state.belief.keys()[0]
        if state[0] == child_state[0] and state[1] == child_state[1]:
            # print('Matched state ' + str(state) +
            #       " and child: " + child.name)
            return child
    return False


def make_observation(state, G, P):
    new_state = match_state(G.hyperedge_successors(
        G.root, G.root.best_action), state)
    if not new_state:
        raise ValueError("state: " + str(state) +
                         " not a successor to " + G.root.name)
    # print(type(new_state))
    G.root = new_state

    new_state.print_node()
    if new_state.best_action:
        print(new_state.best_action.properties)


# print("Next best action is: " + G.root.best_action.name)
# best_action_children(G, P, True)
# make_observation((1, 1), G, P)
# make_observation((2, 1), G, P)

# test for when observed state is not valid child to the root node
# make_observation((1, 0), G, P)


###########
# However, if we had started the search fom (1,1) with the same
# risk bound of cc= 0.09, then the policy should be UP from (1,1)
# to ensure the risk bound is met
###########
# ice_blocks = [(1, 0), (1, 1)]
# model = R2D2Model(ice_blocks)
# algo = RAOStar(model, cc=0.09)
# b_init = {(1, 1, 0): 1.0}
# P, G = algo.search(b_init)
# model.print_model()
# model.print_policy(P)
