#!/usr/bin/env python

# author: Yun Chang
# email: yunchang@mit.edu
# re-implementation of Pedro Santana's RAO* algorithm
# RAO*: a risk-aware planner for POMDP's
# Forward, heuristic planner for partially-observable
# chance-constrained domains

import operator
import numpy as np
import time
import random
from collections import deque

from raostarhypergraph import RAOStarGraphNode, RAOStarGraphOperator, RAOStarHyperGraph
from belief import BeliefState, avg_func, bound_prob
from belief import predict_belief, update_belief, compute_observation_distribution, is_terminal_belief
from enumtree import *

class RAOStar(object):
    # find optimal policy and/or tree representing partially
    # observable domains

    def __init__(self, model, cc=0.0, cc_type='o', fixed_horizon=np.inf,
                 terminal_prob=1.0, debugging=False, randomization=0.0, halt_on_violation=False, random_node_selection=False, time_limit=np.inf, iter_limit=np.inf):

        self.model = model
        self.cc = cc
        self.cc_type = cc_type
        # type of chance constraint could be "overall": execution risk
        # bound at the root (constraints overall execution); "everywhere":
        # bounds the execution risk at every policy node.
        self.terminal_prob = terminal_prob
        self.graph = None  # graph construction
        # Expansion queue list. Each element in list is dictionary, with three key-values.
        # 1) parent_etree_node, 2) current_etree_node, 3) expansion_queue
        # expansion_queue is another dictionary, with three key-values.
        # 1) current_etree_node (redundant, but useful), 2) expansion_node, 3) expansion_hyperedge
        self.queue = []
        self.expanded_queue_list = [] # list of expanded queue. element is dictionary,
                                 # with 3-tuple key: (expansion_node name, expansion_hyperedge name, er_bound)
        self.current_etree_node = None # current expanding enumeration tree node
        self.opennodes = set()
        self.policy_ancestors = {}  # ancestors set for updating policy
        self.halt_on_violation = halt_on_violation
        # whether constraint violations is terminal
        self.debugging = debugging

        self.fixed_horizon = fixed_horizon
        self.random_node_selection = random_node_selection
        self.end_search_on_likelihood = False
        self.time_limit = time_limit
        self.iter_limit = iter_limit

        self.incumbent_policy = None
        self.incumbent_value_list = []

        self.pruning_count = 0

        self.debug("halting", self.halt_on_violation)

        # execution risk cap
        if self.cc_type.lower() in ['overall', 'o']:
            self.cc_type = 'overall'
            self.er_cap = 1.0
        elif self.cc_type.lower() in ['everywhere', 'e']:
            self.cc_type = 'everywhere'
            self.er_cap = self.cc
        else:
            raise TypeError(
                'Choose either overall or everywhere for constraint type')

        # choose the comparison function depending on the type of search
        if model.optimization == 'maximize':
            self.is_better = operator.gt
            self.is_worse = operator.lt
            self.initial_Q = -np.inf
            self.select_best_node = lambda n_list: max(
                n_list, key=lambda expansion: expansion['expansion_node'].value)
            self.select_best_hyperedge = lambda n_list: max(
                n_list, key=lambda expansion: expansion['Q_value'])
            self.incumbent_value = -np.inf
            
        elif model.optimization == 'minimize':
            self.is_better = operator.lt
            self.is_worse = operator.gt
            self.initial_Q = np.inf
            self.select_best_node = lambda n_list: min(
                n_list, key=lambda expansion: expansion['expansion_node'].value)
            self.select_best_hyperedge = lambda n_list: min(
                n_list, key=lambda expansion: expansion['Q_value'])
            self.incumbent_value = np.inf
            
        else:
            raise TypeError('unable to recognize optimization')

        self.start_time = 0.0

        # API functions from model (Model function aliases) #
        self.A = model.actions
        self.T = model.state_transitions
        self.O = model.observations
        self.V = model.values
        self.h = model.heuristic
        self.r = model.state_risk
        self.e = model.execution_risk_heuristic
        self.term = model.is_terminal

    def debug(self, *argv):
        if self.debugging:
            msg = ""
            for item in argv:
                msg += " " + str(item)
            print(msg)

    def search_termination(self, count):

        if len(self.queue) > 0 and (count <= self.iter_limit) and (time.time() - self.start_time <= self.time_limit):
            return False

        return True

    def search(self, b0, time_limit=np.inf, iter_limit=np.inf):
        self.start_time = time.time()
        print('\n RAO* initialized with belief: ' + str(b0) + '\n .\n .\n .')
        self.init_search(b0)

        # choose first expansion 
        expansion = self.choose_expansion()
        
        count = 0
        root = self.graph.root

        interrupted = False

        while not self.search_termination(count):
            count += 1

            self.debug('\n\n\n RAO* iteration: ' + str(count) + '\n\n\n')

            # if chosen expansion is for another etree node, then recover the graph for that node.
            if self.current_etree_node != expansion['current_etree_node']:
                
                target_etree_node = expansion['current_etree_node']

                self.extract_policy()
                
                self.etree.checkout(self.current_etree_node, target_etree_node)

                self.extract_policy()
            
                self.current_etree_node = target_etree_node
                
            expanded_etree_node = self.expand_etree_node(expansion)

            print('-----------------------------------')
            print('expansion')
            print(count)
            print('-----------------------------------')

            # Branch this etree node only if it is not dominated.
            if self.is_better(self.current_etree_node.root_value, self.incumbent_value):

                # update queue
                is_queue_updated = self.update_queue()

                if is_queue_updated:
                    expansion = self.choose_expansion()                

                else:

                    # check whether current policy is a solution or not. (if any leaf node is not terminal, it is not solution)
                    # (and that can happen when no queue was added in the update_queue function becuase all of queue were expanded before)
                    complete_flag = 1
                    for node in self.opennodes:
                        if node.terminal != True:
                            complete_flag = 0
                            break

                    if complete_flag == 1 and self.graph.root.terminal != True:  # if solution is feasible
                        # if current feasible solution is better than incumbent, set it as incumbent
                        if self.is_better(self.graph.root.value, self.incumbent_value):                   
                            self.set_incumbent()

                            print(self.incumbent_value)
                            print(self.graph.root.exec_risk)
                            p, node_num = self.extract_policy()
                            self.incumbent_value_list.append([self.incumbent_value, self.graph.root.exec_risk, node_num, time.time()-self.start_time])
                            print(len(self.queue))
                            # time.sleep(1)
                            
                            # After setting the incumbent solution, remove the dominated solutions in the queue list.
                            self.queue = list(filter(lambda item: self.is_better(item[0]['current_etree_node'].root_value, self.incumbent_value), self.queue))
                            
                    expansion = self.choose_expansion()

            else:
                self.pruning_count += 1
                expansion = self.choose_expansion()

                        
        print('\n RAO* finished planning in ' +
              "{0:.2f}".format(time.time() - self.start_time) + " seconds after " + str(count) + " iterations\n")
        policy = self.extract_policy()

        return policy, self.graph

    def set_incumbent(self):
        self.incumbent_value = self.graph.root.value

    def init_search(self, b0):
        
        # initializes the search fields (initialize graph with start node)       
        self.graph = RAOStarHyperGraph(name='G')
        start_node = RAOStarGraphNode(name=str(b0), value=None, state=BeliefState(b0))
        
        self.set_new_node(start_node, 0, self.cc, 1.0, 1.0)
        self.debug('root node:')
        self.debug(start_node.state.state_print() + " risk bound: " +
                   str(start_node.exec_risk_bound))
        self.graph.add_node(start_node)
        self.graph.set_root(start_node)

        # initializes the search fields (initialize enumeration tree with root node)
        self.etree = EnumTree(name='T')
        root_etree_node = EnumTreeNode(parent_etree_node=None, root_value=self.graph.root.value)
        self.etree.add_node(root_etree_node)
        self.current_etree_node = root_etree_node

        # initialize the queue
        self.update_queue()
        

    def set_new_node(self, node, depth, er_bound, prob, parent_likelihood):
        # sets the fields of a new node
        b = node.state.belief
        node.risk = bound_prob(avg_func(b, self.r))
        # Depth of a node is its dist to the root
        node.depth = depth
        node.set_prob(prob)
        node.set_likelihood(prob * parent_likelihood)
        
        # Probability of violating constraints in a belief state. (never
        # change)
        if is_terminal_belief(b, self.term, self.terminal_prob):
            self.set_terminal_node(node)
        elif node.depth == self.fixed_horizon:
            # self.set_terminal_node(node)
            node.value = avg_func(b, self.h)
            node.terminal = True  # new node is non terminal
            node.best_action = None  # no action associated yet
            node.exec_risk_bound = bound_prob(
                er_bound)  # execution risk bound
            # avg heuristic estimate of execution risk at node
            node.set_exec_risk(node.risk)
            node.risk_upper = node.exec_risk                
        else:
            # the value of a node is the average of the heuristic only when it's
            # first created. After that, the value is given as a function of
            # its children
            node.value = avg_func(b, self.h)
            node.terminal = False  # new node is non terminal
            node.best_action = None  # no action associated yet
            node.exec_risk_bound = bound_prob(
                er_bound)  # execution risk bound
            # avg heuristic estimate of execution risk at node
            node.set_exec_risk(node.risk)

    def set_terminal_node(self, node):
        # set fields of a terminal node
        b = node.state.belief
        node.set_terminal(True)
        node.set_value(avg_func(b, self.h))
        node.set_exec_risk(node.risk)
        node.set_best_action(None)
        self.graph.remove_all_hyperedges(node)  # terminal node has no edges

    def set_deadend_terminal_node(self, node):
        # set fields of a terminal node
        b = node.state.belief 
        
        self.current_etree_node.increment_diff(node, 'value_diff', node.value, avg_func(b, self.h))
        self.current_etree_node.increment_diff(node, 'exec_risk_diff', node.exec_risk, 1.0)
        self.current_etree_node.increment_diff(node, 'prev_best_action', node.best_action, None)
        self.current_etree_node.increment_diff(node, 'current_best_action', node.best_action, None)
        self.current_etree_node.diff[node]['deadend_checked'] = True
        
        node.set_terminal(True)
        node.set_value(avg_func(b, self.h))
        node.set_exec_risk(1.0)
        node.set_best_action(None)

        node.value = np.inf

        # this can be not terminal for another enumeration. So we must not remove hyperedges.
        # self.graph.remove_all_hyperedges(node)  # terminal node has no edges
        
    def update_queue(self):
        self.debug('\n\n******* updating queue list *******')

        # start with finding opennodes
        self.update_policy_open_nodes()

        queue_list = []

        # add all of the combinations of opennodes and possible actions to queue list
        # dominated or infeasible actions are ignored.
        # TODO: More sophisticated dominance rules should be investigated
        for node in self.opennodes:
            # get node's properties
            belief = node.state.belief  # belief state associated to the node
            parent_risk = node.risk  # execution risk for current node
            parent_bound = node.exec_risk_bound  # er bound for current node
            parent_depth = node.depth  # dist of parent to root
            parent_likelihood = node.likelihood  # likelihood that node is reached in policy

            cost_vector = [] # list of cost vector for computing pareto front node-action pairs.
            queue_list_temp = [] # temporary queue list only for this node.
            
            # if node was expanded before, reuse prev operators
            ops = self.graph.all_node_operators(node)
            if len(ops)>0:
                for op in ops:
                    prob = op.properties['prob']
                    prob_safe = op.properties['prob_safe']
                    children = self.graph.hyperedge_successors(node, op)
                    exec_risk = parent_risk + (1.0 - parent_risk) * np.sum([p * child.exec_risk for (p, child) in zip(prob_safe, children)])
                    Q = op.op_value + np.sum([p * child.value for (p, child) in zip(prob, children)])

                    if exec_risk <= parent_bound:

                        # check whether this queue was expanded before. 
                        expanded_flag = 0
                        for expanded_queue in self.expanded_queue_list:
                            if expanded_queue['expansion_node_name'] == node.name and \
                               expanded_queue['expansion_hyperedge_name'] == op.name and \
                               expanded_queue['er_bound'] == node.exec_risk_bound:

                                expanded_flag = 1
                                break

                        # add queue only if this queue was not expanded before.
                        if expanded_flag == 0:
                            queue_list_temp.append({'current_etree_node':self.current_etree_node,'expansion_node':node, 'expansion_hyperedge':op, 'Q_value':Q})

                            if self.model.optimization == 'maximize':
                                cost_vector.append([-Q, node.exec_risk])
                            elif self.model.optimization == 'minimize':
                                cost_vector.append([Q, node.exec_risk])

            else:
                if self.cc_type == 'everywhere':
                    parent_bound = self.cc

                # if the current node is guaranteed to violate constraints and a violation
                # is set to halt process: make node terminal
                if self.halt_on_violation and np.isclose(parent_risk, 1.0):
                    all_node_actions = []
                else:
                    # else get the available actions from model
                    all_node_actions = self.get_all_actions(belief, self.A)

                action_added = False  # flag if a new action has been added

                if len(all_node_actions) > 0:
                    added_count = 0
                    for act in all_node_actions:
                        self.debug("\n", act)

                        child_obj_list, prob_list, prob_safe_list, new_child_idxs = self.obtain_child_objs_and_probs(belief, self.T, self.O, self.r, act)

                        # initializes the new child nodes
                        for c_idx in new_child_idxs:
                            self.set_new_node(
                                child_obj_list[c_idx], parent_depth + 1, 0.0, prob_list[c_idx], parent_likelihood)

                        child_er_bounds, er_bound_infeasible = self.compute_exec_risk_bounds(parent_bound, parent_risk, child_obj_list, prob_safe_list)

                        # updates the values of the execution risk for all children
                        # that will be added to the graph
                        for idx, child in enumerate(child_obj_list):
                            child.exec_risk_bound = child_er_bounds[idx]

                        # average instantaneous value (cost or reward)
                        avg_op_value = avg_func(belief, self.V, act)

                        act_obj = RAOStarGraphOperator(name=str(act), op_value=avg_op_value,
                                                       properties={'prob': prob_list, 'prob_safe': prob_safe_list})
                        # an "Action" object crerated
                        # add edge (Action) to graph
                        self.graph.add_hyperedge(
                            parent_obj=node, child_obj_list=child_obj_list, prob_list=prob_list, op_obj=act_obj)

                        
                        prob_safe = act_obj.properties['prob_safe']
                        children = self.graph.hyperedge_successors(node, act_obj)
                        exec_risk = parent_risk + (1.0 - parent_risk) * np.sum([p * child.exec_risk for (p, child) in zip(prob_safe, children)])

                        Q = act_obj.op_value + np.sum([p * child.value for (p, child) in zip(prob_list, child_obj_list)])
                    
                        if exec_risk <= parent_bound:

                            action_added = True

                            # check whether this queue was expanded before. 
                            expanded_flag = 0
                            for expanded_queue in self.expanded_queue_list:
                                if expanded_queue['expansion_node_name'] == node.name and \
                                   expanded_queue['expansion_hyperedge_name'] == act_obj.name and \
                                   expanded_queue['er_bound'] == node.exec_risk_bound:

                                    expanded_flag = 1
                                    break

                            # add queue only if this queue was not expanded before.
                            if expanded_flag == 0:
                                queue_list_temp.append({'current_etree_node':self.current_etree_node,'expansion_node':node, 'expansion_hyperedge':act_obj, 'Q_value':Q})

                                if self.model.optimization == 'maximize':
                                    cost_vector.append([-Q, node.exec_risk])
                                elif self.model.optimization == 'minimize':
                                    cost_vector.append([Q, node.exec_risk])

                if not action_added:
                    # self.debug('action not added')
                    # self.set_terminal_node(node)

                    # if not is_terminal_belief(node.state.belief, self.term, self.terminal_prob):
                    #         self.mark_deadend(node)

                    if not node.terminal:
                        self.set_deadend_terminal_node(node)
                        queue_list_temp.append({'current_etree_node':self.current_etree_node,'expansion_node':node, 'expansion_hyperedge':None, 'Q_value':None})

            # cost_vector = np.array(cost_vector)
            # pareto_front = list(self.is_pareto(cost_vector))
            # queue_list_temp = [d for (d, remove) in zip(queue_list_temp, pareto_front) if remove]

            queue_list.extend(queue_list_temp)
            

        if len(queue_list)==0:
            return False
        else:
            self.queue.insert(0,queue_list)
            return True
        

    def update_policy_open_nodes(self):
        self.debug('\n\n******* updating policy open nodes *******')
        # self.debug(node.state.mean_b)
        # self.debug('******************************\n')
        # traverse graph starting at root along marked actions, recording ancestors
        # and open nodes
        # starts at root and expands nodes with policy and
        queue = deque([self.graph.root])
        # add nodes with no policy yet to opennodes
        # policy ancestors={}
        self.opennodes = set()
        self.leafnodes = set()
        expanded = []  # not to be mistaken with the expanded list used in dynamic programming
        # simply keep track of the nodes we have expanded before so it doesn't loop forever
        # self.debug(n.)
        # self.debug(queue)

        self.graph.reset_likelihoods()
        self.graph.root.likelihood = 1.0

        while len(queue) > 0:
            node = queue.popleft()

            if node in expanded:
                continue

            parent_l = node.likelihood
            self.debug('\n', node.state.state_print(),
                       'likelihood:', parent_l)

            if node.best_action != None:  # node already has a best action
                self.debug('best_action:', node.best_action.name)
                expanded.append(node)
                children = self.graph.hyperedge_successors(
                    node, node.best_action)
                self.debug("children risk bound", [
                           c.exec_risk_bound for c in children])
                for n in children:
                    transition = None
                    for parent in self.graph.all_node_ancestors(n):
                        if parent[0] == node and parent[1] == node.best_action:
                            transition = parent[2]
                            break
                    print(n.state.state_print(), n.likelihood,
                          parent_l, transition)

                    n.likelihood = parent_l * transition
                    if n not in expanded:
                        queue.append(n)
            else:  # no best action has been assigned yet
                self.debug('leaf node!')
                self.leafnodes.add(node)
                if not node.terminal:
                    self.opennodes.add(node)

        leaf_likelihood = sum(leaf.likelihood for leaf in self.leafnodes)
        if not np.isclose(leaf_likelihood, 1.0):
            raise ValueError('Leaf belief likelihood error, sum is ' +
                             str(leaf_likelihood) + ' and should be 1.0')

        # Sum the likelihood of nonterminal leaf nodes
        # we can terminate search early when this sum + er(root) <= cc
        self.nonterminal_leaf_sum = sum(
            leaf.likelihood for leaf in self.opennodes)
        # add the nonterminal_leaf_sum to the er(root)
        self.likelihood_termination = self.nonterminal_leaf_sum + self.graph.root.exec_risk

        self.debug('************* root risk ', self.graph.root.exec_risk)

        self.debug('\n', 'Selected opennodes:', self.opennodes)

        
    def get_all_actions(self, belief, A):
        if len(belief) > 0:
            all_node_actions = []
            action_ids = set()  # Uses str(a) as ID
            for particle_key, particle_prob in belief.items():
                new_actions = [a for a in A(
                    particle_key) if not str(a) in action_ids]
                # add action and make sure no overlap
                all_node_actions.extend(new_actions)
                action_ids.update([str(a) for a in new_actions])
            return all_node_actions
        else:
            return []

    def expand_etree_node(self, expansion):

        # initialize new etree node and add it to the etree
        new_etree_node = EnumTreeNode(parent_etree_node=self.current_etree_node)
        self.etree.add_node(new_etree_node)
        self.current_etree_node = new_etree_node

        exp_node = expansion['expansion_node']
        exp_action = expansion['expansion_hyperedge']
                
        Z = self.build_ancestor_list(exp_node)
        
        # updates the best action at the node
        for node in Z:
            self.debug('\nupdate values and best action: ' +
                       str(node.state.state_print()))
            self.debug('current Q: ', node.value, "\n")

            if node == exp_node:
                # set best action of current expanding node as current expanding action
                # this is different from original RAO*, becuase we don't choose "best" action anymore.
                all_action_operators = [exp_action]
            else:
                # get all actions available at that node, if the node is an ancestor of the expanding node
                # get all actions (operators) of node from graph
                all_action_operators = [] if node.terminal else self.graph.all_node_operators(node)


            if exp_action==None:
                continue
                
            # risk at the node's belief state (does no depend on the action
            # taken)
            risk = node.risk
            # current *admissible* (optimistic) estimate of the node's Q
            # value
            current_Q = node.value
            # execution risk bound. the execution risk cap depends on type of chance
            # constraint being imposed
            er_bound = min([node.exec_risk_bound, self.er_cap])
            if self.cc_type == 'everywhere':
                er_bound = self.er_cap

            best_action_idx = -1
            best_Q = self.initial_Q  # -inf or inf based on optimization
            best_D = -1  # depth
            exec_risk_for_best = -1.0

            # Estimates value and risk of the current node for each
            # possible action
            for act_idx, act in enumerate(all_action_operators):
                probs = act.properties['prob']
                prob_safe = act.properties['prob_safe']
                children = self.graph.hyperedge_successors(node, act)
                # estimate Q of taking this action from current node. Composed of
                # current reward and the average reward of its children
                Q = act.op_value + \
                    np.sum([p * child.value for (p, child)
                            in zip(probs, children)])
                # Average child depth
                D = 1 + np.sum([p * child.depth for (p, child)
                                in zip(probs, children)])

                # compute an estimate of the er of taking this action from current node.
                # composed of the current risk and the avg execution risk
                # of its children
                if self.cc_type == 'overall':
                    exec_risk = risk + (1.0 - risk) * np.sum([p * child.exec_risk for (p, child)
                                                              in zip(prob_safe, children)])
                # enforcing same risk bound at all steps in the policy
                elif self.cc_type == 'everywhere':
                    exec_risk = risk
                    
                self.debug('action: ' + act.name + ' children: ' + str(children[0].state.state_print()) +
                           ' risk ' + str(exec_risk))
                self.debug('  act_op_value: ', act.op_value)

                for child in children:
                    self.debug(' child_value: ', child.value)

                self.debug('  children Q: ' + str(Q))

                # if execution risk bound has been violated or if Q value for this action is worse
                # than current best, we should definitely not select it.
                if (exec_risk > er_bound) or self.is_worse(Q, best_Q):
                    select_action = False
                    if(exec_risk > er_bound):
                        self.debug(' Action pruned by risk bound')
                # if risk bound respected and Q value is equal or better
                else:
                    select_action = True
                # Test if the risk bound for the current node has been
                # violated
                if select_action:
                    # Updates the execution risk bounds for the children
                    # child_er_bounds, cc_infeasible = self.compute_exec_risk_bounds(
                    #     er_bound, risk, children, prob_safe)


                    # Sungkweon: I think this step in unnecessary. Should be checked though.
                    # for child in children:
                    #     self.debug('  select_action: child ' + child.state.state_print() + " depth: " + str(child.depth) +
                    #                " risk bound: " + str(child.exec_risk_bound) + ' infeasible: ' + str(cc_infeasible))
                    # if not cc_infeasible:  # if chance constraint has not been violated
                    #     for idx, child in enumerate(children):
                    #         child.exec_risk_bound = child_er_bounds[idx]

                    # Updates the best action at node
                    best_Q = Q
                    best_action_idx = act_idx
                    best_D = D
                    exec_risk_for_best = exec_risk
                    
            # Test if some action has been selected
            if best_action_idx >= 0:
                if (not np.isclose(best_Q, current_Q)) and self.is_better(best_Q, current_Q):
                    print('current_Q', current_Q, 'best_Q', best_Q)

                    print(
                        'WARNING: node Q value improved, which might indicate inadmissibility.')

                er_bound_updating_nodes = deque([node])

                while len(er_bound_updating_nodes) > 0:
                    updating_node = er_bound_updating_nodes.popleft()

                    if updating_node.best_action:
                        best_action_updating = updating_node.best_action
                        er_bound_updating = updating_node.exec_risk_bound
                        risk_updating = updating_node.risk
                        probs_updating = best_action_updating.properties['prob']
                        prob_safe_updating = best_action_updating.properties['prob_safe']
                        children_updating = self.graph.hyperedge_successors(updating_node, best_action_updating)
                        
                        child_er_bounds, er_bound_infeasible = self.compute_exec_risk_bounds_etree(er_bound_updating, risk_updating, children_updating, prob_safe_updating, new_etree_node)

                        for idx, child in enumerate(children_updating):
                            child.exec_risk_bound = child_er_bounds[idx]

                        er_bound_updating_nodes.extend(children_updating)
                
        
                # increment differences to be made by this update
                self.current_etree_node.increment_diff(node, 'value_diff', node.value, best_Q)
                self.current_etree_node.increment_diff(node, 'exec_risk_diff', node.exec_risk, exec_risk_for_best)
                self.current_etree_node.increment_diff(node, 'prev_best_action', node.best_action, all_action_operators[best_action_idx])
                self.current_etree_node.increment_diff(node, 'current_best_action', node.best_action, all_action_operators[best_action_idx])

                # updates optimal value est, execution tisk est, and mark
                # best action
                node.set_value(best_Q)
                node.set_exec_risk(exec_risk_for_best)
                node.set_best_action(all_action_operators[best_action_idx])
                self.debug('best action for ' + str(node.state.state_print()) + ' set as ' +
                           str(all_action_operators[best_action_idx].name))
            else:  # no action was selected, so this node is terminal
                self.debug('*\n*\n*\n*\n no best action for ' +
                           str(node.state.state_print()) + '\n*\n*\n*\n')
                if not node.terminal:
                    self.set_deadend_terminal_node(node)

        # update root value
        new_etree_node.root_value = self.graph.root.value
        
    # def expand_best_partial_solution(self,expansion):
    #     # expands a node in the graph currently contained in the best
    #     # partial solution. Add new nodes and edges on the graph

    #     node = expansion['expansion_node']
        
    #     self.debug('\n ******* expanding node *******')
    #     self.debug(node.state.state_print())
    #     # print(node.state.state_print())
    #     self.debug('******************************\n')
    #     belief = node.state.belief  # belief state associated to the node
    #     parent_risk = node.risk  # execution risk for current node
    #     parent_bound = node.exec_risk_bound  # er bound for current node
    #     parent_depth = node.depth  # dist of parent to root
    #     parent_likelihood = node.likelihood  # likelihood that node is reached in policy

    #     if self.cc_type == 'everywhere':
    #         parent_bound = self.cc

    #     self.debug('compute_exec_risk_bounds: parent_bound ',
    #                parent_bound, ' parent_risk ', parent_risk)

    #     # if the current node is guaranteed to violate constraints and a violation
    #     # is set to halt process: make node terminal
    #     if self.halt_on_violation and np.isclose(parent_risk, 1.0):
    #         all_node_actions = []
    #     else:
    #         # else get the available actions from model
    #         all_node_actions = self.get_all_actions(belief, self.A)

    #     action_added = False  # flag if a new action has been added

    #     if len(all_node_actions) > 0:
    #         added_count = 0
    #         for act in all_node_actions:
    #             self.debug("\n", act)

    #             child_obj_list, prob_list, prob_safe_list, new_child_idxs = self.obtain_child_objs_and_probs(belief, self.T, self.O, self.r, act)

    #             # initializes the new child nodes
    #             for c_idx in new_child_idxs:
    #                 self.set_new_node(
    #                     child_obj_list[c_idx], parent_depth + 1, 0.0, prob_list[c_idx], parent_likelihood)

    #             # if parent bound Delta is ~ 1.0, the child nodes are guaranteed to have
    #             # their risk bound equal to 1
    #             if (not np.isclose(parent_bound, 1.0)):
    #                 # computes execution risk bounds for the child nodes
    #                 er_bounds, er_bound_infeasible = self.compute_exec_risk_bounds(parent_bound,
    #                                                                                parent_risk, child_obj_list, prob_safe_list)
    #             else:
    #                 er_bounds = [1.0] * len(child_obj_list)
    #                 er_bound_infeasible = False

    #             # Only creates new operator if all er bounds a non-negative
    #             if not er_bound_infeasible:
    #                 # updates the values of the execution risk for all children
    #                 # that will be added to the graph
    #                 for idx, child in enumerate(child_obj_list):
    #                     child.exec_risk_bound = er_bounds[idx]

    #                 # average instantaneous value (cost or reward)
    #                 avg_op_value = avg_func(belief, self.V, act)

    #                 act_obj = RAOStarGraphOperator(name=str(act), op_value=avg_op_value,
    #                                                properties={'prob': prob_list, 'prob_safe': prob_safe_list})
    #                 # an "Action" object crerated
    #                 # add edge (Action) to graph
    #                 self.graph.add_hyperedge(
    #                     parent_obj=node, child_obj_list=child_obj_list, prob_list=prob_list, op_obj=act_obj)

    #                 action_added = True
    #                 added_count += 1
    #             else:
    #                 self.debug(
    #                     '  action not added - error bound infeasible')

    #     if not action_added:
    #         # self.debug('action not added')
    #         # self.set_terminal_node(node)

    #         if not is_terminal_belief(node.state.belief, self.term, self.terminal_prob):
    #                 self.mark_deadend(node)

    #         if not node.terminal:
    #             self.set_terminal_node(node)

    #     # returns the list of node either added actions to or marked terminal
    #     return nodes_to_expand

    # def update_values_and_best_actions(self, expanded_nodes):
    #     # updates the Q values on nodes on the graph and the current best policy
    #     # for each expanded node at a time
    #     self.debug('\n ****************************')
    #     self.debug('Update values and best actions  ')
    #     self.debug('****************************')

    #     for exp_idx, exp_node in enumerate(expanded_nodes):
    #         Z = self.build_ancestor_list(exp_node)
    #         # updates the best action at the node
    #         for node in Z:
    #             self.debug('\nupdate values and best action: ' +
    #                        str(node.state.state_print()))
    #             self.debug('current Q: ', node.value, "\n")

    #             # all actions available at that node
    #             all_action_operators = [
    #             ] if node.terminal else self.graph.all_node_operators(node)
    #             # get all actions (operators) of node from graph
    #             # risk at the node's belief state (does no depend on the action
    #             # taken)
    #             risk = node.risk
    #             # current *admissible* (optimistic) estimate of the node's Q
    #             # value
    #             current_Q = node.value
    #             # execution risk bound. the execution risk cap depends on type of chance
    #             # constraint being imposed
    #             er_bound = min([node.exec_risk_bound, self.er_cap])
    #             if self.cc_type == 'everywhere':
    #                 er_bound = self.er_cap
                    
    #             best_action_idx = -1
    #             best_Q = self.initial_Q  # -inf or inf based on optimization
    #             best_D = -1  # depth
    #             exec_risk_for_best = -1.0

    #             # Estimates value and risk of the current node for each
    #             # possible action
    #             for act_idx, act in enumerate(all_action_operators):
    #                 probs = act.properties['prob']
    #                 prob_safe = act.properties['prob_safe']
    #                 children = self.graph.hyperedge_successors(node, act)
    #                 # estimate Q of taking this action from current node. Composed of
    #                 # current reward and the average reward of its children
    #                 Q = act.op_value + \
    #                     np.sum([p * child.value for (p, child)
    #                             in zip(probs, children)])
    #                 # Average child depth
    #                 D = 1 + np.sum([p * child.depth for (p, child)
    #                                 in zip(probs, children)])

    #                 # compute an estimate of the er of taking this action from current node.
    #                 # composed of the current risk and the avg execution risk
    #                 # of its children
    #                 if self.cc_type == 'overall':
    #                     exec_risk = risk + (1.0 - risk) * np.sum([p * child.exec_risk for (p, child)
    #                                                               in zip(prob_safe, children)])
    #                 # enforcing same risk bound at all steps in the policy
    #                 elif self.cc_type == 'everywhere':
    #                     exec_risk = risk

    #                 self.debug('action: ' + act.name + ' children: ' + str(children[0].state.state_print()) +
    #                            ' risk ' + str(exec_risk))
    #                 self.debug('  act_op_value: ', act.op_value)

    #                 for child in children:
    #                     self.debug(' child_value: ', child.value)

    #                 self.debug('  children Q: ' + str(Q))

    #                 # if execution risk bound has been violated or if Q value for this action is worse
    #                 # than current best, we should definitely not select it.
    #                 if (exec_risk > er_bound) or self.is_worse(Q, best_Q):
    #                     select_action = False
    #                     if(exec_risk > er_bound):
    #                         self.debug(' Action pruned by risk bound')
    #                 # if risk bound respected and Q value is equal or better
    #                 else:
    #                     select_action = True
    #                 # Test if the risk bound for the current node has been
    #                 # violated
    #                 if select_action:
    #                     # Updates the execution risk bounds for the children
    #                     child_er_bounds, cc_infeasible = self.compute_exec_risk_bounds(
    #                         er_bound, risk, children, prob_safe)
    #                     for child in children:
    #                         self.debug('  select_action: child ' + child.state.state_print() + " depth: " + str(child.depth) +
    #                                    " risk bound: " + str(child.exec_risk_bound) + ' infeasible: ' + str(cc_infeasible))
    #                     if not cc_infeasible:  # if chance constraint has not been violated
    #                         for idx, child in enumerate(children):
    #                             child.exec_risk_bound = child_er_bounds[idx]

    #                         # Updates the best action at node
    #                         best_Q = Q
    #                         best_action_idx = act_idx
    #                         best_D = D
    #                         exec_risk_for_best = exec_risk
    #             # Test if some action has been selected
    #             if best_action_idx >= 0:
    #                 if (not np.isclose(best_Q, current_Q)) and self.is_better(best_Q, current_Q):
    #                     print('current_Q', current_Q, 'best_Q', best_Q)

    #                     print(
    #                         'WARNING: node Q value improved, which might indicate inadmissibility.')

    #                 # updates optimal value est, execution tisk est, and mark
    #                 # best action
    #                 node.set_value(best_Q)
    #                 node.set_exec_risk(exec_risk_for_best)
    #                 node.set_best_action(all_action_operators[best_action_idx])
    #                 self.debug('best action for ' + str(node.state.state_print()) + ' set as ' +
    #                            str(all_action_operators[best_action_idx].name))
    #             else:  # no action was selected, so this node is terminal
    #                 self.debug('*\n*\n*\n*\n no best action for ' +
    #                            str(node.state.state_print()) + '\n*\n*\n*\n')

    #                 # mdeyo: Finally got plans with deadends to work!
    #                 # Deadends = state with no actions available, either
    #                 # because it's an actual deadend or because all actons were
    #                 # too risky.
    #                 # If the deadend was on the optimal path, the planner would
    #                 # just mark it terminal and planning would end before
    #                 # the goal was achieved

    #                 # mdeyo: Current fix is just to mark the deadend state as
    #                 # having execution risk = 1.0 so that the planner will
    #                 # remove the previous action from policy and properly pick
    #                 # the next best action at the parent state
    #                 # node.risk = 1.0
    #                 # node.set_exec_risk(node.risk)

    #                 # mdeyo: alternative, possibly better fix is to update the
    #                 # value instead of the risk, setting the value to +inf when
    #                 # minimizing

    #                 # only mark inf value deadend if not actually the goal
    #                 if not is_terminal_belief(node.state.belief, self.term, self.terminal_prob):
    #                         self.mark_deadend(node)

    #                 if not node.terminal:
    #                     self.set_terminal_node(node)

    #                 # mdeyo: some further testing shows that both these
    #                 # solutions to deadends seem to have the same resulting
    #                 # policies, while updating the cost ends up in a faster
    #                 # search, probably because the Q value prunes the option
    #                 # before risk calculations which are more expensive

    def mark_deadend(self, node):
        # choose the comparison function depending on the type of search
        if self.model.optimization == 'maximize':
            node.value = -np.inf
        elif self.model.optimization == 'minimize':
            node.value = np.inf
        return node

    def compute_exec_risk_bounds(self, parent_bound, parent_risk, child_list, prob_safe_list, is_terminal_node=False):
        # computes the execution risk bounds for each sibling in a list of
        # children of a node
        # msg = 'compute_exec_risk_bounds: parent ' + str(parent_bound) + ' risk ' + str(parent_risk) + ' child_list ' + str(
        #     child_list) + ' prob_safe_list ' + str(prob_safe_list) + ' terminal ' + str(is_terminal_node)
        # self.debug(msg)
        exec_risk_bounds = [0.0] * len(child_list)
        # If the parent bound is almost one, the risk of the children are
        # guaranteed to be feasible
        if np.isclose(parent_bound, 1.0):
            exec_risk_bounds = [1.0] * len(child_list)
            infeasible = False
            self.debug('parent bound close to 1!')
        else:
            # if parent bound isn't one, but risk is almost one, or if parent already violates the risk bound
            # don't try to propagate, since children guaranteed to violate
            if np.isclose(parent_risk, 1.0) or (parent_risk > parent_bound):
                infeasible = True
            # Only if the parent bound and the parent risk are below 1, and the parent risk is below the parent bound,
            # then try to propagate risk
            else:
                infeasible = False
                # risk "consumed" by parent node
                parent_term = (parent_bound - parent_risk) / \
                    (1.0 - parent_risk)

                for idx_child, child in enumerate(child_list):
                    # Risk consumed by the siblings of the current node
                    sibling_term = np.sum(
                        [p * c.exec_risk for (p, c) in zip(prob_safe_list, child_list) if (c != child)])
                    # self.debug('sibling term:' + str(sibling_term))
                    # self.debug('first in min' + str((parent_term -
                    # sibling_term) / prob_safe_list[idx_child]))

                    # exec risk bound, which caps ar 1.0
                    # if self.cc_type == 'overall':
                    exec_risk_bound = min(
                        [(parent_term - sibling_term) / prob_safe_list[idx_child], 1.0])
                    # elif self.cc_type == 'everywhere':
                    # exec_risk_bound = self.cc
                    # else:
                    # A negative bound means that the chance constraint is guaranteed
                    # to be violated. The same is true if the admissible estimate
                    # of the execution risk for a child node violates its upper
                    # bound.
                    if exec_risk_bound >= 0.0:
                        self.debug('  child_exec_risk:', child.exec_risk,
                                   'child_exec_risk_bound', exec_risk_bound)
                        if child.exec_risk <= exec_risk_bound or np.isclose(child.exec_risk, exec_risk_bound):
                            exec_risk_bounds[idx_child] = exec_risk_bound
                        else:
                            self.debug('  INFEASIBLE: risk exceeds bound')
                            infeasible = True
                            break
                    else:
                        self.debug('  INFEASIBLE: impossible risk bound')
                        infeasible = True
                        break
        return exec_risk_bounds, infeasible

    def compute_exec_risk_bounds_etree(self, parent_bound, parent_risk, child_list, prob_safe_list, etree_node, is_terminal_node=False):
        # computes the execution risk bounds for each sibling in a list of
        # children of a node
        # msg = 'compute_exec_risk_bounds: parent ' + str(parent_bound) + ' risk ' + str(parent_risk) + ' child_list ' + str(
        #     child_list) + ' prob_safe_list ' + str(prob_safe_list) + ' terminal ' + str(is_terminal_node)
        # self.debug(msg)
        exec_risk_bounds = [0.0] * len(child_list)
        # If the parent bound is almost one, the risk of the children are
        # guaranteed to be feasible
        if np.isclose(parent_bound, 1.0):
            exec_risk_bounds = [1.0] * len(child_list)
            infeasible = False
            self.debug('parent bound close to 1!')
        else:
            # if parent bound isn't one, but risk is almost one, or if parent already violates the risk bound
            # don't try to propagate, since children guaranteed to violate
            if np.isclose(parent_risk, 1.0) or (parent_risk > parent_bound):
                infeasible = True
            # Only if the parent bound and the parent risk are below 1, and the parent risk is below the parent bound,
            # then try to propagate risk
            else:
                infeasible = False
                # risk "consumed" by parent node
                parent_term = (parent_bound - parent_risk) / \
                    (1.0 - parent_risk)

                for idx_child, child in enumerate(child_list):
                    # Risk consumed by the siblings of the current node
                    sibling_term = np.sum(
                        [p * c.exec_risk for (p, c) in zip(prob_safe_list, child_list) if (c != child)])

                    exec_risk_bound = min(
                        [(parent_term - sibling_term) / prob_safe_list[idx_child], 1.0])

                    # A negative bound means that the chance constraint is guaranteed
                    # to be violated. The same is true if the admissible estimate
                    # of the execution risk for a child node violates its upper
                    # bound.
                    if exec_risk_bound >= 0.0:
                        self.debug('  child_exec_risk:', child.exec_risk,
                                   'child_exec_risk_bound', exec_risk_bound)
                        if child.exec_risk <= exec_risk_bound or np.isclose(child.exec_risk, exec_risk_bound):
                            exec_risk_bounds[idx_child] = exec_risk_bound
                            etree_node.increment_diff(child, 'er_bound_diff', child.exec_risk_bound, exec_risk_bound)

                        else:
                            self.debug('  INFEASIBLE: risk exceeds bound')
                            infeasible = True
                            break
                    else:
                        self.debug('  INFEASIBLE: impossible risk bound')
                        infeasible = True
                        break
        return exec_risk_bounds, infeasible
    
    def build_ancestor_list(self, expanded_node):
        # create set Z that contains the expanded node and all of its ancestors in the explicit graph
        # along marked action arcs (ancestors nodes from best policy)
        # self.debug('build ancestor of: ', expanded_node.name)
        Z = []
        queue = deque([expanded_node])
        while len(queue) > 0:
            node = queue.popleft()
            if node not in Z:
                Z.append(node)
                for parent in self.graph.all_node_ancestors(node):
                    parent = parent[0]
                    if not parent.terminal and parent.best_action != None:
                        if node in self.graph.hyperedge_successors(parent, parent.best_action):
                            queue.append(parent)
        return Z

    def extract_policy(self):
        # extract policy mapping nodes to actions
        # self.debug("===========================")
        # self.debug("=== Extract Policy ========")
        # self.debug("===========================")
        queue = deque([self.graph.root])  # from root
        policy = {}
        k=0
        while len(queue) > 0:
            node = queue.popleft()
            if node.best_action != None:
                policy[node.name] = node.best_action.name
                children = self.graph.hyperedges[node][node.best_action]
                for c in children:
                    queue.append(c)
                    k=k+1
        print(k)
        return policy, k

    def choose_expansion(self):
        # f = open("temp.txt", "a+")
        # f.write("----------------\n")
        # for i in self.queue:
        #     f.write(str(len(i))+",")

        # f.write("\n")

        
        # chooses an element from queue list to be expanded

        # deep copy of the first list of queue list.
        while True:
            if len(self.queue)>0:
                if len(self.queue[0])==0:
                    del self.queue[0]
                else:
                    queue = self.queue[0][:]
                    break
            else:
                raise ValueError('Error: Queue is empty')
                
        if len(queue) > 1:
            # sorting expansions with best nodes first.
            queue_with_best_node = []
            
            if self.model.optimization == 'maximize':
                best_node_value = -np.inf
            elif self.model.optimization == 'minimize':
                best_node_value = np.inf                    
                
            while len(queue)>0:
                etree_node = self.select_best_node(queue)
                if self.is_worse(etree_node['expansion_node'].value, best_node_value):
                    break
                else:
                    queue_with_best_node.append(etree_node)
                    best_node_value = etree_node['expansion_node'].value
                    queue.remove(etree_node)

            # Then, find the expansion with best action
            expansion = self.select_best_hyperedge(queue_with_best_node)
            self.queue[0].remove(expansion)
        else:  # if there is only one, use that one
            expansion = queue.pop()
            del self.queue[0]
            

        self.expanded_queue_list.append({'expansion_node_name':expansion['expansion_node'].name,
                                    'expansion_hyperedge_name':expansion['expansion_hyperedge'].name,
                                    'er_bound':expansion['expansion_node'].exec_risk_bound})

        # for i in self.queue:
        #     f.write(str(len(i))+",")

        # f.write("\n")
        # f.write("----------------\n")
        # f.close()

        # f = open("result2.txt", "a+")
        # f.write("-------------------\n")
        # f.write(expansion['expansion_node'].state.state_print())
        # f.write("\n")
        # f.write(expansion['expansion_hyperedge'].name)
        # f.write("\n")
        return expansion

    def choose_most_likely_leaf(self):
        # chooses the most likely element from open list to be expanded
        # these elements in opennodes are all non-terminal leafs in the policy
        if len(self.opennodes) > 1:
            most_likely = None
            likelihood = 0
            for node in self.opennodes:
                if node.likelihood > likelihood:
                    most_likely = node
                    likelihood = node.likelihood
            if not most_likely:
                raise ValueError(
                    'failed to identify most likely non-terminal leaf')
            node = most_likely
            # selects best node to expand
            # node = self.select_best(self.opennodes)  # select best node
            # self.opennodes.remove(node)
        else:  # if there is only one, use that one
            node = self.opennodes.pop()
        return node

    def obtain_child_objs_and_probs(self, belief, T, O, r, act):
        # predicts new particles using current belief and state transition
        # mdeyo: pred_belief_safe is not being used
        pred_belief, pred_belief_safe = predict_belief(belief, T, r, act)
        # Given the predicted belief, computes the probability distribution of potential observations.
        # Each observations corresponds to a new node on the hypergraph, whose edge is annotated by the
        # prob of that particular observation
        obs_distribution, obs_distribution_safe, state_to_obs = compute_observation_distribution(
            pred_belief, pred_belief_safe, O)
        # for each observation, computes corresponding updated belief
        # self.debug(obs_distribution)
        # self.debug(obs_distribution_safe)
        # self.debug(state_to_obs)
        child_obj_list = []
        prob_list = []
        prob_safe_list = []
        new_child_idxs = []
        count = 0
        for obs, obs_prob in obs_distribution.items():
            # Performs belief state update
            child_blf_state = update_belief(pred_belief, state_to_obs, obs)
            candidate_child_obj = RAOStarGraphNode(
                name=str(child_blf_state), value=None, state=BeliefState(child_blf_state))
            if self.graph.has_node(candidate_child_obj):  # if node already present
                child_obj = self.graph.nodes[candidate_child_obj.name]
            else:
                # the node initiated
                child_obj = candidate_child_obj
                new_child_idxs.append(count)
            child_obj_list.append(child_obj)
            prob_list.append(obs_prob)

            if obs in obs_distribution_safe:
                obs_safe_prob = obs_distribution_safe[obs]
                prob_safe_list.append(obs_safe_prob)
            else:
                prob_safe_list.append(0.0)
            count += 1
        return child_obj_list, prob_list, prob_safe_list, new_child_idxs

    
    def is_pareto(self, costs, maximise=False):
        """
        :param costs: An (n_points, n_costs) array
        :maximise: boolean. True for maximising, False for minimising
        :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
        """
        is_efficient = np.ones(costs.shape[0], dtype = bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                if maximise:
                    is_efficient[is_efficient] = np.any(costs[is_efficient]>=c, axis=1)  # Remove dominated points
                else:
                    is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)  # Remove dominated points
        return is_efficient
