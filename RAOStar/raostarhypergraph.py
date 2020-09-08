#!/usr/bin/env python
#
#  Copyright (c) 2015 MIT. All rights reserved.
#
#   author: Pedro Santana
#   e-mail: psantana@mit.edu
#   website: people.csail.mit.edu/psantana
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#  1. Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in
#     the documentation and/or other materials provided with the
#     distribution.
#  3. Neither the name(s) of the copyright holders nor the names of its
#     contributors or of the Massachusetts Institute of Technology may be
#     used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
#  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
#  AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
"""
RAO*: a risk-aware planner for POMDP's.

Defines the structures used in the RAO* hypergraph.

@author: Pedro Santana (psantana@mit.edu).
@editor: Yun Chang (yunchang@mit.edu)
"""


import sys
from collections import deque


class GraphElement(object):                                               #graph name and properties
    """
    Generic graph element with a name and a unique ID.
    """

    def __init__(self, name=None, properties={}):
        self.name = name
        self.properties = properties

    __hash__ = object.__hash__

    def set_name(self, new_name):
        self.name = new_name

    def set_properties(self, new_properties):
        if isinstance(new_properties, dict):
            self.properties = new_properties
        else:
            raise TypeError(
                'Hypergraph element properties should be given as a dictionary.')

    def __eq__(x, y):
        return isinstance(x, GraphElement) and isinstance(y, GraphElement) and (x.name == y.name)

    def __ne__(self, other):
        return not self == other


class RAOStarGraphNode(GraphElement):                                   #node class 
    """
    Class for nodes in the RAO* hypergraph.
    """

    def __init__(self, value, state, best_action=None, terminal=False, deadend=False, name=None,        #init node with (q val, flags, state, best action, curr risk, action risk, risk bound)
                 properties={}, make_unique=False):
        super(RAOStarGraphNode, self).__init__(name, properties)
        self.value = value  # Q value
        self.terminal = terminal  # Terminal flag
        self.deadend = deadend # Deadend flag. Note that deadend is different from terminal. (added by Sungkweon)
        self.state = state  # Belief state
        self.best_action = best_action  # Best action at the node

        self.risk = 0.0  # Belief state risk
        self.exec_risk = 0.0  # Execution risk
        self.exec_risk_bound = 1.0  # Bound on execution risk
        self.depth = 0  # Initial depth
        self.probability = 0.0
        self.likelihood = 0.0
    __hash__ = object.__hash__

    def __eq__(x, y):
        return isinstance(x, GraphElement) and isinstance(y, GraphElement) and (x.name == y.name) and (x.depth == y.depth)
        

    def set_likelihood(self, l):                          #set attributes function
        self.likelihood = l

    def set_prob(self, prob):
        self.probability = prob

    def set_depth(self, new_depth):
        """Sets new non-negative depth"""
        if new_depth >= 0:
            self.depth = new_depth
        else:
            raise ValueError('Node depths must be non-negative!')

    def set_value(self, new_value):
        self.value = new_value

    def set_state(self, new_state):
        self.state = new_state

    def set_best_action(self, new_best_action):
        if new_best_action == None or isinstance(new_best_action, RAOStarGraphOperator):
            self.best_action = new_best_action
        else:
            raise TypeError(
                'The best action at a node should be of type RAOStarGraphOperator.')

    def set_terminal(self, is_terminal):                                                     #boolean
        self.terminal = is_terminal

    def set_risk(self, new_risk):                                                   #risk is 0 - 1.0
        if new_risk >= 0.0 and new_risk <= 1.0:
            self.risk = new_risk
        else:
            raise ValueError('Invalid risk value: %f' % (new_risk))

    def set_exec_risk(self, new_exec_risk):                                       
        if new_exec_risk >= 0.0 and new_exec_risk <= 1.0:
            self.exec_risk = new_exec_risk
        else:
            raise ValueError('Invalid execution risk: %f' % (new_exec_risk))

    def set_exec_risk_bound(self, new_exec_risk_bound):
        if new_exec_risk_bound >= 0.0 and new_exec_risk_bound <= 1.0:
            self.exec_risk_bound = new_exec_risk_bound
        else:
            raise ValueError('Invalid execution risk bound: %f' %
                             (new_exec_risk_bound))

    def print_node(self):
        # self.value = value         #Q value
        # self.terminal = terminal   #Terminal flag
        # self.state = state         #Belief state
        # self.best_action = best_action #Best action at the node
        # self.risk = 0.0             #Belief state risk
        # self.exec_risk = 0.0        #Execution risk
        # self.exec_risk_bound = 1.0  #Bound on execution risk
        # self.depth = 0              #Initial depth
        best_action_name = None
        if self.best_action:
            best_action_name = self.best_action.name
        print_out = {'state': self.state.belief,
                     'value': self.value,
                     'risk': self.risk,
                     'exec_risk': self.exec_risk,
                     'exec_risk_bound': self.exec_risk_bound,
                     'depth': self.depth,
                     'best_action': best_action_name,
                     'terminal': self.terminal}
        print(print_out)

    def __str__(self):
        """String representation of a Hypergraph node."""
        return self.name


class RAOStarGraphOperator(GraphElement):                                           #link from node to children nodes by an action (link between children nodes themself)
    """
    Class for operators associated to hyperedge in the RAO* hypergraph.
    """

    def __init__(self, name=None, op_value=0.0, properties={}):
        super(RAOStarGraphOperator, self).__init__(name, properties)

        self.op_value = op_value  # Value associated with this operator

        # Dictionary key used to detect that two operators are equal
        # TODO: This fixes the problem with duplicated operators, but doesn't
        # answer the question: how can unduplicated actions give rise to
        # duplicated operators? Answer: because the hypergraph is a graph, and the
        # same node was being expanded through different paths.

    def set_op_value(self, new_value):
        self.op_value = new_value


class RAOStarHyperGraph(GraphElement):                                                           # graph representation
    """
    Class representing an RAO* hypergraph.
    """

    def __init__(self, name=None, properties={}):
        super(RAOStarHyperGraph, self).__init__(name, properties)
        # Dictionary of nodes mapping their string names to themselves
        self.nodes = {}
        # Dictionary of operators mapping their string names to themselves
        self.operators = {}
        # Nested dictionary {parent_key: {operator_key: successors}}
        self.hyperedges = {}
        # Dictionary from child to sets of parents {child_key: set(parents)}
        self.parents = {}

        if (sys.version_info > (3, 0)):
            # Python 3 code in this block
            self.python3 = True
        else:
            # Python 2 code in this block
            self.python3 = False

    def reset_likelihoods(self):
        if self.python3:
            for name, node in self.nodes.items():
                node.likelihood = 0.0
        else:
            for name, node in self.nodes.iteritems():                       #if python 2 is used, it uses nodes.iteritems instead of .items
                node.likelihood = 0.0

    # def mark_all_node_unreachable(self):

    def update_root_and_purge(self, new_root):

        # reset all reachable markings in the graph
        for name, node in self.nodes.items():
            node.reachable = False

        queue = deque([new_root])
        marked = set()
        marked.add(new_root)
        new_root.reachable = True

        while len(queue) > 0:
            node = queue.popleft()

            if not node.terminal:  # node is not terminal

                children = self.all_descendants(node)

                for c in children:
                    if c not in marked:
                        c.reachable = True
                        marked.add(c)
                        queue.append(c)
            # else:  # no best action has been assigned yet
                # should not need to do anything because we are marking
                # children of nodes with best_actions

        for name, node in self.nodes.items():
            if not node.reachable:
                del self.nodes[name]

    def set_nodes(self, new_nodes):                                             #definition of nodes and operators
        self.nodes = new_nodes

    def set_operators(self, new_operators):
        self.operators = new_operators

    def set_hyperedges(self, new_hyperedges):                                                     #definition of hyperedges and parents (must be all listed as dictioneries)
        if isinstance(new_hyperedges, dict):
            self.hyperedge_dict = new_hyperedges
        else:
            raise TypeError('Hyperedges should be given in dictionary form')

    def set_parents(self, new_parents):
        if isinstance(new_parents, dict):
            self.parent_dict = new_parents
        else:
            raise TypeError('Node parents should be given in dictionary form')

    def set_root(self, new_root):                                                #defintion of root, must be of type RAOStarGraphNode as it's the main node, which can be used to reach all other nodes
        if isinstance(new_root, RAOStarGraphNode):
            self.root = new_root
            self.add_node(self.root)
        else:
            raise TypeError(
                'The root of the hypergraph must be of type RAOStarGraphNode.')

    def add_node(self, node):                                                            #adds a single node or operator to the dictionery
        """Adds a node to the hypergraph."""
        if not node in self.nodes:
            self.nodes[node.name] = node

    def add_operator(self, op):
        """Adds an operators to the hypergraph."""
        if not op in self.operators:
            self.operators[op] = op

    def add_hyperedge(self, parent_obj, child_obj_list, prob_list, op_obj):                     #add edges between parent and children, adds nodes if didn't exist as well
        """Adds a hyperedge between a parent and a list of child nodes, adding
        the nodes to the graph if necessary."""
        # Makes sure all nodes and operator are part of the graph.
        self.add_node(parent_obj)
        self.add_operator(op_obj)

        for child in child_obj_list:
            self.add_node(child)

        # Adds the hyperedge
        # TODO: check if the hashing is being done correctly here by __hash__
        if parent_obj in self.hyperedges:

            # #TODO:
            # #Symptom: operators at the hypergraph nodes where being duplicated,
            # #even though actions the hypergraph models (the operator names),
            # #were not.
            # #
            # #Debug conclusions: operators were using their memory ids as hash
            # #keys, causing operators with the same action (name) to be considered
            # #different objects. The duplication would manifest itself when a node
            # #already with outgoing hyperedges (parent_obj in self.hyperedges) was
            # #later dequed and given the same operator. Fortunately, the tests
            # #revealed that different operators with the same action would yield
            # #the same set of children nodes, which indicates that the expansion
            # #is correctly implemented.
            #
            # if op_obj in self.hyperedges[parent_obj]:
            #     #TODO: this should be removed, once I'm confident that the algorithm
            #     #is handling the hypergraph correctly. It checks whether the two
            #     #copies of the same operator yielded the same children at the same
            #     #parent node (which is a requirement), and opens a debugger if they
            #     #don't
            #     prev_children = self.hyperedges[parent_obj][op_obj]
            #     if len(prev_children)!=len(child_obj_list):
            #         print('WARNING: operator %s at node %s yielded children lists with different lengths'%(op_obj.name,parent_obj.name))
            #         import ipdb; ipdb.set_trace()
            #         pass
            #
            #     for child in child_obj_list:
            #         if not child in prev_children:
            #             print('WARNING: operator %s at node %s yielded different sets of children'%(op_obj.name,parent_obj.name))
            #             import ipdb; ipdb.set_trace()
            #             pass

            self.hyperedges[parent_obj][op_obj] = child_obj_list
        else:
            self.hyperedges[parent_obj] = {op_obj: child_obj_list}

        # Records the mapping from children to parent nodes
        for i, child in enumerate(child_obj_list):
            if not (child in self.parents):
                self.parents[child] = set()
            # Added association of action and probability to each parent
            # This is so we can match transition probabilities when calculating
            # likelihoods in the policy
            self.parents[child].add((parent_obj, op_obj, prob_list[i]))

    def remove_all_hyperedges(self, node):                                                                   #list of functions (delete, list, boolean)
        """Removes all hyperedges at a node."""
        if node in self.hyperedges:
            del self.hyperedges[node]

    def all_node_operators(self, node):
        """List of all operators at a node."""
        return list(self.hyperedges[node].keys()) if node in self.hyperedges else []

    def all_descendants(self, node):
        """List of all descendants of a node, from all hyperedges"""
        '''Currently includes repititions!'''
        operators = self.all_node_operators(node)
        descendants = []
        for o in operators:
            descendants.extend(self.hyperedge_successors(node, o))
        return descendants

    def all_node_ancestors(self, node):
        """Set of all node parents, considering all hyperedges."""
        if node in self.parents:
            return self.parents[node]
        return set()

    def policy_successors(self, node):
        if node.best_action:
            return self.hyperedge_successors(node, node.best_action)
        return []

    def hyperedge_successors(self, node, act):
        """List of children associated to a hyperedge."""
        if node in self.hyperedges and act in self.hyperedges[node]:
            return self.hyperedges[node][act]
        return []

    def has_node(self, node):
        """Returns whether the hypergraph contains the node"""
        return (node.name in self.nodes)

    def has_operator(self, op):
        """Returns whether the hypergraph contains the operator."""
        return (op in self.operators)

    def has_ancestor(self, node):
        """Whether a node has ancestors in the graph."""
        return (node in self.parents)
