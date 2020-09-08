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
Enumeration Tree for systematic search

@author: Sungkweon Hong (sk5050@mit.edu).

"""


import sys
import string
import random
from collections import deque


class EnumTreeElement(object):
    """
    Generic graph element with a name and a unique ID.
    """

    def __init__(self, name=None, properties={}):
        self.set_name(name)
        # self.name = name
        self.properties = properties

    __hash__ = object.__hash__

    def set_name(self, new_name):
        if new_name == None:
            self.name = self.id_generator()
        else:
            self.name = new_name

    def set_properties(self, new_properties):
        if isinstance(new_properties, dict):
            self.properties = new_properties
        else:
            raise TypeError(
                'enumtree element properties should be given as a dictionary.')

    def __eq__(x, y):
        return isinstance(x, EnumTreeElement) and isinstance(y, EnumTreeElement) and (x.name == y.name)

    def __ne__(self, other):
        return not self == other

    def id_generator(self, size=6, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))


class EnumTreeNode(EnumTreeElement):
    """
    Class for nodes in the enumeration tree.
    """

    def __init__(self, parent_etree_node, best_action=None, terminal=False, name=None, root_value=None,
                 properties={}, make_unique=False):
        super(EnumTreeNode, self).__init__(name, properties)

        # Parent node of enumeration tree
        self.parent_etree_node = parent_etree_node
        # Dictionary of differences of values from parent enumeration tree node to current enumeration tree node.
        # Key is RAOStarGraphNode's string name, value is another dictionary, which has diff values for value, execution risk and previous best action.
        self.diff = dict()

        # current root value. This is mainly used for pruning.
        self.root_value = root_value

    def init_diff(self, node):
        self.diff[node] = {'value_diff':0, 'exec_risk_diff':0, 'er_bound_diff':0, 'prev_best_action':False, 'current_best_action':False, 'deadend_checked':False}

    def increment_diff(self, node, key, prev_value, new_value):
        if node not in self.diff :
            self.init_diff(node)
            
        if key=='prev_best_action':
            self.diff[node][key] = prev_value
        elif key=='current_best_action':
            self.diff[node][key] = new_value
        else:
            self.diff[node][key] = self.diff[node][key] + (new_value - prev_value)

    # def compute_diff(self, prev_node, new_node):
    #     value_diff = new_node.value - prev_node.value
    #     exec_risk_diff = new_node.exec_risk - prev_node.exec_risk
    #     prev_best_action = prev_node.best_action

    #     self.diff[new_node.name] = {'value_diff':value_diff, 'exec_risk_diff':exec_risk_diff, 'prev_best_action':prev_best_action}


class EnumTree(EnumTreeElement):
    """
    Class representing an enumeration tree.
    """

    def __init__(self, name=None, properties={}):
        super(EnumTree, self).__init__(name, properties)
        # Dictionary of nodes mapping their string names to themselves
        self.nodes = {}
        
    def add_node(self, node):
        """Adds a node to the hypergraph."""
        if not node in self.nodes:
            self.nodes[node.name] = node

    def checkout(self, from_node, to_node):
        # find two different nodes' common ancestor
        first_common_ancestor = self.find_first_common_ancestor(from_node, to_node)

        # from "from_node" to ancestor, undoing all differences
        self.undo_diff(from_node, first_common_ancestor)

        # from ancestor to "to_node", redoing all differences
        self.redo_diff(to_node, first_common_ancestor)

    def find_first_common_ancestor(self, from_node, to_node):
        # find first common ancestor for from and to nodes.
        
        # get all ancestors for both from and to nodes.
        from_node_ancestors = self.all_ancestors(from_node)
        to_node_ancestors = self.all_ancestors(to_node)
        
        fist_common_ancestor = None

        while len(from_node_ancestors)>0 and len(to_node_ancestors)>0 :
            from_node_ancestor = from_node_ancestors.pop()
            to_node_ancestor = to_node_ancestors.pop()

            if from_node_ancestor == to_node_ancestor:
                first_common_ancestor = from_node_ancestor
            else:
                if first_common_ancestor == None:
                    raise ValueError('there is no common ancestor!')
                else:
                    return first_common_ancestor

        if first_common_ancestor == None:
            raise ValueError('there is no common ancestor!')
        else:
            return first_common_ancestor
        

    def all_ancestors(self, node):
        # find all ancestors of node and return it. the order is from node to root.
        ancestors = []
        while True:
            ancestors.append(node)
            parent = node.parent_etree_node

            if parent == None:
                break
            else:
                node = parent

        return ancestors

    def undo_diff(self, from_node, common_ancestor):
        # initialize undoing node as from_node
        undoing_node = from_node

        # undo differences until reaching common ancestor

        while undoing_node != common_ancestor:
            # undo differences for all nodes that has changed in this etree node
            for diff_node in undoing_node.diff:
                diff_node.value -= undoing_node.diff[diff_node]['value_diff']
                diff_node.exec_risk -= undoing_node.diff[diff_node]['exec_risk_diff']
                diff_node.exec_risk_bound -= undoing_node.diff[diff_node]['er_bound_diff']

                if undoing_node.diff[diff_node]['prev_best_action'] != False:
                    diff_node.best_action = undoing_node.diff[diff_node]['prev_best_action']

                if undoing_node.diff[diff_node]['deadend_checked'] == True:
                    diff_node.terminal = False
                    
            undoing_node = undoing_node.parent_etree_node
                
    def redo_diff(self, to_node, common_ancestor):
        redoing_nodes = []
        redoing_node = to_node
        while True:
            if redoing_node == common_ancestor:
                break
            else:
                redoing_nodes.append(redoing_node)
                redoing_node = redoing_node.parent_etree_node

        # redo differences until reaching to_node
        while len(redoing_nodes)>0:
            redoing_node = redoing_nodes.pop()

            # redo differences for all nodes that has changed in this etree node
            for diff_node in redoing_node.diff:
                diff_node.value += redoing_node.diff[diff_node]['value_diff']
                diff_node.exec_risk += redoing_node.diff[diff_node]['exec_risk_diff']
                diff_node.exec_risk_bound += redoing_node.diff[diff_node]['er_bound_diff']
                # diff_node.best_action = redoing_node.diff[diff_node]['current_best_action']

                if redoing_node.diff[diff_node]['current_best_action'] != False:
                    diff_node.best_action = redoing_node.diff[diff_node]['current_best_action']

                if redoing_node.diff[diff_node]['deadend_checked'] == True:
                    diff_node.terminal = True











        

