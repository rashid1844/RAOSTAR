#!/usr/bin/env python

# convert graph obtained from RAO* to json file to be used with
# Matt Deyo's policy visualizer http://mdeyo.com/policy-vis/

### use policy_to_json ###

# Input G is of the RAOStarHyperGraph class found in raostarhypergraph.py

import json
from collections import deque

default_settings = {
    "nodes": {
        "display-properties": ["node_id", "value"],
        "color": "#99ccff",
        "terminal-color": "#7BE141"
    },
    "edges": {
        "display-properties": ["action", "probability"],
        "color": "#99ccff"
    },
    "hierarchical": "true",
    "nodeSpacing": 550,
    "levelSpacing": 200,
    "color": "blue"
}


def node_info(node_object, cc):
    nd = node_object
    if nd.state.state_print:
        state = nd.state.state_print()
    else:
        state = nd.state.belief
    nd_info = {
        "state": state,
        "acceptable-risk-ub": cc,
        "execution-risk-bound": [0, 0, nd.exec_risk_bound],
        "state-risk": nd.risk,
        "value": str(nd.value),
        "is-terminal": str(nd.terminal)}
    return nd_info


def policy_to_json(G, cc, filename, settings=default_settings):
    graph_info = {"nodes": {}, "edges": {}, "settings": settings}
    graph_info["nodes"]["node-0"] = node_info(G.root, cc)
    added_nodes = {G.root.name: 'node-0'}  # node.name: node-i
    queue = deque([G.root])
    n_ind = 1  # nodes str index
    e_ind = 0  # edges str index
    while len(queue) > 0:
        node = queue.popleft()
        if node.best_action != None:
            children = G.hyperedges[node][node.best_action]
            edge_str = "edge-%d" % (e_ind)
            # store edge information
            e_info = {
                "action": str(node.best_action.name),
                "predecessor": added_nodes[node.name],
                "successors": {}}
            for c in children:
                queue.append(c)
                if c.name not in added_nodes:
                    nodestr = "node-%d" % (n_ind)
                    added_nodes[c.name] = nodestr
                else:
                    nodestr = added_nodes[c.name]
                graph_info["nodes"][nodestr] = node_info(c, cc)
                n_ind += 1
                e_info["successors"][nodestr] = {"probability": c.probability}
            graph_info["edges"][edge_str] = e_info
            e_ind += 1
    with open(filename, 'w') as fjson:
        json.dump(graph_info, fjson)
    return graph_info


def graph_to_json(G, cc, filename, settings=default_settings):
    # first place everything in generic dictionary
    graph_info = {"nodes": {}, "edges": {}, "settings": settings}
    nodes = G.nodes
    node_strings = nodes.keys()
    edges = G.hyperedges
    parents = G.hyperedges.keys()
    root = G.root
    added_nodes = {}  # given node_name: node-i
    added_nodes[root.name] = "node-0"
    graph_info["nodes"]["node-0"] = node_info(root, cc)
    # print([(p.name, len(edges[p])) for p in parents])
    # print([c])
    n_ind = 1  # nodes str index
    e_ind = 0  # edges str index
    # add edges and nodes
    for i in range(len(parents)):
        for op in edges[parents[i]]:
            edge_str = "edge-%d" % (e_ind)
            # add parents to node list
            if parents[i].name not in added_nodes:  # add node
                nodestr = "node-%d" % (n_ind)
                added_nodes[parents[i].name] = nodestr
                graph_info["nodes"][nodestr] = node_info(parents[i], cc)
                n_ind += 1
            # store edgfe information
            e_info = {
                "action": str(op.name),
                "predecessor": added_nodes[parents[i].name],
                "successors": {}}
            for c in edges[parents[i]][op]:  # children
                if c.name not in added_nodes:  # add if necessary
                    nodestr = "node-%d" % (n_ind)
                    added_nodes[c.name] = nodestr
                    graph_info["nodes"][nodestr] = node_info(c, cc)
                    n_ind += 1
                e_info["successors"][added_nodes[c.name]] = {
                    "probability": c.probability}
            graph_info["edges"][edge_str] = e_info
            e_ind += 1

    with open(filename, 'w') as fjson:
        json.dump(graph_info, fjson)
    # print(added_nodes)
    return graph_info
