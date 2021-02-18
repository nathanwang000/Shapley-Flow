'''
evaluation code for the linear sanity check examples

see example in linear_attribution.ipynb
'''
from collections import defaultdict
import numpy as np
from on_manifold import FeatureAttribution
from flow import eval_graph, get_source_nodes

def get_error(gt, theta, allowed_nodes=None):
    '''
    gt is ground truth dictionary
    theta is feature attribution by an explaination method
    only compare nodes in allowed nodes
    '''
    if allowed_nodes is not None:
        allowed_node_names = [node.name for node in allowed_nodes]
        
    diff = []
    for node_name in gt:

        if allowed_nodes is not None and node_name not in allowed_node_names:
            continue
            
        diff = np.hstack([diff, gt[node_name] - theta[node_name]])
    return diff

def get_indirect_effect_flow(cf): # flow indirect effect is sum of outgoing edges
    d = defaultdict(int)
    ec = cf.edge_credit
    for node1, v_dict in ec.items():
        for node2, v in v_dict.items():
            d[node1.name] += v

    return d

def get_effect_asv(cf): # asv effect is the sum of outgoing edges of flow
    d = defaultdict(int)
    ec = cf.get_asv_edge_credit(aggregate=False)

    for node1, v_dict in ec.items():
        for node2, v in v_dict.items():
            if node2.is_target_node:
                d[node1.name] = v

    return d

def get_effect_ind(cf):
    assert type(cf) == FeatureAttribution,\
        f"please get cf from the explaination of IndExplainer, now {type(cf)}"
    d = defaultdict(int)
    d.update({k:v for k, v in zip(cf.input_names, cf.values[:,:,0].T)})
    return d

def get_effect_manifold(cf):
    d = defaultdict(int)
    d.update({k:v for k, v in zip(cf.input_names, cf.values[:,:,0].T)})
    return d

def get_direct_effect_flow(cf): # only count the edge directly connected to the target node
    d = defaultdict(int)
    ec = cf.edge_credit
    for node1, v_dict in ec.items():
        for node2, v in v_dict.items():
            if node2.is_target_node:
                d[node1.name] = v

    return d

def get_direct_effect_ground_truth(graph):
    gt = {}
    for n in graph:
        if n.is_target_node: continue

        intervention_on = n.name
        
        d = {}
        for node in graph:
            if not node.is_target_node:
                try:
                    d[node.name] = node.baseline
                except:
                    # this is a grouped noise node
                    d[node.name] = {k: v for k, v in node.baseline.items()}
        
        before = eval_graph(graph, d)
        
        try:
            d.update({intervention_on: n.target}) # foreground
        except:
            d.update({intervention_on: {k: n.target[k] for k in n.target}}) # grouped node
            
        after = eval_graph(graph, d)

        gt[intervention_on] = after - before
    return gt

def get_indirect_effect_ground_truth(graph):
    gt = {}
    for n in graph:
        if n.is_target_node: continue

        intervention_on = n.name
        sources = get_source_nodes(graph)

        d = {}
        for node in sources:
            if not node.is_target_node:
                try:
                    d[node.name] = node.baseline
                except:
                    # this is a grouped noise node
                    d[node.name] = {k: v for k, v in node.baseline.items()}
        
        before = eval_graph(graph, d)

        try:
            d.update({intervention_on: n.target}) # foreground
        except:
            d.update({intervention_on: {k: n.target[k] for k in n.target}}) # grouped noise node

        after = eval_graph(graph, d)

        gt[intervention_on] = after - before
    return gt
