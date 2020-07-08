import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import warnings

class GraphIterator:
    def __init__(self, graph):
        self._graph = graph
        self._index = 0 # keep track of current index

    def __next__(self):
        '''returns next node'''
        if self._index < len(self._graph):
            result = self._graph.nodes[self._index]
            self._index += 1
            return result

        # end of iteration
        raise StopIteration

class Graph:
    '''list of nodes'''
    def __init__(self, nodes, baseline_sampler, target_values,
                 treat_target_differently=True, verbose=False):
        '''
        nodes: sequence of Node object
        baseline_sampler: a function ()->{name: val} where name
                          point to a node in nodes
        target_values: a dictionary {name: val} gives the current
                       value of the explanation instance; name
                       point to a node in nodes
        treat_target_differently: the baseline for target node is 
                                  set using its function on other
                                  node's baseline
        '''
        self.nodes = list(set(nodes))
        self.baseline_sampler = baseline_sampler
        self.target_values = target_values
        self.treat_target_differently = treat_target_differently
        self.verbose = verbose
        self.reset()

    def __len__(self):
        return len(self.nodes)

    def __iter__(self):
        return GraphIterator(self)
    
    def reset(self):
        baseline_values = self.baseline_sampler()
        target_nodes = [] 
        for node in self.nodes:
            if node.is_target_node and self.treat_target_differently:
                target_nodes.append(node)
                continue
            node.set_baseline_target(baseline_values[node.name],
                                     self.target_values[node.name])

            for c in node.children: # only activated edges change visibility
                c.visible_arg_values[node] = node.baseline

        # set the target node baseline using its args
        if self.treat_target_differently:
            assert len(target_nodes) == 1, \
                f"{len(target_nodes)} target node, need 1"
            node = target_nodes[0]
            node.set_baseline_target(node.f(*[arg.baseline for arg in node.args]),
                                     self.target_values[node.name])

        self.check_baseline_target()

    def check_baseline_target(self):
        '''
        check if baseline, target matches the functional form
        '''
        n_targets = 0
        for node in topo_sort(self):
            if node.is_target_node: n_targets += 1
            if len(node.args) > 0:
                computed_baseline = node.f(*[arg.baseline for arg in node.args])
                computed_target = node.f(*[arg.target for arg in node.args])
                
                residue = node.baseline - computed_baseline
                if residue != 0:

                    if self.verbose:
                        print(f"baseline for {node} has residue {residue}")
                        print(f"enforcing baseline for {node}")
                    node.set_baseline_target(computed_baseline, computed_target)

                residue = node.target - computed_target
                if residue != 0:

                    if self.verbose:
                        print(f"warning: outcome for {node} has residue {residue}")
                        print(f"enforcing target for {node}")
                    node.set_baseline_target(computed_baseline, computed_target)

        assert n_targets == 1, f"{n_targets} target node, need 1"
            
class Node:

    def __init__(self, name, f=None, args=[], children=[],
                 is_target_node=False, is_noise_node=False):
        '''
        name: name of the node
        f: functional form of this variable on other variables
        args: arguments node, predessors of the node
        children: children of the node
        target: current value
        baseline: baseline value
        is_target_node: is this node to explain
        is_noise_node: is this node a noise node
        '''
        self.name = name
        self.f = f
        self.is_target_node = is_target_node
        self.is_noise_node = is_noise_node

        # arg values that are visible to the node
        self.visible_arg_values = {}
        
        self.args = []
        for arg in args:
            self.add_arg(arg)
            
        self.children = []
        for c in children:
            self.add_child(c)
        
    def set_baseline_target(self, baseline, target):
        self.target = target
        self.baseline = baseline

        # reset value
        self.last_val = self.baseline
        self.val = self.baseline
        self.from_node = None
        
    def add_arg(self, node):
        '''add predecessor'''
        if node not in self.args:
            self.args.append(node)
        if self not in node.children:
            node.children.append(self)
        
    def add_child(self, node):
        '''add children'''
        if node not in self.children:
            self.children.append(node)
        if self not in node.args:
            node.args.append(self)

    def __repr__(self):
        return self.name

class CreditFlow:

    def __init__(self, verbose=True, nruns=10):
        ''' 
        verbose: whether to print out decision process        
        nruns: number of sampled valid timelines and permutations
        '''
        self.edge_credit = defaultdict(lambda: defaultdict(int))
        self.verbose = verbose
        self.nruns = nruns

    def credit(self, node, val):
        if node is None or node.from_node is None:
            return
            
        # self.edge_credit[node.from_node.name][node.name] += val
        self.edge_credit[node.from_node][node] += val        
        if self.verbose:
            print(f"assign {val} credits to {node.from_node}->{node}")
        self.credit(node.from_node, val) # propagate upward

    def dfs(self, node, order):
        if node.is_target_node:
            self.credit(node, node.val - node.last_val)
            return

        children_order = np.random.permutation(node.children)
        for c in children_order:

            if self.verbose:
                print(f'turn on edge {node}->{c}')
            c.from_node = node
            c.last_val = c.val
            c.visible_arg_values[node] = node.val
            c.val = c.f(*[c.visible_arg_values[arg] for arg in c.args])
            if self.verbose:
                print(f'{c} changes from {c.last_val} to {c.val}')
            self.dfs(c, order)

    def run(self, graph):
        '''
        run shap flow algorithm to fairly allocate credit
        '''

        sources = get_source_nodes(graph)
        for i in range(self.nruns): # random sample valid timelines
            # make value back to baseline
            graph.reset()

            order = list(np.random.permutation(sources))
            if self.verbose:
                print(f"\n----> using order {order}")

            if self.verbose:
                print("baselines " +\
                      ", ".join(map(lambda node: f"{node}: {node.last_val}",
                                    order)))
            # follow the order
            for node in order:
                if self.verbose:
                    print(f"turn on node {node} from {node.val} to {node.target}")
                node.last_val = node.val # update last val
                node.val = node.target # turn on the node
                node.from_node = None # turn off the source
                # note: additional contribution is from random source
                self.dfs(node, order)

    def print_credit(self):
        for node1, d in self.edge_credit.items():
            for node2, val in d.items():
                print(f'credit {node1}->{node2}: {val/self.nruns}')

    def credit2dot(self):
        '''
        convert the graph to pydot graph for visualization:
        e.g. 
        from IPython.display import Image
        G = self.credit2dot()
        G.write_png("graph.png")
        Image(G.png)
        '''
        G = nx.DiGraph()
        for node1, d in self.edge_credit.items():
            for node2, val in d.items():
                w = val/self.nruns
                edge_label = "{:.2f}".format(w)

                color = "orange" if w == 0 else "black"
                width = 1 if w == 0 else abs(w)

                if node1.is_noise_node:
                    G.add_node(node1, shape="point")
                    G.add_edge(node1, node2, weight=w, penwidth=width,
                               color=color,
                               label=edge_label)
                    continue

                if node1 not in G:
                    G.add_node(node1, label=\
                               f"{node1} {node1.target}/{node1.baseline}") 

                if node2 not in G:
                    G.add_node(node2, label=\
                               f"{node2} {node2.target}/{node2.baseline}")

                G.add_edge(node1, node2, weight=w, penwidth=width,
                           color=color,
                           label=edge_label)

        return nx.nx_pydot.to_pydot(G)

def get_source_nodes(graph):
    indegrees = dict((node, len(node.args)) for node in graph)
    sources = [node for node in graph if indegrees[node] == 0]
    return sources

def topo_sort(graph):
    '''
    topological sort of a graph
    '''
    order = []
    indegrees = dict((node, len(node.args)) for node in graph)
    sources = [node for node in graph if indegrees[node] == 0]
    while sources:
        sources = list(np.random.permutation(sources))
        s = sources.pop()
        order.append(s)
        for u in np.random.permutation(s.children):
            indegrees[u] -= 1 # s is satisfied
            if indegrees[u] == 0:
                sources.append(u)
    return order

# sample graph
def build_graph():
    '''
    build and return a graph (list of nodes), to be runnable in main
    '''
    # build the graph: x1->x2, y = x1 + x2
    x1 = Node('x1')
    x2 = Node('x2', lambda x1: x1, [x1])
    y  = Node('target', lambda x1, x2: x1 + x2, [x1, x2], is_target_node=True)
    
    # initialize the values from data: now is just specified
    graph = Graph([x1, x2, y],
                  # sample baseline
                  lambda: {'x1': 0, 'x2': 0, 'target': 0},
                  # target to explain
                  {'x1': 1, 'x2': 1.5, 'target': 2.5})
    
    return graph

# sample runs
def main():

    cf = CreditFlow(verbose=False, nruns=1)
    graph = build_graph()
    cf.run(graph)
    cf.print_credit()

    return cf
    
if __name__ == '__main__':
    main()