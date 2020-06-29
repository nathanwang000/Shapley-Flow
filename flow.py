import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx

class Node:

    def __init__(self, name, f=None, args=[], children=[]):
        '''
        name: name of the node
        f: functional form of this variable on other variables
        args: arguments node, predessors of the node
        children: children of the node
        target: current value
        baseline: baseline value
        '''
        self.name = name
        self.f = f
        
        self.args = []
        for arg in args:
            self.add_arg(arg)
            
        self.children = []
        for c in children:
            self.add_child(c)

    def reset(self):
        self.last_val = self.baseline
        self.val = self.baseline
        self.from_node = None
            
    def set_baseline_target(self, baseline, target):
        self.target = target
        self.baseline = baseline
        self.reset()
        
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

    def __init__(self, verbose=True, nruns=10, permute_edges=False):
        ''' 
        verbose: whether to print out decision process        
        nruns: number of sampled valid timelines and permutations
        permute_edges: whether or not consider different ordering of edges, 
                       default False
        '''
        self.edge_credit = defaultdict(lambda: defaultdict(int))
        self.verbose = verbose
        self.nruns = nruns
        self.permute_edges = permute_edges

    def credit(self, node, val):
        if node is None:
            return
        if node.from_node is None:
            # the effect of noise term
            self.edge_credit[node.name][node.name] += val
            if self.verbose:
                print(f"assign {val} credits to {node.name}->{node}")
            return
            
        self.edge_credit[node.from_node.name][node.name] += val
        if self.verbose:
            print(f"assign {val} credits to {node.from_node}->{node}")
        self.credit(node.from_node, val) # propagate upward

    def dfs(self, node, order):
        if node.name == 'target':
            self.credit(node, node.val - node.last_val)
            return

        if self.permute_edges:
            children_order = np.random.permutation(node.children)
        else:
            # turn on edge follwing the order of [y] + order[:-1]
            children_order = sorted(node.children,
                                    key=lambda x: \
                                    (order[-1:] + order[1:]).index(x))
            
        for c in children_order:

            if self.verbose:
                print(f'turn on edge {node}->{c}')
            c.from_node = node
            c.last_val = c.val
            c.val = c.f(*[arg.val for arg in c.args])
            if self.verbose:
                print(f'{c} changes from {c.last_val} to {c.val}')
            self.dfs(c, order)

    def run(self, graph):
        '''
        run shap flow algorithm to fairly allocate credit
        '''
        # run topological sort
        for i in range(self.nruns): # random sample number of valid timelines
            # make value back to baseline
            for node in graph: node.reset()

            order = topo_sort(graph)
            if len(order) != len(graph):
                print("order cannot be satisfied")
                return
            else:
                if self.verbose:
                    print(f"using order {order}")

            if self.verbose:
                print("baselines " +\
                      ", ".join(map(lambda node: f"{node}: {node.last_val}",
                                    order)))
            # follow the order
            for node in order:
                if self.verbose:
                    print(f"turn on node {node} form {node.val} to {node.target}")
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
            if node1 not in G:
                G.add_node(node1)
            for node2, val in d.items():
                if node2 not in G:
                    G.add_node(node2)
                w = val/self.nruns
                if w != 0:
                    G.add_edge(node1, node2, weight=w, penwidth=abs(w),
                               label=str(w))

        return nx.nx_pydot.to_pydot(G)
    
def topo_sort(graph):
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

def check_baseline_target(graph):
    ''' graph is a list of nodes '''
    for node in graph:
        if len(node.args) > 0:
            residue = node.baseline - node.f(*[arg.baseline for arg in node.args])
            if residue != 0:
                print(f"baseline additive noise for {node} is {residue}")    
            
            residue = node.target - node.f(*[arg.target for arg in node.args])
            if residue != 0:
                print(f"outcome additive noise for {node} is {residue}")

# sample graphs
def build_graph():
    '''
    build and return a graph (list of nodes), to be runnable in main
    '''
    # build the graph: x1->x2, y = x1 + x2
    x1 = Node('x1')
    x2 = Node('x2', lambda x1: x1, [x1])
    y  = Node('target', lambda x1, x2: x1 + x2, [x1, x2])    
    graph = [x1, x2, y]
    
    # initialize the values from data
    x1.set_baseline_target(0, 1)
    x2.set_baseline_target(0, 1.5)
    y.set_baseline_target(0, 2.5)

    # check the amount of noise
    check_baseline_target(graph)
    return graph
    
def build_graph2():
    '''
    build and return a graph (list of nodes), to be runnable in main
    '''
    # build the graph: x1->x2->x3, y = f(x1, x2, x3)
    x1 = Node('x1')
    x2 = Node('x2', lambda x1: x1, [x1])
    x3 = Node('x3', lambda x2: x2, [x2])
    y  = Node('target', lambda x1, x2, x3: x1 + x2 + x3, [x1, x2, x3])    
    graph = [x1, x2, x3, y]
    
    # initialize the values from data
    x1.set_baseline_target(0, 1)
    x2.set_baseline_target(0, 1.5)
    x3.set_baseline_target(0, 1)
    y.set_baseline_target(0, 3.5)

    # check the amount of noise
    check_baseline_target(graph)
    return graph

# sample runs
def main():

    cf = CreditFlow(verbose=False, nruns=1)
    graph = build_graph2()
    cf.run(graph)
    cf.print_credit()

    return cf
    
if __name__ == '__main__':
    main()

        
        
