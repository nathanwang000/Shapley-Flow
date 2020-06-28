import numpy as np
from collections import defaultdict

class Node:

    def __init__(self, baseline, target, name, args=[], children=[]):
        '''
        target: current value
        baseline: baseline value
        name: name of the node
        args: arguments node, predessors of the node
        children: children of the node
        '''
        self.target = target
        self.name = name        
        self.baseline = baseline
        self.reset()
        
        self.f = lambda: self.val # functional form of the node
        
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
            
    def add_arg(self, node):
        '''add predecessor'''
        if node not in self.args:
            self.args.append(node)
        if self not in node.children:
            node.children.append(self)
        
    def add_child(self, node):
        if node not in self.children:
            self.children.append(node)
        if self not in node.args:
            node.args.append(self)

    def __repr__(self):
        return self.name

class CreditFlow:

    def __init__(self, verbose=True):
        self.edge_credit = defaultdict(lambda: defaultdict(int))
        self.verbose = verbose

    def credit(self, node, val):
        if node is None or node.from_node is None:
            return

        self.edge_credit[node.from_node.name][node.name] += val
        if self.verbose:
            print(f"assign {val} credits to {node.from_node}->{node}")
        self.credit(node.from_node, val) # propagate upward

    def dfs(self, node, order):
        if node.name == 'target':
            self.credit(node, node.val - node.last_val)
            return

        for c in sorted(node.children,
                        key=lambda x: (order[-1:] + order[1:]).index(x)):
            # turn on edge follwing the order of [y] + order[:-1]
            # randomly turn on edges
            if self.verbose:
                print(f'turn on edge {node}->{c}')
            c.from_node = node
            c.last_val = c.val
            c.val = c.f(*[arg.val for arg in c.args])
            if self.verbose:
                print(f'{c} changes from {c.last_val} to {c.val}')
            self.dfs(c, order)

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

def main(verbose=False, nruns=100):
    '''
    verbose: whether to print out decision process
    nruns: number of sampled valid timelines and permutations
    '''
    cf = CreditFlow(verbose=verbose)
    
    # build the graph: x1->x2, y = f(x1, x2)
    x1 = Node(0, 1, 'x1')
    x2 = Node(0, 1.5, 'x2', [x1])
    y = Node(0, 2.5, 'target', [x1, x2])

    # define functional form: here is specified, later learn
    y.f = lambda x1, x2: x1 + x2
    x2.f = lambda x1: x1 # randomness assumes due to residue

    # run topological sort
    graph = [x1, x2, y]
    for i in range(nruns): # random sample number of valid timelines
        # make everything back to baseline
        for node in graph: node.reset()
        
        order = topo_sort(graph)
        if len(order) != len(graph):
            print("order cannot be satisfied")
            return
        else:
            if verbose:
                print(f"using order {order}")

        if verbose:
            print("baselines " +\
                  ", ".join(map(lambda node: f"{node}: {node.last_val}",
                                order)))
        # follow the order
        for node in order:
            if verbose:
                print(f"turn on node {node} form {node.val} to {node.target}")
            node.val = node.target # turn on the node
            node.from_node = None # turn off the source
            # note: additional contribution is from random source
            cf.dfs(node, order)

    for node1, d in cf.edge_credit.items():
        for node2, val in d.items():
            print(f'credit {node1}->{node2}: {val/nruns}')

def memoize(f):
    d = {}
    def f_(n):
        if n in d: return d[n]
        d[n] = f(n)
        return d[n]
        
    return f_

@memoize
def T(n):
    if n == 1: return 1
    return sum([T(i) for i in range(1, n)]) + 1

if __name__ == '__main__':
    main()

        
        
