import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import copy
import warnings
from pygraphviz import AGraph
from graphviz import Digraph, Source
from collections.abc import Iterable
import tqdm

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
    def __init__(self, nodes, baseline_sampler, target_sampler,
                 display_translator={}):
        '''
        nodes: sequence of Node object
        baseline_sampler: {name: (lambda: val)} where name
                          point to a node in nodes
        target_sampler: {name: (lambda: val)} where name 
                        point to a node in nodes
                        gives the current value
                        of the explanation instance; it can be
                        stochastic when noise of target is not observed
        display_translator: {name: lamda val: translated val}
                            translate value to human readable
        '''
        self.nodes = list(set(nodes))
        self.baseline_sampler = baseline_sampler
        self.target_sampler = target_sampler
        self.display_translator = defaultdict(lambda: (lambda x: x) )
        for k, v in display_translator.items():
            self.display_translator[k] = v

    def add_node(self, node):
        '''
        add a node to nodes
        '''
        self.nodes.append(node)
        
    def __len__(self):
        return len(self.nodes)

    def __iter__(self):
        return GraphIterator(self)

    def sample(self, sampler, name):
        s = sampler[name]()
        if not isinstance(s, Iterable):
            s = [s]
        return np.array(s)
    
    def sample_all(self, sampler):
        '''
        sampler could be baseline sampler or target sampler
        return {name: val} where val is a numpy array
        '''
        d = {}
        for name in sampler:
            d[name] = self.sample(sampler, name)
        return d
            
    def reset(self):
        
        baseline_values = self.sample_all(self.baseline_sampler)
        target_values = self.sample_all(self.target_sampler)

        n_targets = 0
        for node in topo_sort(self):
            if len(node.args) == 0: # source node
                node.set_baseline_target(baseline_values[node.name],
                                         target_values[node.name])
            else:
                computed_baseline = node.f(*[arg.baseline for arg in node.args])
                computed_target = node.f(*[arg.target for arg in node.args])
                node.set_baseline_target(computed_baseline,
                                         computed_target)

            for c in node.children: # only activated edges change visibility
                c.visible_arg_values[node] = node.baseline

            n_targets += node.is_target_node

        assert n_targets == 1, f"{n_targets} target node, need 1"

class Node:
    '''models feature node as a computing function'''
    def __init__(self, name, f=None, args=[],
                 is_target_node=False, is_noise_node=False,
                 is_dummy_node=False):
        '''
        name: name of the node
        f: functional form of this variable on other variables
        args: arguments node, predessors of the node
        children: children of the node
        target: current value
        baseline: baseline value
        is_target_node: is this node to explain
        is_noise_node: is this node a noise node
        is_dummy_node: is this a dummy node 
                       a dummy node is a node that has one parent and
                       one children that shouldn't be drawn
                       but are in place to support multigraph and
                       graph folding
        '''
        self.name = name
        self.f = f
        self.is_target_node = is_target_node
        self.is_noise_node = is_noise_node
        self.is_dummy_node = is_dummy_node

        # arg values that are visible to the node
        self.visible_arg_values = {}
        
        self.args = []
        for arg in args:
            self.add_arg(arg)
            
        self.children = [] # inferred from args
        
    def set_baseline_target(self, baseline, target):
        self.target = target
        self.baseline = baseline

        # reset value
        self.last_val = self.baseline
        self.val = self.baseline
        self.from_node = None
        
    def add_arg(self, node):
        '''add predecessor, allow multi edge'''
        self.args.append(node)
        node.children.append(self)            
        
    def __repr__(self):
        return self.name

class CreditFlow:
    ''' the main algorithm to get a flow of information '''
    def __init__(self, graph, verbose=False, nruns=10,
                 visualize=False, fold_noise=True):
        ''' 
        graph: causal graph to explain
        verbose: whether to print out decision process        
        nruns: number of sampled valid timelines and permutations
        visualize: whether to visualize the graph build process, 
                   need to be verbose
        fold_noise: whether to show noise node as a point
        '''
        self.graph = graph
        self.edge_credit = defaultdict(lambda: defaultdict(int))
        self.verbose = verbose
        self.nruns = nruns
        self.visualize = visualize
        self.dot = AGraph(directed=True)
        self.penwidth_stress = 5
        self.penwidth_normal = 1
        self.fold_noise = fold_noise

    def credit(self, node, val):
        if node is None or node.from_node is None:
            return

        self.edge_credit[node.from_node][node] += val
        if self.verbose:
            print(f"assign {val} credits to {node.from_node}->{node}")
            if self.visualize:
                if not self.dot.has_edge(node.from_node, node):
                    self.dot.add_edge(node.from_node, node)
                dot_edge = self.dot.get_edge(node.from_node, node)
                dot_edge.attr['color'] = "blue"
                label = dot_edge.attr['label'] or '0'
                dot_edge.attr['label'] = f"{label}+{val}"
                dot_edge.attr['fontcolor'] = "blue"
                dot_edge.attr['penwidth'] = self.penwidth_stress
                viz_graph(self.dot)
                dot_edge.attr['penwidth'] = self.penwidth_normal                
                dot_edge.attr['color'] = "black"
                dot_edge.attr['fontcolor'] = "black"                
                dot_edge.attr['label'] = eval(dot_edge.attr['label'])
                
        self.credit(node.from_node, val) # propagate upward

    def dfs(self, node, order):
        if node.is_target_node:
            self.credit(node, node.val - node.last_val)
            return

        children_order = np.random.permutation(node.children)
        for c in children_order:

            c.from_node = node
            c.last_val = c.val
            c.visible_arg_values[node] = node.val
            c.val = c.f(*[c.visible_arg_values[arg] for arg in c.args])

            if self.verbose:
                print(f'turn on edge {node}->{c}')                
                print(f'{c} changes from {c.last_val} to {c.val}')
                if self.visualize:
                    if not self.dot.has_edge(node, c):
                        self.dot.add_edge(node, c)
                    dot_edge = self.dot.get_edge(node, c)
                    dot_edge.attr['color'] = "orange"
                    dot_c = self.dot.get_node(c)
                    label = dot_c.attr['label']
                    dot_c.attr['label'] = f"{label.split(':')[0]}: {c.val:.1f} ({c.baseline:.1f}->{c.target:.1f})"
                    dot_c_color = dot_c.attr['color']
                    dot_edge.attr['penwidth'] = self.penwidth_stress
                    dot_c.attr['penwidth'] = self.penwidth_stress
                    dot_c.attr['color'] = 'orange'
                    viz_graph(self.dot)
                    dot_edge.attr['penwidth'] = self.penwidth_normal
                    dot_c.attr['penwidth'] = self.penwidth_normal
                    dot_edge.attr['color'] = "black"
                    if c.val == c.baseline:
                        dot_c.attr['color'] = dot_c_color or "black"
                    elif c.val == c.target:
                        dot_c.attr['color'] = "green"
                
            self.dfs(c, order)

    def viz_graph_init(self, graph):
        '''
        initialize self.dot with the graph structure
        '''
        dot = AGraph(directed=True)
        for node in topo_sort(graph):
            if node not in dot:
                dot.add_node(node,
                             label=f"{node.name}: {node.val:.1f} ({node.baseline:.1f}->{node.target:.1f})")
            for p in node.args:
                dot.add_edge(p, node)

        self.dot = dot
        viz_graph(self.dot)

    def reset(self):
        '''reset the graph and initialize the visualization'''
        self.graph.reset()
        if self.visualize:
            self.viz_graph_init(self.graph)
        
    def run(self):
        '''
        run shap flow algorithm to fairly allocate credit
        '''
        sources = get_source_nodes(self.graph)
        # random sample valid timelines
        for i in tqdm.trange(self.nruns, desc='sampling'):
            # make value back to baselines
            self.reset()
            
            order = list(np.random.permutation(sources))
            if self.verbose:
                print(f"\n----> using order {order}")
                print("baselines " +\
                      ", ".join(map(lambda node: f"{node}: {node.last_val}",
                                    order)))
            # follow the order
            for node in order:
                node.last_val = node.val # update last val
                node.val = node.target # turn on the node
                node.from_node = None # turn off the source

                if self.verbose:
                    print(f"turn on edge from external source to {node}")
                    print(f"{node} changes from {node.last_val} to {node.val}")
                    if self.visualize:
                        if node not in self.dot:
                            self.dot.add_node(node)
                        dot_node = self.dot.get_node(node)
                        label = dot_node.attr['label']
                        dot_node.attr['label'] = f"{label.split(':')[0]}: {node.val:.1f} ({node.baseline:.1f}->{node.target:.1f})"
                        dot_node.attr['penwidth'] = self.penwidth_stress
                        dot_node_color = dot_node.attr['color']
                        dot_node.attr['color'] = 'orange'
                        viz_graph(self.dot)
                        dot_node.attr['penwidth'] = self.penwidth_normal
                        if node.val == node.baseline:
                            dot_node.attr['color'] = dot_node_color or "black"
                        elif node.val == node.target:
                            dot_node.attr['color'] = "green"
                        
                self.dfs(node, order)

    def print_credit(self, edge_credit=None):
        if edge_credit is None: edge_credit = self.edge_credit
        for node1, d in edge_credit.items():
            for node2, val in d.items():
                print(f'credit {node1}->{node2}: {val/self.nruns}')


    def credit2dot_pygraphviz(self, edge_credit, format_str, idx=-1):
        '''
        pygraphviz version of credit2dot
        idx: the index of target to visualize, if negative assumes sum
        '''
        G = AGraph(directed=True)

        max_v = 0
        for node1, d in edge_credit.items():
            for node2, val in d.items():
                max_v = max(abs(val/self.nruns), max_v)
        
        for node1, d in edge_credit.items():
            for node2, val in d.items():
                
                v = val/self.nruns
                edge_label = format_str.format(v)

                red = "#ff0051"
                blue = "#008bfb"
                if idx < 0:
                    color = f"{blue}ff"
                else:
                    color = f"{blue}ff" if v < 0 else f"{red}ff" # blue and red
                
                max_w = 5
                min_w = 0.05
                width = abs(v) / max_v * (max_w - min_w) + min_w
                
                if node1.is_dummy_node:
                    continue # should be covered in the next case

                if node2.is_dummy_node:
                    node2 = node2.children[0]

                for node in [node1, node2]:
                    if node not in G:
                        if node.is_noise_node and self.fold_noise:
                            G.add_node(node, shape="point")
                        else:
                            if idx < 0:
                                G.add_node(node, label=node.name)
                            else:
                                txt = self.graph.display_translator\
                                    [node.name](node.target[idx])
                                if type(txt) is str:
                                    fmt = "{}: {}"
                                else:
                                    fmt = "{}: " + format_str
                                G.add_node(node, label=\
                                           fmt.format(node, txt))

                G.add_edge(node1, node2)
                e = G.get_edge(node1, node2)                
                e.attr["weight"] = v
                e.attr["penwidth"] = width
                e.attr["color"] = color
                e.attr["label"] = edge_label
                min_c, max_c = 60, 255
                alpha = "{:0>2}".format(hex(
                    int(abs(v) / max_v * (max_c - min_c) + min_c)
                )[2:]) # skip 0x
                if idx < 0:
                    e.attr["fontcolor"] = f"{blue}{alpha}"
                else:
                    e.attr["fontcolor"] = f"{blue}{alpha}" if v < 0 else\
                        f"{red}{alpha}"
        return G
        
    def credit2dot(self, format_str="{:.2f}", idx=-1):
        '''
        convert the graph to pydot graph for visualization
        e.g.:
        G = cf.credit2dot()
        viz_graph(G)

        idx: the index of target to visualize, if negative assumes sum
        '''
        edge_credit = defaultdict(lambda: defaultdict(int))
        
        # simplify for dummy intermediate node for multi-graph
        for node1, d in self.edge_credit.items():
            for node2, val in d.items():
                if node1.is_dummy_node:
                    continue # should be covered in the next case
                if node2.is_dummy_node:
                    node2 = node2.children[0]

                if idx < 0 and len(val) == 1:
                    idx = 0
                if idx < 0:
                    edge_credit[node1][node2] += np.mean(np.abs(val))
                else:
                    edge_credit[node1][node2] += val[idx]

        return self.credit2dot_pygraphviz(edge_credit, format_str, idx)

class GraphExplainer:
    # todo: do this later
    def __init__(self, graph, baseline_sampler, nsamples=100):
        '''
        graph: graph to explain
        baseline_sampler: sampler for background value
        nsamples: how many runs for each data point
        '''

        def idx_f(idx, f):
            ''' '''
            def f_():
                return f(idx)
            return f_
        
        self.graph = graph
        self.nsamples = nsamples
        self.graph.baseline_sampler = baseline_sampler

    def shap_values(self, X):
        """ Estimate the SHAP values for a set of samples.

        Parameters
        ----------
        X : numpy.array, pandas.DataFrame or scipy.csr_matrix
            A matrix of samples (# samples x # features) on which to explain 
            the model's output.

        Returns
        -------
        For models with a single output this returns a matrix of SHAP values
        (# samples x # features). Each row sums to the difference between the 
        model output for that sample and the expected value of the model output
        (which is stored as expected_value attribute of the explainer).
        """
        """
        this should just be edge_credit, but make credit a vector;
        Now the result is really shap_graphs, not shap_values
        """
        pass

##### helper functions
# graph visualization
def viz_graph(G):
    '''only applicable in ipython notebook setting 
    convert G (pygraphviz) to graphviz format and display with 
    ipython display
    '''
    display(Source(G.string()))

def save_graph(G, name):
    '''
    G is a pygraphviz object;
    save G to a file with name
    '''
    G.layout(prog='dot')
    G.draw(name)
def translator(names, X, X_display):
    '''
    X and X_display are assumed to be convertible to np array
    of shape (n, d)
    '''
    def f(A, B):
        d = dict((a, b) for a, b in zip(A, B))
            
        def f_(val):
            if val in d:
                return d[val]
            return val
        return f_
    
    res = {}
    X = np.array(X).T
    X_display = np.array(X_display).T
    assert X.shape == X_display.shape, "translation must have same shape"
    
    for i, name in enumerate(names):
        res[name] = f(X[i], X_display[i])
    return res

# graph algorithm
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

# graph operation
def flatten_graph(graph):
    '''
    given a graph, return a graph with the graph flattened
    '''

    def node_function(f, node):
        '''
        helper function to record node in context
        f: function Node -> val
        '''
        def f_():
            return f(node)
        return f_

    graph = copy.deepcopy(graph)
    for node in topo_sort(graph):

        if not node.is_target_node:
            if node.args != []:
                
                for arg in node.args: # remove parent's link to this node
                    idx = arg.children.index(node)
                    arg.children = arg.children[:idx] + arg.children[idx+1:]
                
                node.spare_args = node.args
                node.args = [] # remove this node's link to parent
                
                graph.baseline_sampler[node.name] = node_function(lambda node:\
                                                                  node.f(*[graph.baseline_sampler[arg.name]()\
                                                                           for arg in node.spare_args]),
                                                                  node)
                graph.target_sampler[node.name] = node_function(lambda node:\
                                                                node.f(*[graph.target_sampler[arg.name]()\
                                                                         for arg in node.spare_args]),
                                                                node)
                
    return graph

def eval_graph(graph, val_dict):
    for node in topo_sort(graph):
        if node.name in val_dict:
            node.val = val_dict[node.name]
        else:
            node.val = node.f(*[a.val for a in node.args])

        if node.is_target_node:
            return node.val

def merge_nodes(nodes):
    '''
    nodes: assume nodes follow topological order, with the last being the target
    helper function for boundary_graph
    '''
    def merge_h(node1, node2):
        '''assume node 1 depend on node2, otherwise return node1
        helper function for merge_nodes'''
        if node2 in node1.args:

            def create_f(node1, node2):
                def f(*args):
                    node2_args = args[:len(node2.args)]
                    node1_args = args[len(node2.args):]
                    v = node2.f(*node2_args)
                    args = []
                    idx = 0
                    for a in node1.args:
                        if a.name == node2.name:
                            args.append(v)
                        else:
                            args.append(node1_args[idx])
                            idx += 1
                    return node1.f(*args)
                return f

            args = node2.args + [a for a in node1.args if a.name != node2.name]
            node1 = Node(node1.name, create_f(node1, node2), args)

        return node1


    y = nodes[-1]
    assert y.is_target_node, "assumes last one is the target"
    for node in nodes[:-1][::-1]:
        y = merge_h(y, node)
        y.is_target_node = True
    
    # rewire parents: add dumminess to work with multi graph
    dummy_nodes = [Node(arg.name + '_dummy{}'.format(i), lambda x: x, [arg],
                        is_dummy_node=True) for i, arg in enumerate(y.args)]
    y.args = dummy_nodes
    
    for node in dummy_nodes:
        node.children = [y]
            
    # remove link to old node
    for node in dummy_nodes:
        for parent in node.args:
            for n in nodes:
                parent.children = [c for c in parent.children if c.name != n.name]
    
    return y, dummy_nodes
        
def boundary_graph(graph, boundary_nodes=[]):
    '''
    boundary nodes are names of nodes on the boundary
    we collapse all model nodes into a single node
    default just use the source nodes
    '''
    graph = copy.deepcopy(graph)
    
    # find the boundary node's ancestor and include it in the graph with dfs
    boundary_nodes = [node for node in graph if node.name in boundary_nodes] \
        + get_source_nodes(graph)
    visited = set()
    def dfs(node):
        if node in visited: return
        visited.add(node)
        for n in node.args:
            dfs(n)
    
    for node in boundary_nodes:
        dfs(node)
    boundary_nodes = visited
    
    # group C = (D, M)
    nodes_in_d = [] # data side
    nodes_to_merge = []
    for node in topo_sort(graph):
        if node not in boundary_nodes:
            nodes_to_merge.append(node)
            if node.is_target_node:
                break
        else:
            nodes_in_d.append(node)
    
    # merge node in M
    node, dummy_nodes = merge_nodes(nodes_to_merge)
    nodes = nodes_in_d + [node] + dummy_nodes

    return Graph(nodes, graph.baseline_sampler, graph.target_sampler)

def single_source_graph(graph):
    '''
    check if a graph has a single source. If not,
    add an artificial source node
    '''
    graph = copy.deepcopy(graph)
    sources = get_source_nodes(graph)
    if len(sources) == 1:
        return graph
    
    s = Node('seed', is_noise_node=True)
    for node in sources:
        node.add_arg(s)

        def create_f(graph, node):
            def f(s):
                if s == 0:
                    return graph.sample(graph.baseline_sampler, node.name)
                else:
                    return graph.sample(graph.target_sampler, node.name)
                
            return f
        
        node.f = create_f(graph, node)

    graph.baseline_sampler[s.name] = lambda: 0
    graph.target_sampler[s.name] = lambda: 1
    graph.add_node(s)
    graph.reset()
    return graph

def hcluster_graph(graph, source_names, cluster_matrix, verbose=False):
    '''
    cluster_matrix: of form https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage
    source_names: name of the source nodes
    graph: a flat graph with input features
    '''
    graph = copy.deepcopy(graph)
    nodes = sorted(get_source_nodes(graph),
                   key=lambda node: source_names.index(node.name))
    for row in cluster_matrix:
        node1 = nodes[int(row[0])]
        node2 = nodes[int(row[1])]
        if verbose:
            print(f'merging {node1} and {node2} (dist={row[2]}, n={row[3]})')
        s = Node(f"{node1} x {node2}", is_noise_node=True)
        node1.add_arg(s)
        node2.add_arg(s)
        
        def create_f(graph, node):
            def f(s):
                if s == 0:
                    return graph.sample(graph.baseline_sampler, node.name)
                else:
                    return graph.sample(graph.target_sampler, node.name)
            return f

        node1.f = create_f(graph, node1)
        node2.f = create_f(graph, node2)

        graph.baseline_sampler[s.name] = lambda: 0
        graph.target_sampler[s.name] = lambda: 1
        graph.add_node(s)
        nodes.append(s)
        
    graph.reset()
    return graph

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
                  {'x1': lambda: 0},
                  # target to explain
                  {'x1': lambda: 1})
    
    return graph

# sample runs
def main():

    graph = build_graph()    
    cf = CreditFlow(graph, verbose=False, nruns=1)
    cf.run()
    cf.print_credit()

    return cf
    
if __name__ == '__main__':
    main()
