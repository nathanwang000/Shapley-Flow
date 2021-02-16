'''
pseudo code for the exact algorithm

assumptions: a) single baseline

adaptation to multiple baselines can be easily extended
'''
import math

def ShapleyFlow(G):
    '''
    Input: G
    Output: \psi: E -> R; that is the edge_credit
    '''
    def run(node):

        def save_state(node, state):
            # record original settings from the node and downward
            if len(node.children) == 0: return
            for c in node.children:
                state[c] = {}
                state[c]['from_node'] = c.from_node # default None
                state[c]['last_val'] = c.last_val # default baseline
                state[c]['visible_val'] = dict((n, v) for n, v in\
                                               c.visible_arg_values.items())
                state[c]['val'] = c.val # default baseline
                save_state(c, state)
        
        def load_state(node, state):
            if len(node.children) == 0: return
            for c in node.children:
                c.from_node = state[c]['from_node']
                c.last_val = state[c]['last_val']
                c.visible_arg_values = state[c]['visible_val']
                c.val = state[c]['val']
                load_state(c, state)

        # basecase
        edge_credit = defaultdict(lambda: defaultdict(int))
        if len(node.children) == 0: # leaf node
            credit = node.val - node.last_val            
            return {node.from_node: {node: credit}}

        # save the current state
        state = {}
        save_state(node, state)
        
        # try all permutations of message ordering
        permutations = itertools.chain.from_iterable(
            itertools.permutations(node.children)
        )
        nruns = math.factorial(len(node.children))

        for _i in range(nruns):
            children_order = next(permutations)

            # restore to original state
            load_state(node, state)

            # update the value through depth first search
            for c in children_order:
                c.from_node = node
                c.last_val = c.val
                c.visible_arg_values[node] = node.val # message arrives
                c.val = c.f(*[c.visible_arg_values[arg] for arg in c.args])
                ec = run(c)

                # update edge credit of all upstream nodes of the current node
                for node1, v in ec.items():
                    for node2, credit in v.items():
                        edge_credit[node1][node2] += credit / nruns # average over runs, because will update c this many times
                                
                credit = np.vstack([ec[node][c] for c in node.children if c in ec[node]]).sum(0)
                if node.from_node is not None:
                    edge_credit[node.from_node][node] += credit/ nruns
                
        return edge_credit

    # preprocess the G
    G = single_source_G(G)
    G.reset() # set all features to the baseline value
    s = get_source_nodes(G)[0]
    s.val = s.target # set the source to target value
    
    # run the algorithm
    return run(s)

def run_bruteforce_sampling(self):
    '''
    run shap flow algorithm to fairly allocate credit for distributed edge
    axioms
    '''
    sources = get_source_nodes(self.graph)

    nruns = sum([math.factorial(n.children) for n in self.graph])
    
    for _ in run_range:
        # make value back to baselines
        self.reset()

        order = list(np.random.permutation(sources))

        # follow the order
        for node in order:
            node.last_val = node.val # update last val
            node.val = node.target # turn on the node
            node.from_node = None # turn off the source

            self.dfs(node)

    # normalize edge credit
    for _, v in self.edge_credit.items():
        for node2 in v:
            v[node2] = v[node2] / nruns
