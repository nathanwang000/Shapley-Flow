import sys
import dill
import numpy as np
import xgboost
import pandas as pd

def wrap_run(cf):
    cf.run()

    edge_credit = {}
    for node1, d in cf.edge_credit.items():
        for node2, val in d.items():
            if node1.name not in edge_credit:
                edge_credit[node1.name] = {}
            if node2.name not in edge_credit[node1.name]:
                edge_credit[node1.name][node2.name] = 0
            edge_credit[node1.name][node2.name] += val

    return edge_credit

if __name__ == '__main__':
    fn1 = sys.argv[1] # get credit flow
    fn2 = sys.argv[2]
    with open(fn1, 'rb') as f:
        o = wrap_run(dill.loads(f.read()))

    # print(o)
    with open(fn2, 'wb') as f:
        dill.dump(o, f)

