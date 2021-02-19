'''
helper file for parallel credit flow in flow.py
used in run_subprocess
'''
import sys
import dill
import numpy as np
import xgboost
import pandas as pd
import time

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
    start_time = time.time()
    time_log = []
    
    fn1 = sys.argv[1] # get credit flow
    fn2 = sys.argv[2]
    with open(fn1, 'rb') as f:
        cf = f.read()

    now = time.time()
    time_log.append(f"done reading: {now - start_time:.2f} s")
    start_time = now

    cf = dill.loads(cf)
    cf.run()

    now = time.time()
    time_log.append(f"done credit flow: {now - start_time:.2f} s")
    start_time = now

    edge_credit = {}
    for node1, d in cf.edge_credit.items():
        for node2, val in d.items():
            if node1.name not in edge_credit:
                edge_credit[node1.name] = {}
            if node2.name not in edge_credit[node1.name]:
                edge_credit[node1.name][node2.name] = 0
            edge_credit[node1.name][node2.name] += val
    o = edge_credit

    now = time.time()
    time_log.append(f"done edge convert: {now - start_time:.2f} s")
    start_time = now
    
    with open(fn2, 'wb') as f:
        dill.dump(o, f)

    now = time.time()
    time_log.append(f"done output: {now - start_time:.2f} s")
    start_time = now

    with open(fn2 + "_" + "time", 'w') as f:
        f.write("\n".join(time_log))
