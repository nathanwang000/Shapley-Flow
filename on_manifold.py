'''
perform on manifold perturbation
'''
import sys
sys.path = ['/home/jiaxuan/msr_intern_2020/shap'] + sys.path # todo: change hard code
import pandas as pd
import shap
import numpy as np
import copy
import tqdm
import itertools
import math
from scipy.spatial import distance

class FeatureAttribution:
    '''
    an object can be drawn with bar charts
    e.g. shap.plots.bar(self)
    '''
    def __init__(self, values, input_names):
        self.values = values
        self.input_names = input_names

    def df(self, max_display=None, show=True, values=None):
        l = len(self.input_names)
        return pd.DataFrame(self.values).rename(
            columns={i:name for i,name in zip(range(l), self.input_names)})
    
    def draw(self, sample_ind=-1, max_display=None, show=True, values=None):
        if values is None: values = self.values
        class D(): pass
        b = D()
        b.input_names = self.input_names
        b.values = values[sample_ind]
        b.data = None # np.arange(len(input_names))        
        b.transform_history = []
        shap.plots.bar(b, max_display=max_display, show=show)

class OnManifoldExplainer:

    def __init__(self, f, X, nruns=100, sigma_sq=0.1, orderings=None,
                 single_bg=True):
        '''
        f: the model to explain, when called evaluate the model
        X: background value samples from X, assumes dataframe
        nruns: how many runs for each data point
        orderings: specifies what possible orderings to try; assumes
                   it is a list of list (inner list contains permutation
                   of indices); this is useful for ASV
        sigma: kernel width, per Aas et al. 2019
        '''
        self.nruns = nruns
        self.bg = np.array(X[:1]) # for single bg case
        self.bg_dist = np.array(X)
        self.feature_names =  list(X.columns)
        self.f = f
        self.sigma_sq = sigma_sq
        self.orderings = orderings
        self.single_bg = single_bg

    def mahalanobis_dist_sq(self, v1, v2, c):
        '''
        output: v[i]^T @ inv(C) @ v[i] for each i
        v1: foreground sample (1, d)
        v2: background samples (n, d)
        c: covariance matrix (d, d)
        '''
        v = v1 - v2 # (n, d)
        d = v.shape[1]
        inv = np.linalg.inv(c + 1e-8 * np.random.uniform(0,1,c.shape))\
            if d > 1 else 1 / (c + 1e-8)
        dist_sq = (v.dot(inv) * v).sum(1) / d
        return dist_sq

    def mahalanobis_dist_sq2(self, v1, v2, c):
        '''
        output: v[i]^T @ inv(C) @ v[i] for each i
        v1: foreground sample (1, d)
        v2: background samples (n, d)
        c: covariance matrix (d, d)
        '''
        d = v1.shape[1]        
        inv = np.linalg.inv(c + 1e-10 * np.random.uniform(0,1,c.shape))\
            if d > 1 else 1 / (c + 1e-10)
        dist_sq = []
        for i in range(len(v2)):
            dist_sq.append(distance.mahalanobis(v1[0], v2[i], inv))
        dist_sq = np.array(dist_sq)**2 / d
        return dist_sq

    def mahalanobis_dist_sq3(self, v1, v2, c):
        '''
        output: v[i]^T @ inv(c) @ v[i] for each i where v[i] = v1-v2[i]
        v1: foreground sample (1, d)
        v2: background samples (n, d)
        c: covariance matrix (d, d)
        '''
        def cov(X): # same output as np.cov
            Ex = X - X.mean(0).reshape(1, -1)
            return Ex.T @ Ex / (Ex.shape[0] - 1)
            
        v = v1 - v2
        n, d = v2.shape

        if d == 1:
            inv = 1/(c + 1e-10)
            dist_sq = (v * inv * v).ravel()
        else:
            Ex = v2 - v2.mean(0).reshape(1, -1) # need to calculate cov(v2) b/c bg
            u, s, vh = np.linalg.svd(Ex) # u: (n, n), vh: (d, d)
            
            # fill s: numerical error if using np version when n < d b/c not invertible
            m = np.zeros((n, d))
            m[:len(s), :len(s)] = np.diag(s)
            s = m # (n, d)
            # assert np.allclose(Ex, np.dot(u, np.dot(s, vh))), "must match"

            S = s.T @ s # (d, d)
            inv_S = ((n-1) / (np.diag(S) + 1e-10)).reshape(1, d) # (1, d)

            # print((vh.T * inv_S).dot(vh)) # the real S^-1

            # inv(c) = vh.T (n-1) / diag(s**2) vh
            a = v @ vh.T # (n, d)
            dist_sq = a * inv_S # (n, d)
            dist_sq = (dist_sq * a).sum(1) / d # (n,)
            
        return dist_sq
    
    def payoff(self, C, x):
        '''
        C is the coalition; on manifold version
        x: the sample to explain
        '''
        # this is only needed because I have single baseline
        # remove when I account for more
        if len(C) == 0:
            if self.single_bg:
                return self.f(self.bg)
            else:
                return self.f(self.bg_dist).mean()

        bg_c = self.bg_dist[:, [i for i in C]]
        # if bg_c is 1x1, then np.cov doesn't work
        S_c = np.cov(bg_c.T)  # each row of cov function need to be a variable
        x_c = x[[i for i in C]]

        # calculate mahalanobis distance
        v1 = x_c.reshape(1, -1) # (1, d)
        v2 = bg_c # (n, d)
        dist_sq = self.mahalanobis_dist_sq3(v1, v2, S_c) # could choose 1 2 or 3
        
        # calculate the kernel weights
        exponent = dist_sq / 2 / self.sigma_sq
        # exponent -= exponent.max() # avoid numerical error
        w = np.exp(-exponent) # (n_bg,)

        # get the weighted output
        current_x = copy.deepcopy(self.bg_dist)
        for c in C:
            current_x[:, c] = x[c]
        o = self.f(current_x) # (n_bg,)

        # print(f'example {i}: coalition {C}')
        # print(f'example {i} with weight: {w}')
        # print(f'example {i}: current x = {x}')

        # avoid division by 0
        if w.sum() == 0:
            w = np.ones_like(w)

        v = (o * w / w.sum()).sum()
        # print("C, v, distsq:", C, v, dist_sq)
        return v
        
    def shap_values(self, X):
        """ Estimate the SHAP values for a set of samples.

        Parameters
        ----------
        X : pandas.DataFrame
            A matrix of samples (# samples x # features) on which to explain 
            the model's output.

        Returns
        -------
        feature attribution object that can be drawn
        """
        payoff = self.payoff
        n_fg, d = X.shape
        self.fg = np.array(X)
        self.values = np.zeros((n_fg, d))

        nruns = self.nruns if self.nruns <= math.factorial(d) else math.factorial(d)
        for sample in tqdm.trange(len(X)):
            permutations = itertools.permutations(list(range(d)))
            x = np.array(X)[sample]
            for i in range(nruns):
                # sample a random ordering of features
                if self.orderings is None:
                    if self.nruns < math.factorial(d):
                        order = np.random.permutation(d)
                    else:
                        order = next(permutations)
                else:
                    order = self.orderings[np.random.choice(len(self.orderings))]
                    assert (np.array(sorted(order)) == np.arange(d)).all()
                # follow the ordering to calculate payoff function difference
                C = []
                v_last = payoff(C, x)
                for i in order:
                    C.append(i)
                    v =  payoff(C, x)
                    self.values[sample, i] += v - v_last
                    v_last = v

        self.values /= nruns
        return FeatureAttribution(self.values, self.feature_names)
        
class IndExplainer:

    def __init__(self, f, X, nruns=100):
        '''
        f: the model to explain, when called evaluate the model
        X: background value samples from X, assumes dataframe
        nruns: how many runs for each data point
        '''
        self.nruns = nruns
        # this only support single baseline, see ipynb how to use
        # linearity to compute multiple baseline ind explainer
        self.bg = np.array(X[:1]) 
        self.bg_dist = np.array(X)
        self.feature_names =  list(X.columns)
        self.f = f

    def payoff(self, C):
        '''
        C is the coalition; independent perturbation version
        '''
        x = copy.deepcopy(self.bg).repeat(len(self.fg), 0)
        for c in C:
            x[:, c] = self.fg[:, c]
        return self.f(x)

    def shap_values(self, X):
        """ Estimate the SHAP values for a set of samples.

        Parameters
        ----------
        X : pandas.DataFrame
            A matrix of samples (# samples x # features) on which to explain 
            the model's output.

        Returns
        -------
        feature attribution object that can be drawn
        """
        payoff = self.payoff
        n_fg, d = X.shape
        self.fg = np.array(X)
        self.values = np.zeros((n_fg, d))

        for i in tqdm.trange(self.nruns):
            # sample an random ordering of features
            order = np.random.permutation(d)
            # follow the ordering to calculate payoff function difference
            C = []
            v_last = payoff(C)
            for i in order:
                C.append(i)
                v =  payoff(C)
                self.values[:, i] += v - v_last
                v_last = v

        self.values /= self.nruns
        return FeatureAttribution(self.values, self.feature_names)
