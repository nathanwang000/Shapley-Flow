'''
perform on manifold perturbation
'''
import sys
sys.path = ['/home/jiaxuan/msr_intern_2020/shap'] + sys.path # todo: change hard code
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
        self.data = None # np.arange(len(input_names))        

    def draw(self, sample_ind=-1, max_display=None, show=True, values=None):
        if values is None: values = self.values
        class D(): pass
        b = D()
        b.input_names = self.input_names
        b.values = values[sample_ind]
        b.data = self.data
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
        '''
        self.nruns = nruns
        self.bg = np.array(X[:1]) # todo: currently only support one baseline
        self.bg_dist = np.array(X)
        self.feature_names =  list(X.columns)
        self.f = f
        self.sigma_sq = sigma_sq
        self.orderings = orderings
        self.single_bg = single_bg

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
        S_c = np.cov(bg_c.T)  # each row of cov function need to be a variable

        x_c = x[[i for i in C]]

        # # calculate mahalanobis distance
        diff = (x_c.reshape(1, -1) - bg_c) # (n_bg, d)
        inv = np.linalg.inv(S_c + 1e-8 * np.random.uniform(0,1,S_c.shape))\
            if len(C) > 1 else 1 / (S_c + 1e-8)
        dist_sq = (diff.dot(inv) * diff).sum(1) / len(C)

        # another implementation of the previous 3 lines: todo: deal with numerical issue after submission!
        # def cov(X): # save as np.cov
        #     Ex = X - X.mean(0).reshape(1, -1)
        #     return Ex.T @ Ex / (Ex.shape[0] - 1)
            
        # diff = (x_c.reshape(1, -1) - bg_c) # (n_bg, d)
        # if len(C) == 1:
        #     inv = 1/(S_c + 1e-10)
        #     dist_sq = diff * inv * diff
        # else:
        #     inv = np.linalg.inv(S_c + 1e-10 * np.random.uniform(0,1,S_c.shape))\
        #         if len(C) > 1 else 1 / (S_c + 1e-10)
        #     gt = []
        #     for i in range(len(bg_c)):
        #         gt.append(distance.mahalanobis(x_c, bg_c[i], inv))
        #     gt = np.array(gt)**2 / len(C)
        #     dist_sq = gt
            
        #     # Ex = bg_c - bg_c.mean(0).reshape(1, -1)
        #     # u, s, vh = np.linalg.svd(Ex) # u: (n, n), vh: (d, d)
        #     # # # fill s
        #     # # m = np.zeros(max(len(u), len(vh)))
        #     # # m[:len(s)] = s
        #     # # s = np.diag(m)
        #     # # s = s[:len(u), :len(vh)]
            
        #     # # inv(S_c) = vh.T (n-1) / diag(s**2) vh
        #     # a = (diff @ vh.T)[:, :len(s)]
        #     # s = (bg_c.shape[0]-1) / (s**2+1e-10)
        #     # dist_sq = a * s
        #     # dist_sq = (dist_sq * a).sum(1) / len(C)
        #     # print(dist_sq)
        
        # calculate the kernel weights
        exponent = dist_sq / 2 / self.sigma_sq
        # exponent -= exponent.max() # avoid numerical error
        w = np.exp(-exponent) # (n_bg,)

        # if len(C) > 1:
        #     _, s, _ = np.linalg.svd(inv)
        #     print(dist_sq, w, s)

        # get the weighted output
        current_x = copy.deepcopy(self.bg_dist)
        for c in C:
            current_x[:, c] = x[c]
        o = self.f(current_x) # (n_bg,)

        # print(f'example {i}: coalition {C}')
        # print(f'example {i} with weight: {w}')
        # print(f'example {i}: current x = {x}')

        v = (o * w / w.sum()).sum()

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
        # this assumes a single baseline
        nruns = self.nruns if self.nruns <= math.factorial(d) else math.factorial(d)
        
        for sample in tqdm.trange(len(X)):
            permutations = itertools.permutations(list(range(d)))
            x = np.array(X)[sample]
            for i in range(nruns):
                # sample an random ordering of features
                if self.orderings is None:
                    if self.nruns <= math.factorial(d):
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
        self.bg = np.array(X[:1]) # todo: currently only support one baseline
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
