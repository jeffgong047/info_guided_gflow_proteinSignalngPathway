import math
import numpy as np

from scipy.special import gammaln

from dag_gflownet.scores.base import BaseScore, LocalScore
from collections import namedtuple
import pandas as pd

from scipy.special import gammaln

import inspect

def logdet(array):
    _, logdet = np.linalg.slogdet(array)
    return logdet


from pyitlib import discrete_random_variable as drv

def conditionalMutual_discrete(indexes, dataframe):
    '''

    :param indexes: [ix,iy,iz]
    :param dataframe:
    :return: I(X,Y|Z)
    '''
    ix,iy,iz = indexes
    #   assert ix!=iy and iy!=iz and ix!=iz
    # print('conditional mutual information,',drv.information_mutual_conditional(dataframe.iloc[:,ix],dataframe.iloc[:,iy],dataframe.iloc[:,iz]))
    # print('indices x,y,z,',ix,iy,iz)
    return drv.information_mutual_conditional(dataframe.iloc[:,ix],dataframe.iloc[:,iy],dataframe.iloc[:,iz])

def mutual_info_discrete(indexes,dataframe):
    ix,iy = indexes
    # assert ix!=iy
    return drv.information_mutual(dataframe.iloc[:,ix],dataframe.iloc[:,iy])

def threepoint_info_discreteF(indexes, dataframe):
    I_ix_iy_Given_z = conditionalMutual_discrete(indexes,dataframe)
    I_ix_iy = mutual_info_discrete(indexes[0:2],dataframe)
    return I_ix_iy-I_ix_iy_Given_z


class BGeScore(BaseScore):
    r"""BGe score.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing the (continuous) dataset D. Each column
        corresponds to one variable. The dataset D is assumed to only
        contain observational data (a `INT` column will be treated as
        a continuous variable like any other).

    prior : `BasePrior` instance
        The prior over graphs p(G).

    mean_obs : np.ndarray (optional)
        Mean parameter of the Normal prior over the mean $\mu$. This array must
        have size `(N,)`, where `N` is the number of variables. By default,
        the mean parameter is 0.

    alpha_mu : float (default: 1.)
        Parameter $\alpha_{\mu}$ corresponding to the precision parameter
        of the Normal prior over the mean $\mu$.

    alpha_w : float (optional)
        Parameter $\alpha_{w}$ corresponding to the number of degrees of
        freedom of the Wishart prior of the precision matrix $W$. This
        parameter must satisfy `alpha_w > N - 1`, where `N` is the number
        of varaibles. By default, `alpha_w = N + 2`.
    """
    def __init__(
            self,
            data,
            prior,
            mean_obs=None,
            alpha_mu=1.,
            alpha_w=None
        ):
        num_variables = len(data.columns)
        if mean_obs is None:
            mean_obs = np.zeros((num_variables,))
        if alpha_w is None:
            alpha_w = num_variables + 2.
        super().__init__(data, prior)
        self.mean_obs = mean_obs
        self.alpha_mu = alpha_mu
        self.alpha_w = alpha_w
        self.c_neg = -50
        self.c_pos = 50
        self.num_samples = self.data.shape[0]
        self.t = (self.alpha_mu * (self.alpha_w - self.num_variables - 1)) / (self.alpha_mu + 1)

        T = self.t * np.eye(self.num_variables)
        data = np.asarray(self.data)
        data_mean = np.mean(data, axis=0, keepdims=True)
        data_centered = data - data_mean

        self.R = (T + np.dot(data_centered.T, data_centered)
            + ((self.num_samples * self.alpha_mu) / (self.num_samples + self.alpha_mu))
            * np.dot((data_mean - self.mean_obs).T, data_mean - self.mean_obs)
        )
        all_parents = np.arange(self.num_variables)
        self.log_gamma_term = (
            0.5 * (math.log(self.alpha_mu) - math.log(self.num_samples + self.alpha_mu))
            + gammaln(0.5 * (self.num_samples + self.alpha_w - self.num_variables + all_parents + 1))
            - gammaln(0.5 * (self.alpha_w - self.num_variables + all_parents + 1))
            - 0.5 * self.num_samples * math.log(math.pi)
            + 0.5 * (self.alpha_w - self.num_variables + 2 * all_parents + 1) * math.log(self.t)
        )

    def local_score(self, target, indices):
        num_parents = len(indices)

        if indices:
            variables = [target] + list(indices)

            log_term_r = (
                0.5 * (self.num_samples + self.alpha_w - self.num_variables + num_parents)
                * logdet(self.R[np.ix_(indices, indices)])
                - 0.5 * (self.num_samples + self.alpha_w - self.num_variables + num_parents + 1)
                * logdet(self.R[np.ix_(variables, variables)])
            )
        else:
            log_term_r = (-0.5 * (self.num_samples + self.alpha_w - self.num_variables + 1)
                * np.log(np.abs(self.R[target, target])))

        return LocalScore(
            key=(target, tuple(indices)),
            score=self.log_gamma_term[num_parents] + log_term_r,
            prior=self.prior(num_parents)
        )

    def get_local_scores(self, target, indices, indices_after=None):
        all_indices = indices if (indices_after is None) else indices_after
        local_score_after = self.local_score(target, all_indices)
        if indices_after is not None:
            local_score_before = self.local_score(target, indices)
        else:
            local_score_before = None
        return (local_score_before, local_score_after)


    def soft_info_constraint(self,indices,adjacency_matrix):
        '''

        :param indices:
        :param adjacency_matrix:
        notation: xi->xj is the notation of the assumed new edge being added
        :return:
        '''
        source, target = indices
        dataframe = self.data
        [child_xi,pa_xi,child_xj, pa_xj] = self.getNeighborhoods(source,target, adjacency_matrix)
        #score structures
        v_structures_score = self.score_v_structures(indices, pa_xj ,child_xi, child_xj ,dataframe)
        inverseV_structures_score = self.score_inverseV_structures(indices,child_xi,dataframe)
        #  do_structures = score_do_structures(child_xi, pa_xi, child_xj, pa_xj)
        constraint_score = v_structures_score + inverseV_structures_score
        return constraint_score




    def score_v_structures(self,new_edge_idx, pa_xj,child_xi,child_xj,dataframe):
        score = 0
        x_i,x_j = new_edge_idx
        if len(pa_xj) >0:
            for pxj in pa_xj:
                I_pxj_xi_xj = threepoint_info_discreteF([pxj,x_i,x_j],dataframe)
                #      print('three point info v:  pa_xj->xj<-xi ',I_pxj_xi_xj)
                if I_pxj_xi_xj < -0.4:
                    score -= self.c_neg
                else:
                    score += self.c_neg
        intersect = np.intersect1d(child_xj, child_xi)
        if len(intersect) >0:
            for s in intersect:
                I_xi_xj_s = threepoint_info_discreteF([x_i,x_j,s],dataframe)
                #  print('three point info v: xi->child(xi)<-xj')
                if I_xi_xj_s< -0.4:
                    score -=self.c_neg
                else:
                    score +=0.2*self.c_neg
        return score

    def score_inverseV_structures(self, new_edge_idx,child_xi,dataframe):
        score = 0
        x_i,x_j = new_edge_idx
        if len(child_xi) >0:
            for s in child_xi:  #I(x,y|z) I(x,y,z) =I(X,Y)- I(x,y|z)
                I_xj_cxi_xi = threepoint_info_discreteF([s,x_j,x_i],dataframe)
                # print('three point info v:  xj<-xi->child(xi)')
                if I_xj_cxi_xi>0.4:
                    score +=self.c_pos
                else:
                    score -=0.2*self.c_pos
        return score

    def getNeighborhoods(self,source, target, adjacency_matrix):
        child_xi = self.to_index(adjacency_matrix[0, source],[source,target])
        pa_xi = self.to_index(adjacency_matrix[0, :,source],[source,target])
        child_c = self.to_index(adjacency_matrix[0,target],[source,target])
        pa_c = self.to_index(adjacency_matrix[0,:,target], [source,target])
        return [child_xi,pa_xi, child_c, pa_c]

    def to_index(self,arr,exclude):
        indices=[]
        for i in range(len(arr)):
            #breakpoint()
            if arr[i] ==1 and i not in exclude:
                indices.append(i)
        return indices
