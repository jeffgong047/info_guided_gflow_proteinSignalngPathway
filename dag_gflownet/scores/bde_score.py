import math
import numpy as np
import pandas as pd

from scipy.special import gammaln
from collections import namedtuple

from dag_gflownet.scores.base import BaseScore, LocalScore
import inspect

StateCounts = namedtuple('StateCounts', ['key', 'counts'])

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

class BDeScore(BaseScore):
    """BDe score.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing the (discrete) dataset D. Each column
        corresponds to one variable. If there is interventional data, the
        interventional targets must be specified in the "INT" column (the
        indices of interventional targets are assumed to be 1-based).
    
    prior : `BasePrior` instance
        The prior over graphs p(G).

    equivalent_sample_size : float (default: 1.)
        The equivalent sample size (of uniform pseudo samples) for the
        Dirichlet hyperparameters. The score is sensitive to this value,
        runs with different values might be useful.
    """
    def __init__(self, data, prior, equivalent_sample_size=1.):
        if 'INT' in data.columns:  # Interventional data
            # Indices should start at 0, instead of 1;
            # observational data will have INT == -1.
            self._interventions = data.INT.map(lambda x: int(x) - 1)
            data = data.drop(['INT'], axis=1)
        else:
            self._interventions = np.full(data.shape[0], -1)
        self.equivalent_sample_size = equivalent_sample_size
        super().__init__(data, prior)

        self.state_names = {
            column: sorted(self.data[column].cat.categories.tolist())
            for column in self.data.columns
        }
        self.c_neg = -50
        self.c_pos = 50

    def get_local_scores(self, target, indices, indices_after=None):
        # Get all the state counts
        #print(inspect.getouterframes( inspect.currentframe() )[1])
        #print("get_local_scores: ", "indices" , indices)
        state_counts_before, state_counts_after = self.state_counts(
            target, indices, indices_after=indices_after)
        local_score_after = self.local_score(*state_counts_after)
        if state_counts_before is not None:
            local_score_before = self.local_score(*state_counts_before)
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
            for s in child_xi:
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




    def state_counts(self, target, indices, indices_after=None):
        # Source: pgmpy.estimators.BaseEstimator.state_counts()
        #print("state_counts: ", 'target', target, 'indices', indices)
        all_indices = indices if (indices_after is None) else indices_after
        parents = [self.column_names[index] for index in all_indices]
        variable = self.column_names[target]

        data = self.data[self._interventions != target]
        data = data[[variable] + parents].dropna()

        state_count_data = (data.groupby([variable] + parents)
                                .size()
                                .unstack(parents))

        if not isinstance(state_count_data.columns, pd.MultiIndex):
            state_count_data.columns = pd.MultiIndex.from_arrays(
                [state_count_data.columns]
            )

        parent_states = [self.state_names[parent] for parent in parents]
        columns_index = pd.MultiIndex.from_product(parent_states, names=parents)

        state_counts_after = StateCounts(
            key=(target, tuple(all_indices)),
            counts=(state_count_data
                .reindex(index=self.state_names[variable], columns=columns_index)
                .fillna(0))
        )

        if indices_after is not None:
            subset_parents = [self.column_names[index] for index in indices]
            if subset_parents:
                data = (state_counts_after.counts
                    .groupby(axis=1, level=subset_parents)
                    .sum())
            else:
                data = state_counts_after.counts.sum(axis=1).to_frame()

            state_counts_before = StateCounts(
                key=(target, tuple(indices)),
                counts=data
            )
        else:
            state_counts_before = None

        return (state_counts_before, state_counts_after)

    def local_score(self, key, counts):
        counts = np.asarray(counts)
        num_parents_states = counts.shape[1]
        num_parents = len(key[1])

        log_gamma_counts = np.zeros_like(counts, dtype=np.float_)
        alpha = self.equivalent_sample_size / num_parents_states
        beta = self.equivalent_sample_size / counts.size

        # Compute log(gamma(counts + beta))
        gammaln(counts + beta, out=log_gamma_counts)

        # Compute the log-gamma conditional sample size
        log_gamma_conds = np.sum(counts, axis=0, dtype=np.float_)
        gammaln(log_gamma_conds + alpha, out=log_gamma_conds)

        local_score = (
            np.sum(log_gamma_counts)
            - np.sum(log_gamma_conds)
            + num_parents_states * math.lgamma(alpha)
            - counts.size * math.lgamma(beta)
        )


        return LocalScore(
            key=key,
            score=local_score,
            prior=self.prior(num_parents)
        )
