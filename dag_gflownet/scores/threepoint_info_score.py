from pyitlib import discrete_random_variable as drv
import numpy as np
from dag_gflownet.scores.base import BaseScore, LocalScore
import logging



class threepoint_info_score(BaseScore):

    def __init__(self,dataframe,prior):
        super().__init__(dataframe, prior)
        self.Magnitude =100
        self.c_neg =-self.Magnitude
        self.c_pos =self.Magnitude
        self.threshold = 0.4/35

    def get_local_scores(self, target, indices, indices_after=None): # trivial implementation of get_local_scores to comply with BaseScore
        return soft_info_constraint([target,indices],self.dataframe)

    def conditionalMutual_discrete(self,indexes, dataframe):
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

    def mutual_info_discrete(self,indexes,dataframe):
        ix,iy = indexes
        # assert ix!=iy
        return drv.information_mutual(dataframe.iloc[:,ix],dataframe.iloc[:,iy])

    def threepoint_info_discreteF(self, indexes, dataframe):
        I_ix_iy_Given_z = self.conditionalMutual_discrete(indexes,dataframe)
        I_ix_iy = self.mutual_info_discrete(indexes[0:2],dataframe)
        return I_ix_iy-I_ix_iy_Given_z

    def supervision(self):
        pass


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
        constraint_score = inverseV_structures_score + v_structures_score
        return constraint_score



    def score_v_structures(self,new_edge_idx, pa_xj,child_xi,child_xj,dataframe):
        score = 0
        x_i,x_j = new_edge_idx
        if len(pa_xj) >0:
            for pxj in pa_xj:
                I_pxj_xi_xj = self.threepoint_info_discreteF([pxj,x_i,x_j],dataframe)
                #      print('three point info v:  pa_xj->xj<-xi ',I_pxj_xi_xj)
                if I_pxj_xi_xj < -self.threshold:
                    logging.info(f'reward v structure, {I_pxj_xi_xj}')
                    score -= self.c_neg
                else:
                    logging.info(f'punish v structure, {I_pxj_xi_xj}')
                    score += self.c_neg  #notice we did not change this part
        intersect = np.intersect1d(child_xj, child_xi)
        if len(intersect) >0:
            for s in intersect:
                I_xi_xj_s = self.threepoint_info_discreteF([x_i,x_j,s],dataframe)
                #  print('three point info v: xi->child(xi)<-xj')
                if I_xi_xj_s< -self.threshold:
                    logging.info(f'reward v structure, {I_xi_xj_s}')
                    score -=self.c_neg
                else:
                    logging.info(f'punish v structure, {I_xi_xj_s}')
                    score +=0.2*self.c_neg
        return score

    def score_inverseV_structures(self, new_edge_idx,child_xi,dataframe):
        score = 0
        x_i,x_j = new_edge_idx
        if len(child_xi) >0:
            for s in child_xi:
                I_xj_cxi_xi = self.threepoint_info_discreteF([s,x_j,x_i],dataframe)
                # print('three point info v:  xj<-xi->child(xi)')
                if I_xj_cxi_xi>self.threshold:
                    logging.info(f'reward inverse V structure, {I_xj_cxi_xi}')
                    score +=self.c_pos
                else:
                    logging.info(f'punish inverse V structure, {I_xj_cxi_xi}')
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

