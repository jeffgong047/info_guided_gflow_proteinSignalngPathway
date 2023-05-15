import numpy as np
import scipy.stats._entropy as H
from pyitlib import discrete_random_variable as drv






def conditionalMutual_discrete(indexes, dataframe):
    '''

    :param indexes: [ix,iy,iz]
    :param dataframe:
    :return: I(X,Y|Z)
    '''
  #  breakpoint() # debug
    ix,iy,iz = indexes
    return drv.information_mutual_conditional(dataframe.iloc[:,ix],dataframe.iloc[:,iy],dataframe.iloc[:,iz])


def mutual_info_cal(indexes, dataframe):
    '''
    This function calculates two point mutual information
    :args observation of variable x , obx ; observation of variable y, oby ;
    :return two point mutual information

    formula  I(X,Y) = D_kl(P(X,Y)|P(X)*P(Y))
    '''
    #
    assert len(indexes)==2
    p_x_y = joint_distri(indexes ,dataframe)
    MI_x_y = np.sum()
    pass
    return MI_x_y

def entropy(index,dataframe):
    '''
    :param index: index of random variable
    :param dataframe: pandas dataframe
    :return: entropy
    '''
    return H(dataframe.iloc[:,index])

def condition_entropy():
    pass

def joint_entropy():
    pass



def three_point_multivariate_information(indexes, dataframe):
    '''

    :param indexes: [] contains indexes
    :param dataframe: dataframe that contains all the data
    :return: I(X,Y,Z)
    '''
    ix,iy,iz = indexes
    H_X = entropy(ix,dataframe)
    H_Y = entropy(iy,dataframe)
    H_Z = entropy(iz, dataframe)
    H_XY = entropy([ix,iy],dataframe)
    H_YZ = entropy([iy,iz],dataframe)
    H_XZ = entropy([ix,iz], dataframe)
    H_XYZ = entropy([ix,iy,iz], dataframe)
    return H_X + H_Y + H_Z - H_XY - H_YZ - H_XZ + H_XYZ

def joint_distribution(indexes,dataframe):
    pass

def frequency_to_distribution_discrete(index, dataframe):
    '''
    :param index:
    :param dataframe:
    :return:

    pesudocode: use counting technique to calculate the statistics of variables in the index, assuming variables from discrete
    '''
    boundary = dataframe['boundary']
    (left , right) = boundary
    hash_f = lambda left,list: list-left
    for i in index:
        data = dataframe.iloc(i) # ith row or column
        hash_list = hash_f(left,data)
        count_list = np.zeros((right-left))
        num_of_data = len(hash_list)
        for i in range(num_of_data):
            count_list[hash_list[i]]+=1
        count_list /= num_of_data




def mutual_info_decomposition_threePoint(index_X,index_Y,index_Z,dataframe):
    '''

    :param X,Y,Z are realization of random variables and Z is the mid node, index_? is the index value of ? in the dataframe
    :return: conditional_mutual_information
    '''
    I_X_Y = mutual_info_cal(indexes = [index_X,index_Y], dataframe = dataframe)
    I_X_Y_Z = three_point_multivariate_information(indexes= [index_X,index_Y,index_Z], dataframe=dataframe)
    I_XY_Given_Z = I_X_Y - I_X_Y_Z
    return I_XY_Given_Z


def do_structure():
    pass


