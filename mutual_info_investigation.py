import sklearn
import numpy as np
import pandas as pd
from dag_gflownet.utils.graph import sample_erdos_renyi_linear_gaussian
from dag_gflownet.utils.sampling import sample_from_linear_gaussian
import os
from pyitlib import discrete_random_variable as drv
from numpy.random import default_rng
from pathlib import Path



import pandas as pd
import urllib.request
import gzip

from pathlib import Path
from numpy.random import default_rng
from pgmpy.utils import get_example_model
from pgmpy.models import BayesianNetwork
from dag_gflownet.utils.graph import sample_erdos_renyi_linear_gaussian
from dag_gflownet.utils.sampling import sample_from_linear_gaussian
import os

def download(url, filename):
    if filename.is_file():
        return filename
    filename.parent.mkdir(exist_ok=True)

    # Download & uncompress archive
    with urllib.request.urlopen(url) as response:
        with gzip.GzipFile(fileobj=response) as uncompressed:
            file_content = uncompressed.read()

    with open(filename, 'wb') as f:
        f.write(file_content)

    return filename


def check_no_redundancy(a, b, c):
    # Create a set from the three integers
    s = set([a, b, c])
    # Check if the length of the set is 3
    if len(s) == 3:
        return True
    else:
        return False


def get_basic_statistics(choice, normalize):
    '''
    Enumerate over all elements in the dataset to get statistics for
    dict= {key='x_y_z' value= [mutual information, conditional mutual information, threepoint information ] ,structure}
    :param choice:
    :return:
    '''
    pd_basic_stat = pd.DataFrame(columns=["nodes_index","MI","conMI", "threepointInfo"])

    #load graph
    num_samples = 875 #defautl
    if choice == 'erdos_renyi':
        # num_samples =100
        # num_variables = 5
        # num_edges = 5
        # graph = sample_erdos_renyi_linear_gaussian(
        #     num_variables=num_variables,
        #     num_edges=num_edges,
        #     loc_edges=0.0,
        #     scale_edges=1.0,
        #     obs_noise=0.1,
        #     rng=default_rng()
        # )
        # data = sample_from_linear_gaussian(
        #     graph,
        #     num_samples=num_samples,
        #     rng=default_rng()
        # )
        data = pd.read_csv('erdos_renyi_samples.csv')
    elif choice =='hemato':
        data = pd.read_csv('./data/hematoData.csv')
    elif choice =='signaling_like':
        data = pd.read_csv('./data/synthetic_data/synthetic_15_3/synthetic_15.csv')
    elif choice == 'nonlinear_gaussian':
        breakpoint()
        data = pd.read_csv('./data/nonlinear_gaussian_samples_12.csv')

    elif choice == 'sachs_observational':
       # graph = sachs_to_groundTruth(get_example_model('sachs'))
        filename = download(
            'https://www.bnlearn.com/book-crc/code/sachs.data.txt.gz',
            Path('data/sachs.data.txt')
        )
        data = pd.read_csv(filename, delimiter='\t', dtype=float)
        data = (data - data.mean()) / data.std()  # Standardize data
#     num_nodes = len(data.columns)
#     for ix in range(num_nodes):
#         for iy in range(num_nodes):
#             for iz in range(num_nodes):
#                 if check_no_redundancy(ix,iy,iz):
#                     if normalize:
#                         info_measures =get_normalized_information_theoretic_measures(ix, iy, iz,data)
#                         instance = {"nodes_index":f'{ix}_{iy}_{iz}', "MI":info_measures[0], "conMI":info_measures[1],"threepointInfo":info_measures[2]}
#                         pd_basic_stat =pd.concat([pd_basic_stat, pd.DataFrame([instance])], ignore_index=True)
#                     else:
#                         info_measures =get_information_theoretic_measures(ix, iy, iz,data)
#                         instance = {"nodes_index":f'{ix}_{iy}_{iz}', "MI":info_measures[0], "conMI":info_measures[1],"threepointInfo":info_measures[2]}
#                         pd_basic_stat =pd.concat([pd_basic_stat, pd.DataFrame([instance])], ignore_index=True)
#                 # struture = get_local_structure(ix,iy,iz)
# #                key = f'{ix}_{iy}_{iz}'
# #                basic_statistics[key]=", ".join([f"{i}" for i in info_measures])#.join(f'{struture}') #([f'{graph[ix]}-{graph[iy]}-{graph-iz}']=f'{i} for i in info_measures}'
#     pd_basic_stat.to_csv(f'normalized_{choice}_statistics.csv', index=False)

    entropies = [entropy(i,data) for i in range(len(data.columns))]
    print(entropies)
    np.save('entropies.npy',np.array(entropies))


def write_statistics(basic_statiscs):
    with open('basic_satistics_erdos_renyi','wb') as f:
        basic_statiscs.to_csv()
def get_local_structure(ix,iy,iz):
    pass
def global_investigation():
    '''
    investigate the potential of soft-info-constraint in reflecting global goodness of fit
    :return:
    '''
    pass

def get_information_theoretic_measures(col_ix,col_iy,col_iz,data):
    I_xy = mutual_info_discrete([col_ix,col_iy],data)
    I_xyGz = conditionalMutual_discrete([col_ix, col_iy, col_iz], data)
    I_xyz = I_xy - I_xyGz
    return [I_xy,I_xyGz,I_xyz]

def get_normalized_information_theoretic_measures(col_ix,col_iy,col_iz,data):
    breakpoint()
    n_I_xy = normalized_mutual_info_discrete([col_ix,col_iy],data)
    n_I_xyGz = normalized_conditionalMutual_discrete([col_ix,col_iy,col_iz],data)
    n_I_xyz  = normalized_threepoint_info_discreteF([col_ix,col_iy,col_iz],data)
    return [n_I_xy, n_I_xyGz, n_I_xyz]

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

def entropy(index, dataframe):
    return drv.entropy(dataframe.iloc[:,index])

def normalized_mutual_info_discrete(indexes,dataframe):
    MI = mutual_info_discrete(indexes,dataframe)
    return MI/(entropy(indexes[0], dataframe) + entropy(indexes[1], dataframe))
def normalized_conditionalMutual_discrete(indexes, dataframe):
    C_MI = conditionalMutual_discrete(indexes, dataframe)
    return C_MI/(entropy(indexes[0], dataframe) + entropy(indexes[1], dataframe))

def normalized_threepoint_info_discreteF(indexes,dataframe):
    threepointInfo = threepoint_info_discreteF(indexes, dataframe)
    return threepointInfo/ (entropy(indexes[0], dataframe) + entropy(indexes[1], dataframe))



def main():
    choice = ['erdos_renyi','signaling_like', 'sachs_observational', 'nonlinear_gaussian','hemato']
    normalize = True
    get_basic_statistics(choice[1], normalize)
  #  write_statistics(basic_statistics)

if __name__=="__main__":
    main()