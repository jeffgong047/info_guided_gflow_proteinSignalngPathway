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
from dibs.inference import JointDiBS
from dibs.target import make_nonlinear_gaussian_model
import jax.random as random
import numpy as np
import jax.numpy as jnp



def generate_nonlinear_gaussian_model(n_vars=5):
    key = random.PRNGKey(0)
    # simulate some data
    key, subk = random.split(key)
    data, model = make_nonlinear_gaussian_model(key=subk, n_vars=n_vars)
    return data, model


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

def sachs_to_groundTruth(sachs_result):
    ground_truth = sachs_result
    ground_truth.add_edges_from([('PIP3', 'Akt'), ('Plcg', 'PKC'), ('PIP2', 'PKC')])
    ground_truth.remove_edge('Plcg', 'PIP3')
    ground_truth.add_edge('PIP3', 'Plcg')
    return ground_truth

def get_data(name, args, rng=default_rng()):
    if name == 'erdos_renyi_lingauss':
        graph = sample_erdos_renyi_linear_gaussian(
            num_variables=args.num_variables,
            num_edges=args.num_edges,
            loc_edges=0.0,
            scale_edges=1.0,
            obs_noise=0.1,
            rng=rng
        )
        data = sample_from_linear_gaussian(
            graph,
            num_samples=args.num_samples,
            rng=rng
        )
        data.to_csv('erdos_renyi_samples.csv', index=False)
        breakpoint()
        score = 'bge'
    elif name =="nonlinear_gaussian":
        num_variables = 12
        data_dibs, graph = generate_nonlinear_gaussian_model(num_variables)
        data = pd.DataFrame(data=np.array(data_dibs.x), columns=[i for i in range(data_dibs.x.shape[1])])
        data.to_csv('nonlinear_gaussian_samples_12.csv', index=False)
      #  data =pd.read_csv('./data/nonlinear_gaussian_samples_12.csv')
        breakpoint()
        score = 'bge'
    elif name == 'synthetic_signaling':
        graph = None
        path = os.path.join('./data' ,'synthetic_data')
        with open(path+f'synthetic_signaling_{args.nodes_num}_{args.data_instance_index}', 'wb') as f:
            data = pd.read_csv(f)
        score = 'bde'
    elif name == 'sachs_continuous':
        graph = sachs_to_groundTruth(get_example_model('sachs'))
        filename = download(
            'https://www.bnlearn.com/book-crc/code/sachs.data.txt.gz',
            Path('data/sachs.data.txt')
        )
        data = pd.read_csv(filename, delimiter='\t', dtype=float)
        data = (data - data.mean()) / data.std()  # Standardize data
        score = 'bge'

    elif name =='sachs_interventional':
        graph = sachs_to_groundTruth(get_example_model('sachs'))
        filename = download(
            'https://www.bnlearn.com/book-crc/code/sachs.interventional.txt.gz',
            Path('data/sachs.interventional.txt')
        )
        data = pd.read_csv(filename, delimiter=' ', dtype='category')
        score = 'bde'
    elif name =="sachs_allConditioned":
        graph = sachs_to_groundTruth(get_example_model('sachs'))
        data = pd.read_csv('data/intervention_all_noWesternBolts.csv' , dtype='category') #csv file contains all the intervention data
        score = 'bde'
    elif name=="hematopoietic":
        graph = sachs_to_groundTruth(get_example_model('sachs'))
        filename ='data/hematoData.csv'
        data = pd.read_csv(filename,delimiter=',', dtype='category')
        score= 'bde'
    elif name =="demo":
        edges = [('A', 'B'), ('C', 'B'), ('C', 'D')]
        graph = BayesianNetwork(edges)
        filename ='data/demo.csv'
        data = pd.read_csv(filename, delimiter=',', dtype='category')
        score ='bde'
    elif name=='demo_interven':
        edges = [('A', 'B'), ('C', 'B'), ('C', 'D')]
        graph = BayesianNetwork(edges)
        filename = 'data/demo_intervene.csv'
        data = pd.read_csv(filename, delimiter=',', dtype= 'category')
        score = 'bde'
    elif name =='do_structure':
        edges = [('Z', 'X'), ('X', 'Y'), ('Z', 'Y')]
        graph = BayesianNetwork(edges)
        filename = 'data/do_structure.csv'
        data = pd.read_csv(filename, delimiter=',', dtype= 'category')
        score = 'bde'
    elif name=='synthetic15_3_observational':
        nodes = [0, 1, 2, 13, 12, 14, 4, 7, 11, 3, 8, 5, 6, 10, 9]
        filename ='./data/synthetic_data/synthetic_15_3/synthetic_15.csv'
        edges = [
            (0, 1),
            (0, 2),
            (0, 13),
            (0, 12),
            (0, 14),
            (0, 4),
            (0, 7),
            (0, 11),
            (2, 3),
            (2, 8),
            (2, 14),
            (3, 4),
            (3, 5),
            (3, 6),
            (3, 7),
            (3, 11),
            (6, 10),
            (7, 9),
            (8, 12),
        ]

        # Create a BayesianModel with the nodes and edges
        graph = BayesianNetwork(edges)
        data = pd.read_csv(filename, delimiter=',', dtype='category')
        score = 'bde'
    else:
        raise ValueError(f'Unknown graph type: {name}')

    return graph, data, score
