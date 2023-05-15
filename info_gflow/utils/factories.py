from numpy.random import default_rng

from dag_gflownet.scores import BDeScore, BGeScore, priors,threepoint_info_score
from dag_gflownet.utils.data import get_data


def get_prior(name, **kwargs):
    prior = {
        'uniform': priors.UniformPrior,
        'erdos_renyi': priors.ErdosRenyiPrior,
        'edge': priors.EdgePrior,
        'fair': priors.FairPrior
    }
    return prior[name](**kwargs)


def get_scorers(args, rng=default_rng()):
    # Get the data
    graph, data, score = get_data(args.graph, args, rng=rng)

    # Get the prior
    prior = get_prior(args.prior, **args.prior_kwargs)

    # Get the scorer
    scores = {'bde': BDeScore, 'bge': BGeScore}
    bayesian_scorer = scores[score](data, prior, **args.scorer_kwargs)
    if args.info_constraint:
        info_constraint_score = threepoint_info_score(data,prior)
    return {'scorer':bayesian_scorer,'info_constraint_scorer':info_constraint_score}, data, graph
