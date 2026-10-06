"""Building the graph and the model from a run configuration (shared by the scripts)."""
import json
import os

import torch

from metapath.data import load_graph
from metapath.graph import RelationGraph, build_adjacency
from metapath.model import MetapathEncoder


def relation_graph(graph, cfg, device):
    A = build_adjacency(graph, use_edge_weights=cfg['use_edge_weights'], adjacency=cfg['adjacency'],
                        featured_only=cfg['featured_only'],
                        shuffle_seed=cfg['seed'] if cfg['shuffle_relations'] else None)
    return RelationGraph(A, graph['features'].shape[0]).to(device)


def build_model(cfg, num_relations):
    return MetapathEncoder(num_relations, num_layers=cfg['num_layers'], num_channels=cfg['num_channels'],
                           head=cfg['head'], beta=cfg['beta'], dropout=cfg['dropout'])


def load_run(run_dir, device):
    """(config, graph, relation graph, model with the trained weights) of a training run."""
    with open(os.path.join(run_dir, 'config.json')) as f:
        cfg = json.load(f)
    graph = load_graph(cfg['graph'])
    rel_graph = relation_graph(graph, cfg, device)
    model = build_model(cfg, rel_graph.num_relations).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, 'model.pt'), map_location=device))
    model.eval()
    return cfg, graph, rel_graph, model
