"""Generate the visual embeddings of the state classes with a trained run (.pred file).

The .pred holds one 2049-d row per state (weights + bias of a ResNet-101 classifier), in
the order of metapath.data.STATES. --perturb applies a weight-assignment ablation
(paper Table 2) to the learned relation weights first.

    python predict.py --run runs/metapath_l3_c3
    python predict.py --run runs/metapath_l3_c3 --perturb random
"""
import argparse
import os

import torch

from metapath.common import load_run
from metapath.data import STATES, state_node_ids
from metapath.model import PERTURBATIONS


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', required=True)
    p.add_argument('--perturb', choices=list(PERTURBATIONS))
    p.add_argument('--seed', type=int, default=0, help='for --perturb random')
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg, graph, rel_graph, model = load_run(args.run, device)
    if args.perturb:
        PERTURBATIONS[args.perturb](model)
    rows = state_node_ids(graph).to(device)
    with torch.no_grad():
        pred = model(rel_graph, graph['features'].to(device), rows).cpu()

    path = os.path.join(args.run, 'states%s.pred' % ('_' + args.perturb if args.perturb else ''))
    torch.save(pred, path)
    print('nodes:', ', '.join(graph['nodes'][i] for i in rows.tolist()))
    print('saved %s (%d states: %s)' % (path, len(STATES), ', '.join(STATES)))


if __name__ == '__main__':
    main()
