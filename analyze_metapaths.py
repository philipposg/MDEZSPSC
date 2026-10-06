"""Rank the learned metapaths (paper Sec. 4.3, Table 5).

The weight of metapath t_1 -> ... -> t_l in a channel is alpha^(1)[t_1] * ... * alpha^(l)[t_l];
channels are averaged, as their embeddings are.

    python analyze_metapaths.py --run runs/metapath_l3_c3 --top 5
"""
import argparse

import numpy as np
import torch

from metapath.common import load_run


def metapath_scores(model):
    alphas = model.relation_weights()  # each (channels x relations)
    scores = alphas[0]
    for a in alphas[1:]:
        scores = scores.unsqueeze(-1) * a.view(a.shape[0], *([1] * (scores.dim() - 1)), a.shape[1])
    return scores.mean(dim=0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', required=True)
    p.add_argument('--top', type=int, default=5)
    args = p.parse_args()
    _, graph, _, model = load_run(args.run, torch.device('cpu'))
    relations = [r.split('/')[-1] for r in graph['relations']] + ['Identity']
    with torch.no_grad():
        scores = metapath_scores(model)
    flat = scores.flatten()
    order = flat.argsort(descending=True)
    for title, idx in (('Top', order[:args.top]), ('Bottom', order.flip(0)[:args.top])):
        print('\n%s-%d metapaths of length %d:' % (title, args.top, scores.dim()))
        for rank, i in enumerate(idx, 1):
            path = np.unravel_index(int(i), scores.shape)
            print('%d  %s  %.3e' % (rank, ' -> '.join(relations[int(t)] for t in path), flat[i].item()))


if __name__ == '__main__':
    main()
