"""Train the metapath encoder.

The 1000 ImageNet classes are the guide nodes: their generated embeddings are fit to the
ResNet-101 classifier weights (Eq. 5). The last 5% of the classes are held out to select
the checkpoint. Writes <out>/config.json, <out>/model.pt and <out>/log.json.

    python train.py --out runs/metapath_l3_c3
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F

from metapath.common import build_model, relation_graph
from metapath.data import load_graph, node_ids, read_lines, resnet101_classifier


def get_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--graph', default='data/graph_ilsvrc.pt')
    p.add_argument('--classes', default='data/ilsvrc_concepts.txt', help='ConceptNet nodes of the 1000 ImageNet classes')
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=0)
    # metapaths
    p.add_argument('--num_layers', type=int, default=3, help='metapath length')
    p.add_argument('--num_channels', type=int, default=3, help='number of learned metapaths')
    p.add_argument('--adjacency', default='fastgtn', choices=['fastgtn', 'strict'])
    p.add_argument('--featured_only', type=int, default=1, choices=[0, 1],
                   help='ignore neighbours without a word embedding (all-zero features)')
    p.add_argument('--use_edge_weights', action='store_true', help='weight edges by their random-walk weight')
    p.add_argument('--shuffle_relations', action='store_true', help='ablation: random relation labels')
    # projection and loss
    p.add_argument('--head', default='mlp', choices=['linear', 'mlp'])
    p.add_argument('--beta', type=float, default=0.5, help='weight of the node\'s own embedding')
    p.add_argument('--dropout', type=float, default=0.5, help='dropout of the mlp head')
    p.add_argument('--loss', default='contrastive', choices=['l2', 'contrastive'],
                   help='contrastive: rank each class closest to its own visual embedding; l2: Eq. 5')
    p.add_argument('--tau', type=float, default=0.05, help='temperature of the contrastive loss')
    # optimization
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--weight_decay', type=float, default=1e-3)
    p.add_argument('--batch', type=int, default=100, help='guide classes per step')
    p.add_argument('--val_fraction', type=float, default=0.05)
    p.add_argument('--max_epochs', type=int, default=200)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--select', default='mrr', choices=['mrr', 'l2'], help='checkpoint selection on held-out classes')
    return p


def l2_loss(a, b):
    return ((a - b) ** 2).sum() / (2 * len(a))


def val_metrics(output, targets, first):
    """L2 loss and retrieval rank of each held-out class (targets[first:]) among all classes."""
    true = torch.arange(first, len(targets), device=output.device)
    sims = output @ targets.T
    rank = (sims > sims[torch.arange(len(true)), true][:, None]).sum(dim=1) + 1
    return {'l2': l2_loss(output, targets[first:]).item(), 'hit1': (rank == 1).float().mean().item(),
            'hit5': (rank <= 5).float().mean().item(), 'mrr': (1.0 / rank).mean().item()}


def main():
    cfg = vars(get_parser().parse_args())
    torch.manual_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg['out'], exist_ok=True)
    with open(os.path.join(cfg['out'], 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)

    graph = load_graph(cfg['graph'])
    classes = node_ids(read_lines(cfg['classes']), graph['nodes']).to(device)
    targets = resnet101_classifier()[:len(classes)].to(device)
    n_train = round(len(classes) * (1 - cfg['val_fraction']))
    batches = [(s, min(s + cfg['batch'], n_train)) for s in range(0, n_train, cfg['batch'])]

    rel_graph = relation_graph(graph, cfg, device)
    features = graph['features'].to(device)
    model = build_model(cfg, rel_graph.num_relations).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    def loss_fn(s, e):
        output = model(rel_graph, features, classes[s:e])
        if cfg['loss'] == 'contrastive':
            logits = output @ targets[:n_train].T / cfg['tau']
            return F.cross_entropy(logits, torch.arange(s, e, device=device))
        return l2_loss(output, targets[s:e])

    log = {'train_loss': [], 'val_l2': [], 'val_hit1': [], 'val_hit5': [], 'val_mrr': []}
    best_score, best_epoch = -float('inf'), 0
    for epoch in range(1, cfg['max_epochs'] + 1):
        model.train()
        train_loss = 0
        for s, e in batches:
            loss = loss_fn(s, e)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / len(batches)
        model.eval()
        with torch.no_grad():
            val = val_metrics(model(rel_graph, features, classes[n_train:]), targets, n_train)
        log['train_loss'].append(train_loss)
        for k, v in val.items():
            log['val_' + k].append(v)
        print('epoch %d  train %.4f  val l2 %.4f  hit@1 %.3f  hit@5 %.3f  mrr %.4f' % (
            epoch, train_loss, val['l2'], val['hit1'], val['hit5'], val['mrr']), flush=True)

        score = val['mrr'] if cfg['select'] == 'mrr' else -val['l2']
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save(model.state_dict(), os.path.join(cfg['out'], 'model.pt'))
        elif epoch - best_epoch > cfg['patience']:
            break

    log['best_epoch'] = best_epoch
    with open(os.path.join(cfg['out'], 'log.json'), 'w') as f:
        json.dump(log, f)
    print('best epoch %d, saved %s' % (best_epoch, os.path.join(cfg['out'], 'model.pt')))


if __name__ == '__main__':
    main()
