"""Export a ZSL-KG ConceptNet graph (graph_data.pt) to the graph package used here.

Only needed to (re)create the released file. graph_data.pt holds, for every node n, a
random-walk adjacency list rw_adj_lists[n] = [(neighbour, relation, weight), ...]; each
entry becomes an edge target=neighbour <- source=n of that relation.

    python scripts/export_graph.py --graph_data .../ilsvrc/graph_data.pt --out data/graph_ilsvrc.pt
"""
import argparse

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--graph_data', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    g = torch.load(args.graph_data)
    num_rel = len(g['relations'])
    target, source, relation, weight = [], [], [], []
    for node, triplets in g['rw_adj_lists'].items():
        for neighbour, rel, w in triplets:
            if rel < num_rel:
                target.append(neighbour)
                source.append(node)
                relation.append(rel)
                weight.append(w)
    package = {
        'nodes': list(g['nodes']),
        'relations': list(g['relations']),
        'features': g['features'].float(),
        'edges': torch.tensor([target, source, relation], dtype=torch.long),
        'weights': torch.tensor(weight, dtype=torch.float),
    }
    torch.save(package, args.out)
    print('%d nodes, %d relations, %d edges -> %s' % (
        len(package['nodes']), num_rel, package['edges'].shape[1], args.out))


if __name__ == '__main__':
    main()
