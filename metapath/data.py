"""Graph package, concept lists and state classes."""
import os

import torch
import torch.nn.functional as F

# The 9 object states, in the class order of the evaluation (and of the .pred rows).
STATES = ['closed', 'containing', 'empty', 'filled', 'folded', 'open', 'plugged', 'unfolded', 'unplugged']
# ConceptNet word of each state, where it differs from the state name.
STATE_WORDS = {'unfolded': 'unfold'}
# States present in each test set (the classifier only scores these).
TEST_SET_STATES = {
    'osdd': STATES,
    'vaw_states': STATES,
    'cgqa': ['closed', 'empty', 'filled', 'folded', 'open'],
    'mit': ['closed', 'empty', 'filled', 'folded', 'open'],
}


def load_graph(path):
    """dict with nodes, relations, features (N x 300 GloVe), edges (3 x E: target, source,
    relation; the target aggregates the source) and weights (E random-walk weights)."""
    return torch.load(path)


def read_lines(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def node_ids(names, nodes):
    """Node ids of full ConceptNet node names (e.g. /c/en/tench)."""
    index = {node: i for i, node in enumerate(nodes)}
    missing = [n for n in names if n not in index]
    if missing:
        raise KeyError(f"not in the graph: {missing[:10]}")
    return torch.tensor([index[n] for n in names])


def state_node_ids(graph, states=STATES):
    """Node of each state word: its base node /c/en/<word>, or if that does not exist the
    sense node /c/en/<word>/... with the most edges."""
    words = [STATE_WORDS.get(s, s) for s in states]
    degree = torch.bincount(graph['edges'][0], minlength=len(graph['nodes']))
    best = {}
    for i, node in enumerate(graph['nodes']):
        parts = node.split('/')
        if len(parts) < 4 or parts[3] not in words:
            continue
        score = float('inf') if node == '/c/en/' + parts[3] else degree[i].item()
        if parts[3] not in best or score > best[parts[3]][1]:
            best[parts[3]] = (i, score)
    missing = [w for w in words if w not in best]
    if missing:
        raise KeyError(f"states not in the graph: {missing}")
    return torch.tensor([best[w][0] for w in words])


def resnet101_classifier():
    """Ground-truth visual embeddings: ResNet-101 fc weights and bias, L2-normalized (1000 x 2049)."""
    from torchvision.models import resnet101
    fc = resnet101(pretrained=True).fc
    with torch.no_grad():
        return F.normalize(torch.cat([fc.weight, fc.bias.unsqueeze(1)], dim=1))


def data_path(data_dir, *parts):
    return os.path.join(data_dir, *parts)
