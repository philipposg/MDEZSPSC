"""Relation-typed adjacency of the knowledge graph and the sparse metapath operator.

A_t (one per relation type t, plus the identity) are row-normalized adjacencies. A
metapath step with relation weights alpha is Q = sum_t alpha_t A_t; all A_t are merged
into one sparsity pattern so that Q is a single sparse matrix per step.
"""
import warnings

import torch

warnings.filterwarnings('ignore', message='.*Sparse CSR tensor support is in beta.*')


def _coalesce(row, col, value, num_nodes):
    """Sum duplicate (row, col) entries."""
    keys, inverse = torch.unique(row * num_nodes + col, return_inverse=True)
    summed = torch.zeros(len(keys), dtype=value.dtype).index_add_(0, inverse, value)
    return keys // num_nodes, keys % num_nodes, summed


def build_adjacency(graph, use_edge_weights=False, adjacency='fastgtn', featured_only=False, shuffle_seed=None):
    """One adjacency (edge_index, value) per relation, plus the identity as the last entry.

    graph: dict with 'edges' (3 x E: node, neighbour, relation), 'weights' (E random-walk
        weights), 'relations', 'features'. Node i aggregates the features of its neighbours.
    use_edge_weights: weight edges by their random-walk weight instead of 1.
    adjacency 'fastgtn': every relation also gets self-loops of weight 1e-20, so a node with
        no edge of that type keeps its own features (the row becomes the identity).
    adjacency 'strict': a node with no edge of that type gets nothing from that relation.
    featured_only: ignore neighbours without a word embedding (all-zero features).
    shuffle_seed: ablation, randomly reassign relation labels to edges (sizes kept).
    """
    num_nodes = graph['features'].shape[0]
    num_rel = len(graph['relations'])
    node, neighbour, rel = graph['edges']
    weight = graph['weights'].float() if use_edge_weights else torch.ones(node.shape[0])

    if featured_only:
        keep = graph['features'].norm(dim=1)[neighbour] > 0
        node, neighbour, rel, weight = node[keep], neighbour[keep], rel[keep], weight[keep]
    if shuffle_seed is not None:
        rel = rel[torch.randperm(len(rel), generator=torch.Generator().manual_seed(shuffle_seed))]

    A = []
    eye = torch.arange(num_nodes)
    for t in range(num_rel):
        mask = rel == t
        row, col, value = _coalesce(node[mask], neighbour[mask], weight[mask], num_nodes)
        keep = value != 0
        row, col, value = row[keep], col[keep], value[keep]
        if adjacency == 'fastgtn':
            row, col = torch.cat([row, eye]), torch.cat([col, eye])
            value = torch.cat([value, torch.full((num_nodes,), 1e-20)])
        degree = torch.zeros(num_nodes).index_add_(0, row, value)
        inv = degree.pow(-1)
        inv[torch.isinf(inv)] = 0
        A.append((torch.stack([row, col]), inv[row] * value))
    A.append((torch.stack([eye, eye]), torch.ones(num_nodes)))
    return A


def _crow(sorted_rows, num_rows):
    counts = torch.bincount(sorted_rows, minlength=num_rows)
    return torch.cat([counts.new_zeros(1), counts.cumsum(0)])


class RelationGraph:
    """The relation adjacencies merged into one CSR sparsity pattern.

    weighted(alpha) gives the values of sum_t alpha_t A_t on that pattern.
    """

    def __init__(self, A, num_nodes):
        index = torch.cat([e for e, _ in A], dim=1)
        value = torch.cat([v for _, v in A])
        rel = torch.cat([torch.full((e.shape[1],), t, dtype=torch.long) for t, (e, _) in enumerate(A)])
        keys, pos = torch.unique(index[0] * num_nodes + index[1], return_inverse=True)
        # (nonzeros x relations): weighted values = mix @ alpha
        self.mix = torch.sparse_coo_tensor(torch.stack([pos, rel]), value, (len(keys), len(A))).coalesce()
        self.row, self.col = keys // num_nodes, keys % num_nodes
        self.crow = _crow(self.row, num_nodes)
        self.t_perm = torch.argsort(self.col * num_nodes + self.row)  # transpose, for backward
        self.t_col = self.row[self.t_perm]
        self.t_crow = _crow(self.col[self.t_perm], num_nodes)
        self.num_nodes = num_nodes
        self.num_relations = len(A)
        self.shape = (num_nodes, num_nodes)

    def to(self, device):
        for name, value in vars(self).items():
            if torch.is_tensor(value):
                setattr(self, name, value.to(device))
        return self

    def weighted(self, alpha):
        return torch.sparse.mm(self.mix, alpha.unsqueeze(1)).squeeze(1)


class WeightedSpMM(torch.autograd.Function):
    """Q @ H for Q given by its values on a RelationGraph's pattern, differentiable in both.

    torch.sparse.mm's backward w.r.t. the sparse values materializes a dense N x N matrix
    (1.3 TB for this graph); here that gradient is computed at the nonzeros only.
    """
    CHUNK = 1 << 18

    @staticmethod
    def forward(ctx, value, H, graph):
        ctx.save_for_backward(value, H)
        ctx.graph = graph
        return torch.sparse_csr_tensor(graph.crow, graph.col, value, graph.shape) @ H

    @staticmethod
    def backward(ctx, grad):
        value, H = ctx.saved_tensors
        g = ctx.graph
        grad_value = grad_H = None
        if ctx.needs_input_grad[0]:
            grad_value = torch.empty_like(value)
            for s in range(0, value.numel(), WeightedSpMM.CHUNK):
                rows, cols = g.row[s:s + WeightedSpMM.CHUNK], g.col[s:s + WeightedSpMM.CHUNK]
                grad_value[s:s + WeightedSpMM.CHUNK] = (grad[rows] * H[cols]).sum(dim=1)
        if ctx.needs_input_grad[1]:
            grad_H = torch.sparse_csr_tensor(g.t_crow, g.t_col, value[g.t_perm], g.shape[::-1]) @ grad
        return grad_value, grad_H, None
