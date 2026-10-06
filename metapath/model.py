"""Metapath encoder (paper Sec. 3.2).

For each of C channels (learned metapaths) and each step k of the metapath:
    Q^(k)_c = sum_t softmax(w^(k)_c)_t A_t            relation weights, Eq. 2
    X_P,c   = Q^(l)_c ... Q^(1)_c X_A                 metapath aggregation, Eq. 3-4a
    E_c     = LeakyReLU(f_c(beta X_A + (1 - beta) X_P,c))   projection, Eq. 4b
and E = normalize(mean_c E_c). Propagation stays in the 300-d input space and only the
requested rows are projected, so the whole graph fits on one GPU.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from metapath.graph import WeightedSpMM


class MetapathEncoder(nn.Module):
    def __init__(self, num_relations, num_layers=3, num_channels=3, input_dim=300, output_dim=2049,
                 head='linear', beta=0.5, dropout=0.5):
        """num_relations includes the identity; num_layers is the metapath length."""
        super().__init__()
        self.num_channels = num_channels
        self.beta = beta
        # one (channels x relations) weight matrix per metapath step
        self.metapath_logits = nn.ParameterList(
            [nn.Parameter(0.1 * torch.randn(num_channels, num_relations)) for _ in range(num_layers)]
        )
        if head == 'linear':
            self.heads = nn.ModuleList([nn.Linear(input_dim, output_dim, bias=False) for _ in range(num_channels)])
            for h in self.heads:
                nn.init.xavier_uniform_(h.weight)
        elif head == 'mlp':
            hidden = (input_dim + output_dim) // 2
            self.heads = nn.ModuleList([
                nn.Sequential(nn.Dropout(dropout), nn.Linear(input_dim, hidden), nn.LeakyReLU(0.2),
                              nn.Dropout(dropout), nn.Linear(hidden, output_dim))
                for _ in range(num_channels)
            ])
        else:
            raise ValueError(head)

    def relation_weights(self):
        """Softmax relation weights alpha, one (channels x relations) tensor per step."""
        return [F.softmax(w, dim=1) for w in self.metapath_logits]

    def forward(self, graph, X, rows):
        """Visual embeddings (len(rows) x output_dim) of the nodes `rows`."""
        alphas = self.relation_weights()
        out = []
        for c in range(self.num_channels):
            H = X
            for alpha in alphas:
                H = WeightedSpMM.apply(graph.weighted(alpha[c]), H, graph)
            Z = self.beta * X[rows] + (1 - self.beta) * H[rows]
            out.append(F.leaky_relu(self.heads[c](Z), 0.2))
        return F.normalize(torch.stack(out).mean(dim=0))


# Weight-assignment ablations (paper Sec. 4.1, Table 2), applied to the learned logits.
def assign_random(model):
    for w in model.metapath_logits:
        nn.init.uniform_(w, -1.0, 1.0)


def assign_uniform(model):
    for w in model.metapath_logits:
        nn.init.ones_(w)


def invert_sign(model):
    for w in model.metapath_logits:
        w.data = -w.data


def invert_magnitude(model):
    for w in model.metapath_logits:
        w.data = torch.where(w.data != 0, 1 / w.data, w.data)


PERTURBATIONS = {'random': assign_random, 'uniform': assign_uniform,
                 'invert_sign': invert_sign, 'invert_magn': invert_magnitude}
