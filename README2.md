# Metapath-driven Embeddings for Zero-Shot Object State Classification

Code for the ICPR 2026 paper *Metapath-driven Embeddings for Zero-Shot Object State
Classification* (Gouidis, Papoutsakis, Patkos, Argyros, Plexousakis).

A Graph Transformer Network learns soft metapaths, i.e. sequences of relation types,
over a ConceptNet knowledge graph and uses them to project word embeddings of graph
nodes into the visual space of a ResNet-101. Trained on the 1000 ImageNet classes as
guide nodes, it then generates classifiers for unseen object states (open, closed,
empty, ...) without any image of them.

## Setup

```bash
# first install the PyTorch build matching your CUDA driver (https://pytorch.org/get-started/locally/),
# e.g. pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt     # PyTorch >= 1.13 and torchvision; no graph libraries needed
bash scripts/download_data.sh       # graph, ImageNet class nodes and test sets into data/
```

`data/graph_ilsvrc.pt` holds the ConceptNet graph around the ImageNet classes and the
state words: 574,270 nodes, 50 relation types, 2.34M edges with random-walk weights,
and 300-d GloVe node features. It was created from the ZSL-KG graph with
`scripts/export_graph.py`.

The data files are on Zenodo ([10.5281/zenodo.23187515](https://doi.org/10.5281/zenodo.23187515)).
Each test set is an image folder `data/test/<test set>/<state>/*.jpg` (images are resized to
224x224 and normalized with the ImageNet mean and std in `evaluate.py`; an HDF5 file with one
dataset per state, `images/<state>`: `(N, 3, 224, 224)` float32, already normalized, works too).
Test sets:

| test set | source | images | states |
|---|---|---|---|
| osdd | OSDD (Gouidis et al., VISAPP 2022) | 3,109 | closed, containing, empty, filled, folded, open, plugged, unfolded, unplugged |
| cgqa | C-GQA (Mancini et al., TPAMI 2022) | 497 | closed, empty, filled, folded, open |
| mit | MIT-States (Isola et al., CVPR 2015) | 354 | closed, empty, filled, folded, open |
| vaw_states | VAW (Pham et al., CVPR 2021) | 1,584 | the 9 OSDD states |

## Usage

```bash
# 1. learn the metapaths and the projection on the ImageNet guide classes
python train.py --out runs/metapath_l3_c3

# 2. generate the classifiers of the 9 states (runs/metapath_l3_c3/states.pred)
python predict.py --run runs/metapath_l3_c3

# 3. zero-shot state classification: mean per-class top-1 accuracy
python evaluate.py --pred runs/metapath_l3_c3/states.pred --test_set osdd --test data/test/osdd
```

Default configuration: metapath length 3 (`--num_layers`), 3 channels (`--num_channels`),
an MLP projection (300 -> 1174 -> 2049) trained with a contrastive loss that ranks each
guide class closest to its own visual embedding (`--head mlp --loss contrastive --tau 0.05`),
learning rate 0.001, and neighbours without a word embedding ignored (`--featured_only 1`).
The linear projection with the L2 loss of Eq. 5 is `--head linear --loss l2 --lr 0.01`.

Each state is the base ConceptNet node of its word (`/c/en/open`, not a sense node such
as `/c/en/open/n`); see `metapath.data.state_node_ids`.

**Ablations** (paper Sec. 4.1):

```bash
python train.py --out runs/l1 --num_layers 1                  # metapath length
python train.py --out runs/c1 --num_channels 1                # number of softmax channels
python train.py --out runs/lr --lr 0.01                       # learning rate
python predict.py --run runs/metapath_l3_c3 --perturb random  # weight assignment: random,
                                                              # uniform, invert_sign, invert_magn
python train.py --out runs/shuffled --shuffle_relations       # random relation labels
python train.py --out runs/nograph --num_layers 0 --num_channels 1   # node's own embedding only
```

Further options: `--adjacency strict` (a relation a node lacks contributes nothing,
instead of acting as the identity), `--use_edge_weights` (random-walk edge weights),
`--beta`. See `python train.py --help`.

**Learned metapaths** (paper Sec. 4.3, Table 5):

```bash
python analyze_metapaths.py --run runs/metapath_l3_c3 --top 5
```

## Code

| file | |
|---|---|
| `metapath/graph.py` | relation adjacencies A_t and the sparse metapath operator sum_t alpha_t A_t |
| `metapath/model.py` | the metapath encoder and the weight-assignment ablations |
| `metapath/data.py` | graph loading, ImageNet class and state nodes, ResNet-101 targets |
| `train.py`, `predict.py`, `evaluate.py`, `analyze_metapaths.py` | the scripts above |

Training uses the whole graph on one GPU (about 8 GB for length 3 and 3 channels):
features are propagated in the 300-d input space, only the guide nodes are projected,
and the gradient of the relation weights is computed at the graph's nonzeros only.

## Citation

```bibtex
@inproceedings{gouidis2026metapath,
  title     = {Metapath-driven Embeddings for Zero-Shot Object State Classification},
  author    = {Gouidis, Filippos and Papoutsakis, Konstantinos and Patkos, Theodore and
               Argyros, Antonis and Plexousakis, Dimitris},
  booktitle = {International Conference on Pattern Recognition (ICPR)},
  year      = {2026}
}
```
