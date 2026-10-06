"""Zero-shot object-agnostic state classification with a .pred file.

The .pred rows replace the classifier of an ImageNet-pretrained ResNet-101; each test image
is assigned the highest-scoring state among the states of its test set. Reports the mean
per-class top-1 accuracy (paper Sec. 4).

The test set is either
  - an HDF5 file with one dataset per state, images/<state>: (N, 3, 224, 224) float32,
    already resized to 224x224 and normalized with the ImageNet mean and std, or
  - an image folder <dir>/<state>/*.jpg (resized and normalized here the same way).

    python evaluate.py --pred runs/metapath/states.pred --test_set osdd --test data/test/osdd.h5
"""
import argparse

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet101

from metapath.data import STATES, TEST_SET_STATES

TRANSFORM = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def h5_batches(path, states, batch):
    """(images, labels) batches of an HDF5 test set."""
    import h5py
    with h5py.File(path, 'r') as f:
        for name in f['images']:
            images = f['images'][name]
            for s in range(0, len(images), batch):
                chunk = torch.from_numpy(images[s:s + batch]).float()
                yield chunk, torch.full((len(chunk),), states.index(name))


def folder_batches(path, states, batch):
    """(images, labels) batches of an image-folder test set."""
    data = datasets.ImageFolder(path, TRANSFORM)
    label_of_folder = torch.tensor([states.index(name) for name in data.classes])
    for images, folders in torch.utils.data.DataLoader(data, batch_size=batch, num_workers=4):
        yield images, label_of_folder[folders]


def evaluate(pred, test_set, test, batch=64, device='cpu'):
    states = TEST_SET_STATES[test_set]
    weights = pred[[STATES.index(s) for s in states]]
    model = resnet101(pretrained=True)
    model.fc = nn.Linear(weights.shape[1] - 1, len(states))
    with torch.no_grad():
        model.fc.weight.copy_(weights[:, :-1])
        model.fc.bias.copy_(weights[:, -1])
    model.to(device).eval()

    batches = h5_batches if test.endswith(('.h5', '.hdf5')) else folder_batches
    preds, labels = [], []
    with torch.no_grad():
        for images, label in batches(test, states, batch):
            preds.append(model(images.to(device)).argmax(dim=1).cpu())
            labels.append(label)
    preds, labels = torch.cat(preds), torch.cat(labels)
    per_class = {states[c]: (preds[labels == c] == c).float().mean().item() for c in labels.unique().tolist()}
    return sum(per_class.values()) / len(per_class), per_class, (preds == labels).float().mean().item()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pred', required=True)
    p.add_argument('--test_set', required=True, choices=list(TEST_SET_STATES))
    p.add_argument('--test', required=True, help='HDF5 file or image folder')
    p.add_argument('--batch', type=int, default=64)
    args = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mean, per_class, overall = evaluate(torch.load(args.pred, map_location='cpu'), args.test_set,
                                        args.test, args.batch, device)
    for state, acc in per_class.items():
        print('%-10s %6.2f' % (state, 100 * acc))
    print('top-1 accuracy             %6.2f' % (100 * overall))
    print('mean per-class accuracy    %6.2f' % (100 * mean))


if __name__ == '__main__':
    main()
