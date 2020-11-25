from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import numpy as np

kwargs = {'batch_size': 32,
          'num_workers': 1,
          'pin_memory': True,
          'shuffle': False}


def mnist_transform():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])


def mnist_train():
    return datasets.MNIST('data', train=True, download=True, transform=mnist_transform())


def mnist_test():
    return datasets.MNIST('data', train=False, download=True, transform=mnist_transform())


def sorted_indices(dataset):
    labels = np.array([x[1] for x in dataset])
    return labels.argsort()


def trainset(size):
    dataset = mnist_train()
    subset = Subset(dataset, range(size))
    return DataLoader(subset, **kwargs)


def random_trainset(size):
    dataset = mnist_train()
    random_indices = np.random.choice(range(len(dataset)), size=size, replace=False)
    subset = Subset(dataset, random_indices)
    return DataLoader(subset, **kwargs)


def testset(size):
    dataset = mnist_test()
    subset = Subset(dataset, range(size))
    return DataLoader(subset, **kwargs)


def sorted_testset(size):
    dataset = mnist_test()
    subset = Subset(dataset, sorted_indices(dataset)[:size])
    return DataLoader(subset, **kwargs)


def randomized_testset(size):
    dataset = mnist_test()
    random_indices = np.random.choice(range(len(dataset)), size=size, replace=False)
    subset = Subset(dataset, random_indices)
    return DataLoader(subset, **kwargs)


def random_subsets(subsets, total_size):
    dataset = mnist_train()
    subset_sizes = np.full(subsets, total_size // subsets)
    subset_sizes[:total_size % subsets] += 1
    indices = np.random.choice(np.arange(len(dataset)), size=total_size, replace=False)
    indices = np.random.permutation(indices)
    subsets = []
    offset = 0
    for size in subset_sizes:
        subsets.append(Subset(dataset, list(indices[offset:offset + size])))
        offset += size
    return subsets


def random_trainsets(subsets, total_size):
    subsets = random_subsets(subsets, total_size)
    return [DataLoader(subset, **kwargs) for subset in subsets]


def get_targets(dataloader) -> np.ndarray:
    targets = np.empty((len(dataloader.dataset)))
    y = 0
    for (_, target) in dataloader:
        end = y + len(target)
        targets[y:end] = target
        y = end
    return targets


if __name__ == '__main__':
    np.random.seed(0)
    d = randomized_testset(10000)
    print(get_targets(d))
