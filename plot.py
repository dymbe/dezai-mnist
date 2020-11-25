import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


class Line:
    def __init__(self, ys, xs=None, yerr=None, label=None):
        self.ys = ys
        if xs is None:
            self.xs = np.arange(len(ys))
        else:
            self.xs = xs
        self.yerr = yerr
        if label is None:
            self.label = ""
        else:
            self.label = label

    def plot(self, color):
        if self.yerr is None:
            plt.plot(self.xs, self.ys, c=color, label=self.label)
        else:
            plt.errorbar(self.xs, self.ys, yerr=self.yerr, ecolor="lightblue", c="blue")


def distinct_colors(n, name='hsv'):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def plot_lines(lines):
    colors = distinct_colors(len(lines))

    for line, color in zip(lines, colors):
        line.plot(color)

    plt.legend()
    plt.title("")
    plt.ylim(0, 100)
    plt.xlabel("Test-examples")
    plt.ylabel("Score (%)")
    plt.show()


def plot_vote(votes, image, label):
    x = np.arange(len(votes))
    fig, axarr = plt.subplots(1, 2)
    axarr[0].bar(x, votes)
    axarr[0].set_xticks(x)
    axarr[0].set_ylim(0, 1)
    axarr[0].set_yticks(np.linspace(0, 1, 11))
    axarr[0].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axarr[0].set_xlabel("Digit")
    axarr[0].set_ylabel("Share of votes")
    axarr[1].imshow(image, cmap='gray')
    axarr[1].axis("off")
    fig.suptitle(f"Ground truth = {label}")

    plt.show()


if __name__ == '__main__':
    from datasets import testset
    from utils import mv

    project = "m375-ts12000-e5"

    ds = testset(10000)
    outputs = np.load(f"test_results/{project}/outputs.npy")
    targets = np.load(f"test_results/{project}/targets.npy")

    print(np.mean(mv(outputs) == targets))

    for i, (img_tensor, img_label) in enumerate(ds.dataset):
        preds = outputs[i].argmax(axis=1)
        counts = np.bincount(preds)
        counts = counts / np.sum(counts)
        my_image = img_tensor.squeeze().numpy()
        plot_vote(counts, my_image, img_label)
        input(f"Image {i} - Press any key to continue")
