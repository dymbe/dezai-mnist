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


def plot_bars(ys, bars_labels, image, ground_truth):
    if len(ys.shape) < 2:
        ys = ys.reshape(1, -1)

    x = np.arange(ys.shape[1])

    fig, axarr = plt.subplots(1, 2)
    total_width = 0.8
    single_width = total_width / len(ys)

    for i, y in enumerate(ys):
        x_off = x - total_width / 2 + i * single_width
        axarr[0].bar(x_off, y, width=single_width, label=labels[i])

    axarr[0].set_xticks(x)
    axarr[0].set_ylim(0, 1)
    axarr[0].set_yticks(np.linspace(0, 1, 11))
    #axarr[0].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axarr[0].set_xlabel("Label")
    axarr[0].set_ylabel("Probability")
    axarr[0].legend()
    axarr[1].imshow(image, cmap='gray')
    axarr[1].axis("off")
    fig.suptitle(f"Ground truth = {ground_truth}")

    plt.show()


if __name__ == '__main__':
    from datasets import testset
    import mnistnn
    from utils import mv, average_models

    project = "m375-ts12000-e5-lr1.0-init-v0"

    outputs = np.load(f"test_results/{project}/outputs.npy")
    targets = np.load(f"test_results/{project}/targets.npy")

    models = mnistnn.load_models(project)
    fed_model = average_models(models)

    ds = testset(10000)

    fed_outputs = fed_model.outputs(ds)
    #fed_predictions = fed_outputs.argmax(axis=1)

    #print("np.mean(fed_predictions == targets) =", np.mean(fed_predictions == targets))

    mean_vote = outputs.mean(axis=1)
    mean_vote_predictions = mean_vote.argmax(axis=1)

    predictions = outputs.argmax(axis=2)
    scores = np.mean(predictions == targets.reshape(-1, 1), axis=0)

    print("np.mean(scores) =", np.mean(scores))
    print("np.std(scores) =", np.std(scores))

    print(np.mean(mv(outputs) == targets))

    labels = ["Aggregated model output",
              "All models mean outputs"]

    for i, (img_tensor, img_label) in enumerate(ds.dataset):
        votes = np.bincount(predictions[i], minlength=10)
        vote_shares = votes / np.sum(votes)
        my_image = img_tensor.squeeze().numpy()
        plot_bars(np.vstack((fed_outputs[i], mean_vote[i])), labels, my_image, img_label)
        input(f"Image {i} - Press any key to continue")
