import copy
from datasets import random_trainsets, testset
from mnistnn import MnistNN, load_models
from utils import average_models


def fl():
    trainsets = random_trainsets(375, 12000)
    federated_model = MnistNN()
    step = 375

    for _ in range(5):
        for i in range(0, len(trainsets), step):
            models = [copy.deepcopy(federated_model) for _ in range(step)]
            for j, model in enumerate(models):
                print(j)
                model.train(trainsets[i + j], epochs=5)
            federated_model = average_models(models)
            print(i)
            print(federated_model.test_score(testset()))


if __name__ == '__main__':
    models = load_models("m375-ts12000-e5-lr1.0-init")
    federated_model = average_models(models)
    print(federated_model.test_score(testset()))
