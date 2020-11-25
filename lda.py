import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from datasets import random_trainset, sorted_testset

np.random.seed(0)

x_train, y_train = zip(*list(random_trainset(32)))
x_train = np.array([x.numpy() for x in x_train])
x_train = x_train.squeeze().reshape(x_train.shape[0], -1)

y_train = np.array(y_train)

lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)

x_test, y_test = zip(*list(sorted_testset(10000)))
x_test = np.array([x.numpy() for x in x_test])
x_test = x_test.squeeze().reshape(x_test.shape[0], -1)
y_test = np.array(y_test)
