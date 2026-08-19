# Author: Xiang CHEN <xiangchen.ai@outlook.com>
# License: Apache Software License

import matplotlib.pyplot as plt
import numpy as np

from sperm.linear_model import LinearRegression, Ridge
from sperm.priors import Decreasing, Increasing, Lipschitz

###################################################
# Preparation
###################################################
prior_sets = [
    None,
    {0: Decreasing()},
    {0: Lipschitz(1)},
    {0: (Increasing(), Lipschitz(2))},
]

X = np.array([0, 1, 2, 3, 4]).reshape([-1, 1])
y = np.array([1, 3.1, 5.5, 7.5, 9.9])

###################################################
# LinearRegression
###################################################
plt.figure(figsize=(16,16))
for idx, priors in enumerate(prior_sets):
    reg = LinearRegression(priors=priors).fit(X, y)
    y_pred = reg.predict(X)
    plt.subplot(2, 2, idx+1)
    plt.plot(X, y, 'r+', label='truth')
    plt.plot(X, y_pred, 'k-', label='pred')
    plt.legend()
    plt.title(reg.priors)
plt.tight_layout()
plt.savefig('LinearRegression.png')

###################################################
# Ridge
###################################################
plt.figure(figsize=(16,16))
for idx, priors in enumerate(prior_sets):
    reg = Ridge(priors=priors).fit(X, y)
    y_pred = reg.predict(X)
    plt.subplot(2, 2, idx+1)
    plt.plot(X, y, 'r+', label='truth')
    plt.plot(X, y_pred, 'k-', label='pred')
    plt.legend()
    plt.title(reg.priors)
plt.tight_layout()
plt.savefig('Ridge.png')
