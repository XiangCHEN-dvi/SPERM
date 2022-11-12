import numpy as np
import sperm
from sperm.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt

###################################################
# Preparation
###################################################
shape_prior_list = [
    None,
    '0:decreasing',
    ['0:Lipschitz:1'],
    ['0:increasing', '0:Lipschitz:2'],
]

X = np.array([0, 1, 2, 3, 4]).reshape([-1, 1])
y = np.array([1, 3.1, 5.5, 7.5, 9.9])

###################################################
# LinearRegression
###################################################
plt.figure(figsize=(16,16))
for idx, shape_prior in enumerate(shape_prior_list):
    reg = LinearRegression(shape_prior=shape_prior).fit(X, y)
    y_pred = reg.predict(X)
    plt.subplot(2, 2, idx+1)
    plt.plot(X, y, 'r+', label='truth')
    plt.plot(X, y_pred, 'k-', label='pred')
    plt.legend()
    plt.title(reg.shape_prior)
plt.tight_layout()
plt.savefig('LinearRegression.png')

###################################################
# Ridge
###################################################
plt.figure(figsize=(16,16))
for idx, shape_prior in enumerate(shape_prior_list):
    reg = Ridge(shape_prior=shape_prior).fit(X, y)
    y_pred = reg.predict(X)
    plt.subplot(2, 2, idx+1)
    plt.plot(X, y, 'r+', label='truth')
    plt.plot(X, y_pred, 'k-', label='pred')
    plt.legend()
    plt.title(reg.shape_prior)
plt.tight_layout()
plt.savefig('Ridge.png')
