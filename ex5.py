import numpy as np
import matplotlib.pyplot as plt
import itertools


A = np.asarray([[-0.3, -1.0, 1.3, 0.5, -0.2, 0.5],
                [-1.0, -2.5, 2.3, 0.8, 1.5, -0.5]])
SIGMA = np.asarray([[1, 0],
                    [0, 1]])

p_x_0 = np.asarray([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # p(x_i=0) = 0.5 for all i
p_x_1 = np.asarray([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])  # p(x_i=1) = 0.5 for all i
k = 6  # number of causes (hidden variables)


# p(y|x1,...,x6) - distributed according to N(Ax,I)
def prob_y_given_x(y, x):
    mu = np.dot(A, x)
    exp_val = np.dot(np.dot(np.transpose(y-mu), np.linalg.inv(SIGMA)), (y-mu))
    density = np.exp((-1/2)*exp_val)/(np.sqrt(np.linalg.det(SIGMA) * np.power(2 * np.pi, 2)))
    return density


# compute p(y)
def prob_y(y):
    # get all combinations
    combinations = list(map(list, itertools.product([0, 1], repeat=k)))
    p_y = 0
    p_x = np.prod(p_x_0)  # since all probabilities are 0.5 - for any x, p(x) = pi_over_i(p_x_i=0)
    for comb in combinations:
        x = np.asarray(comb)
        p_y += prob_y_given_x(y, x) * p_x  # p(y) = sigma_over_x(p(y|x)*p(x))

    return p_y


# compute p(x_i|y) for all x_i
def prob_x_i_given_y(y, p_y):

    # get all combinations
    p_x = np.prod(p_x_0)  # since all probabilities are 0.5 - for any x, p(x) = pi_over_i(p_x_i=0)
    combinations = list(map(list, itertools.product([0, 1], repeat=k - 1)))  # get all combinations of 5 random variables

    p_x_0_given_y = np.asarray(np.zeros(k))  # save p(x_i=0|y) for all i
    for i in range(k):
        p_xi_0_given_y = 0  # p(x_i=0|y)
        for comb in combinations:
            c = list(comb)
            c.insert(i, 1)  # insert x_i=0 at the relevant position
            x = np.asarray(c)
            p_xi_0_given_y += prob_y_given_x(y, x) * p_x  # p(x_i=0|y) = sigma_over_x(p(y|x)*p(x))/p(y)
        p_x_0_given_y[i] = p_xi_0_given_y/p_y

    return p_x_0_given_y


if __name__ == '__main__':

    y = np.asarray([2.64, 2.56])
    p_y = prob_y(y)
    p_x_0_given_y = prob_x_i_given_y(y, p_y)
    