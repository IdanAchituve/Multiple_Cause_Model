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


# calc q using mean field
def prob_q(prev_q_1, y):

    q_0 = np.zeros(k)  # q(x_i=0) for all x_i
    q_1 = np.zeros(k)  # q(x_i=1) for all x_i

    for i in range(k):
        prev_q_1_cpy = prev_q_1.copy()
        prev_q_1_cpy[i] = 0  # replace the i-th value with 0
        expr_0 = np.dot(A, prev_q_1_cpy) - y
        q_0[i] = np.exp((-1/2) * np.power(np.linalg.norm(expr_0), 2))

        prev_q_1_cpy = prev_q_1.copy()
        prev_q_1_cpy[i] = 1  # replace the i-th value with 1
        expr_1 = np.dot(A, prev_q_1_cpy) - y
        q_1[i] = np.exp((-1/2) * np.power(np.linalg.norm(expr_1), 2))

    return q_0/(q_0+q_1), q_1/(q_0+q_1)


# apply variational EM algorithm using mean field as approximation for p(x1,...,x6|y;theta) which is hard to estimate
# here we are only doing the E step because we have the true parameters of the M step
def variational_EM(y):

    # get all combinations
    combinations = list(map(list, itertools.product([0, 1], repeat=k)))
    p_x = np.prod(p_x_0)  # since all probabilities are 0.5 - for any x, p(x) = pi_over_i(p_x_i=0)

    q_0 = np.copy(p_x_0)  # initial q values are all 0.5
    q_1 = 1 - q_0
    likelihood = []
    delta = 1
    iter = 0
    while delta > 0:
        if iter > 0:
            q_0, q_1 = prob_q(q_1, y)

        l = 0
        # L(q) = sigma_over_x(q(x)log(f(x,y))) + H(q)
        for comb in combinations:
            x = np.asarray(comb)
            q = np.where(x == 1, q_1, q_0)  # take from q_1 values corresponding to x_i=1, otherwise take from q_0
            prod_q = np.prod(q)
            l += prod_q * np.log(prob_y_given_x(y, x) * p_x) - prod_q*np.log(prod_q)

        likelihood.append(l)
        delta = (likelihood[-1] - likelihood[-2]) if iter > 0 else 1
        iter += 1
        print("likelihood iteration " + str(iter) + ": " + str(l))

    return likelihood, q_0


if __name__ == '__main__':

    y = np.asarray([2.64, 2.56])
    p_y = prob_y(y)
    p_x_0_given_y = prob_x_i_given_y(y, p_y)
    p_x_1_given_y = 1 - p_x_0_given_y
    likelihood, q_0 = variational_EM(y)