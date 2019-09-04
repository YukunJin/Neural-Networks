import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))
def cost_quadratic(z,answer):
    return ((answer-z)**2).mean()
def dquadratic_dz(z,answer):
    return -2*(answer-z)
def cost_cross_entropy(z,answer):
    return -answer @ np.log(z) - (1-answer)@np.log(1-z)
def dcross_dz(z,answer):
    return (1-answer)/(1-z)-answer/z
