import numpy as np
import sys
class Layers:
    def __init__(self,input_size,output_size,layer_id):
        self.Weights = np.random.randn(input_size,output_size)
        self.bias = np.random.randn(output_size)
        self.id = layer_id
        self.input_size = input_size
        self.output_size = output_size
        self.x = np.zeros(input_size)
        self.y = np.zeros(output_size)
        self.z = np.zeros(output_size)

    def setWeight(self,Weights):
        self.Weights = Weights

    def setBias(self,bias):
        self.bias = bias

    def evaluate(self,input_vec,f):
        self.x = input_vec.copy()
        self.y = (self.Weights.T @ input_vec + self.bias).copy()
        self.z = f(self.y).copy()
        return self.z
    def output(self,input_vec,f):
        return f(self.Weights.T @ input_vec + self.bias)
