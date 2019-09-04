import numpy as np
class Hopfield:
    def __init__(self,size):
        self.size = size;
        self.state = np.zeros(size)
        self.biases = np.zeros(size)
        for i in range(size):
            self.biases[i] = np.random.choice([-1,1])
        self.W = np.zeros([size,size])
    def setState(self,state):
        self.state = state
    def setWeight(self,W):
        self.W = W
    def updateState(self,idx):
        totalWeight = 0;
        for i in range(len(self.W[idx])):
            totalWeight += self.W[idx][i] * self.state[i]
        if(totalWeight > self.biases[idx]):
            self.state[idx] = 1
        else:
            self.state[idx] = -1
    def getState(self):
        return self.state
    def calcEng(self):
        weightSum = 0
        biasesSum = 0
        for i in range(self.W.shape[0]):
            biasesSum += self.biases[i] * self.state[i]
            for j in range(len(self.W[i])):
                weightSum += self.W[i,j] * self.state[i] * self.state[j]
        return -0.5 * weightSum + biasesSum
    def converge(self):
        energy_list = []
        for i in range(10):
            self.updateState(np.random.choice(len(self.state)))
            energy_list.append(self.calcEng())
        if(energy_list[0] == energy_list[-1]):
            return True
        return False

    def update(self,neuron):
        target_state = self.getState().copy()
        target_state[neuron] =  -target_state[neuron]
        return self.getState(),target_state
