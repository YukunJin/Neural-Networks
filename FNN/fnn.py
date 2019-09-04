from Layers import *
from util import *
class FNN:

    def __init__(self,input_size):
        self.numberOfLayers = 1
        self.layer_list = []
        self.input_size = input_size

    def addLayer(self,input_size,output_size):

        if(len(self.layer_list) != 0):
            prev_layer = self.layer_list[-1]
            if (prev_layer.output_size != input_size):
                print("Warning: New_Layer.input_size != Prev_Layer.output_size" )
                sys.exit(0)
        self.numberOfLayers += 1
        l = Layers(input_size,output_size,self.numberOfLayers)
        self.layer_list.append(l)


    def feed(self,input_vec,f):
        output_vec = input_vec
        for i in self.layer_list:
            output_vec = i.evaluate(output_vec,f)
        return output_vec
    def predict(self,input_vec,f):
        output_vec = input_vec
        for i in self.layer_list:
            output_vec = i.output(output_vec,f)
        return output_vec

    def act_func(self,f,input_vec):
        return f(input_vec)
    def mini_batch(self,train_data,expected,batches):
        size = len(train_data)

        batch_input = []
        batch_expected = []
        for i in range(len(train_data)):
            batch_input.append(train_data[i])
            batch_expected.append(expected[i])
        batch_input = np.array(batch_input).reshape(int(size/batches),batches)
        batch_expected = np.array(batch_expected).reshape(int(size/batches),batches)
        return batch_input,batch_expected

    def train(self,alpha,train_data,expected,f=sigmoid,dc_dz=dcross_dz,batches = 1,epoches = 1000):
        train_data = np.array(train_data)
        expected = np.array(expected)
        cost = []
        if(len(train_data) != len(expected)):
            print("Warning: Training data sizes do not match")
            sys.exit(0)
        for epoch in range(epoches):
            idx = np.arange(len(train_data))
            np.random.shuffle(idx)
            train_data = train_data[idx]
            expected = expected[idx]
            for x,y in zip(train_data[:batches],expected[:batches]):
                self.feed(x,f)
                delta_list = {}
                for i in range(len(self.layer_list)-1,-1,-1):
                    if i == len(self.layer_list)-1:
                        delta = dc_dz(self.layer_list[i].z,y) * d_sigmoid(self.layer_list[i].y)
                        delta_list[i] = delta
                    else:
                        delta = (delta_list[i+1] * self.layer_list[i+1].Weights.T) @ d_sigmoid(self.layer_list[i].y)
                        delta_list[i] = delta
                    weights_gradient = np.outer(delta,self.layer_list[i].x,)
                    bias_gradient = delta
                    self.layer_list[i].Weights -= alpha*weights_gradient.T
                    self.layer_list[i].bias -= alpha*bias_gradient
