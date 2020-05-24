import numpy as np
from activation import relu, sigmoid, softmax
import matplotlib.pyplot as plt
import math
import time
import random


def log(num: float, base: float = 10) -> float:
    return math.log(num, base)


class Modal:

    def __init__(self, input_shape: int, lr: float = 0.1, seed: bool = False, weights_cost: float = 0.001):
        # control
        if lr == 0:
            print("lr should not be 0!")
        elif lr < 0:
            print("lr should be positive!")
        if weights_cost < 0:
            print("weights_cost should be positive!")
        if seed:
            np.random.seed(42)
        self.layers = []
        self.input_shape = input_shape
        self.lr = lr
        self.weights_cost = weights_cost

    class LayerDense:
        def __init__(self, size: int, before_size: int, activation, weights=None, biases=None):
            self.activation = activation
            # [[weight, weight ...], [weight, weight ...] ...]
            self.weights = weights if weights is not None else np.random.randn(size, before_size) * np.sqrt(1 / before_size + size)
            self.biases = biases if biases is not None else np.zeros((1, size))     # [[0, 0, ...]]
            self.size = size
            self.before_size = before_size

            # control error
            if len(self.weights) != size:
                raise Exception (F"Error, num of nuerus creted = {len(self.weights)} but shuold be: {size}")
            elif len(self.weights[0]) != before_size:
                raise Exception (F"Error, num of weights in ecsh neurun = {len(self.weights[0])} but should be: {before_size}")

        def calculate(self, inputs):
            """
            :param inputs: [input, input ...] input = []
            :return: [output, output ...] output = []
            """
            return self.activation(np.dot(inputs, self.weights.T) + self.biases)

    def add(self, num: int, activation, weights=None, biases=None):
        self.layers.append(self.LayerDense(num, self.layers[-1].size if len(self.layers) != 0 else self.input_shape,
                                           activation, weights, biases))

    def backpropagation(self, inputs, y):
        """
        :param inputs: [input, input ...] input = []
        :return: int - the cost
        """
        lst = self.feed_forward(inputs)

        dalta = []

        error = y - lst[-1]
        dalta.append(error)     # * self.lr)
        temp = lst[-2].T.dot(dalta[-1])
        new_w = self.layers[-1].weights.T + ((temp - self.weights_cost * self.layers[-1].weights.T) * self.lr)
        self.layers[-1].weights = new_w.T

        # revered loop
        for i_layer in range(len(self.layers) - 2, 0, -1):
            dalta.append(dalta[-1].dot(self.layers[i_layer + 1].weights) * np.array(self.layers[i_layer].activation(lst[i_layer], True)))
            temp = lst[i_layer - 1].T.dot(dalta[-1])
            new_w = self.layers[i_layer].weights.T + ((temp - self.weights_cost * self.layers[i_layer].weights) * self.lr)
            self.layers[i_layer].weights = new_w.T

        dalta.append(dalta[-1].dot(self.layers[1].weights) * np.array(self.layers[0].activation(lst[0], True)))
        temp = inputs.T.dot(dalta[-1])
        new_w = self.layers[0].weights.T + ((temp - self.weights_cost * self.layers[0].weights.T) * self.lr)
        self.layers[0].weights = new_w.T

    def feed_forward(self, inputs):
        """
        :param inputs: [input, input ...] input = []
        :return: [[output layer[0], output layer[1]...], [...], ...] output layer[i] = []
        """
        lst = []
        for i, layer in enumerate(self.layers):
            if i == 0:
               lst.append(np.array(layer.calculate(inputs)))
            else:
                lst.append(np.array(layer.calculate(lst[-1])))
        return lst

    def predict(self, inputs):
        """
        :param inputs: [input, input ...] input = []
        :return: [output, output ...] output = []
        """
        # contorl bug
        if len(inputs[0]) != self.input_shape:
            raise Exception("worng shape input")

        return self.feed_forward(inputs)[-1]

    def fit(self, x, y, batch_size: int, epochs: int = 1):
        for _ in range(epochs):
            lst_x = []
            lst_y = []
            for i, input in enumerate(x):
                if len(lst_x) < batch_size:
                    lst_x.append(input)
                    lst_y.append(y[i])
                else:
                    # call backpropagation
                    self.backpropagation(np.array(lst_x), np.array(lst_y))
                    lst_x = []
                    lst_y = []
                    lst_x.append(input)
                    lst_y.append(y[i])
            if len(lst_x) != 0:
                # call backpropagation
                self.backpropagation(np.array(lst_x), np.array(lst_y))


m = Modal(2, seed=True)
m.add(3, sigmoid)
m.add(1, sigmoid)
m.backpropagation(np.array([[0, 0]]), np.array([[0]]))

X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([ [0],   [1],   [1],   [0]])


for _ in range(30000):
    # m.backpropagation(X, Y)
    m.fit(X, Y, batch_size=4, epochs=1)

lst = m.predict(X)
# print(lst)
print([round(i[0], 4) for i in lst])

def print_w_b(modal: Modal):
    for layer in modal.layers:
        print(F"layer.w: {layer.weights}")
        print(F"layer.b: {layer.biases}")
        print()

print_w_b(m)








