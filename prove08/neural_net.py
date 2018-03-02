import numpy as np
import math
from sklearn import preprocessing

# use to switch between classification and regression
regression = True


class Connection:
    def __init__(self, connected_neuron):
        self.connected_neuron = connected_neuron
        self.weight = np.random.normal()
        self.dWeight = 0.0


class Neuron:
    momentum = .000001
    learning_rate = .05

    def __init__(self, layer):
        self.connections = list()
        self.error = float()
        self.gradient = float()
        self.output = float()
        self.outputNeuron = False
        # make first without connections to the left
        if layer is None:
            pass
        # connect neuron to layer before it
        else:
            for neuron in layer:
                connection = Connection(neuron)
                self.connections.append(connection)

    def add_error(self, err):
        self.error += err

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x * 1.0))

    def d_sigmoid(self, x):
        return x * (1.0 - x)

    def set_error(self, err):
        self.error = err

    def set_output(self, output):
        self.output = output

    def get_output(self):
        return self.output

    def feed_forward(self):
        output_sum = 0
        # no connections, no need to feed forward
        if len(self.connections) == 0:
            return
        for connection in self.connections:
            # print("output", connection.connected_neuron.get_output())
            # print("weight", connection.weight)
            # output of node on left * its weight
            output_sum += connection.connected_neuron.get_output() * connection.weight
        if regression:
            output = output_sum
        else:
            output = self.sigmoid(output_sum)
        self.set_output(output)

    def back_propagate(self):
        # calculate gradient (error * derivative of sigmoid
        self.gradient = self.error * self.d_sigmoid(self.output)
        # for each connection
        for connection in self.connections:
            # set your change in weight to momentum * output * gradient + learningrate * connections change in weight
            connection.dWeight = Neuron.momentum * (connection.connected_neuron.output * self.gradient) \
                                 + self.learning_rate * connection.dWeight
            # set your new weight equal to weight + change in weight
            connection.weight = connection.weight + connection.dWeight
            # change error on the connected neuron
            connection.connected_neuron.add_error(connection.weight * self.gradient)
        self.error = 0


class Network:
    def __init__(self, shape):
        print(shape)
        self.layers = list()
        for neurons in shape:
            # one layer, will append to layers
            layer = []
            for i in range(neurons):
                # first layer has no neurons to the left
                if len(self.layers) == 0:
                    layer.append(Neuron(None))
                # connect neuron to each node in the layer before it
                else:
                    layer.append(Neuron(self.layers[-1]))
            # bias node
            layer.append(Neuron(None))
            # set output of bias node to -1
            layer[-1].set_output(-1)
            # append layer to the layers
            self.layers.append(layer)
        # set last neurons in net to output neurons
        for output in self.layers[-1]:
            if not output.get_output() == -1:
                output.outputNeuron = True

    # sets inputs of the first first layer
    def set_input(self, inputs):
        for i in range(len(inputs)):
            self.layers[0][i].set_output(inputs[i])

    # get error of the net given a target
    def get_error(self, target):
        err = 0
        for i in range(len(target)):
            e = (target[i] - self.layers[-1][i].get_output())
            err += e ** 2
        err /= len(target)
        err = math.sqrt(err)
        return err

    # call the feed forward method on each node in the net
    def feed_forward(self):
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.feed_forward()

    # call the back propagate method for each neuron in the net
    def back_propagate(self, output):
        # find the error of the output
        for i in range(len(output)):
            print(output[i] - self.layers[-1][i].get_output())
            self.layers[-1][i].set_error(output[i] - self.layers[-1][i].get_output())
        # back propagate in reverse so we are using the correct weights
        for layer in self.layers[::-1]:
            for neuron in layer:
                neuron.back_propagate()

    def get_result(self):
        output = list()
        # loop though each neuron in output layer and get output
        for neuron in self.layers[-1]:
            output.append(neuron.get_output())
        # remove the bias node
        output.pop()
        return output


class Classifier:
    def fit(self, data_train, target_train):
        if regression:
            num_output_node = 1
        else:
            num_output_node = len(np.unique(target_train))
        shape = list()
        # input neurons
        shape.append(len(data_train[0]))
        # hidden layers
        shape.append(5)
        # output neurons
        shape.append(num_output_node)
        net = Network(shape)
        # set upper limit to training classifier
        limit = 0
        while True:
            err = 0
            target_train = np.array(target_train, dtype=int)
            for i in range(len(data_train)):
                output = np.zeros(num_output_node, dtype=int)
                if regression:
                    output[0] = target_train[i]
                else:
                    output[target_train[i]] = 1
                # print(output)
                net.set_input(data_train[i])
                net.feed_forward()
                net.back_propagate(output)
                # print(limit, net.get_result())
                err += net.get_error(output)
            limit += 1
            # print(limit)
            if err < 5 or limit > 100:
                break
        return Model(net)


class Model:
    def __init__(self, net):
        self.net = net

    def predict(self, data_test):
        predictions = np.zeros(len(data_test), dtype=int)
        for i in range(len(data_test)):
            self.net.set_input(data_test[i])
            self.net.feed_forward()
            print(self.net.get_result())
            predictions[i] = np.argmax(self.net.get_result())
        return predictions
