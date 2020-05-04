import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles


# Create layers
class NeuralLayer:
    # Receive number of connections, number of neurons and activation function
    def __init__(self, n_connections, n_neurons, act_fun):
        self.act_fun = act_fun
        self.bias = np.random.rand(1, n_neurons) * 2 - 1  # Generate a random vector with values from -1 to 1
        self.weight = np.random.rand(n_connections, n_neurons) * 2 - 1  # Initialize the weights with random values


# Create Neural Network
def create_nn(_topology, act_fun):
    nn_layers = []
    for l, layer in enumerate(_topology[:-1]):
        nn_layers.append(NeuralLayer(_topology[l], _topology[l + 1], act_fun))
    return nn_layers


# Train function
def train(_neural_net, x, y, cost_fun, learn_r=0.05, _train=False):
    # forward pass
    out = [(None, x)]
    for l, layer in enumerate(_neural_net):
        z = out[-1][1] @ _neural_net[l].weight + _neural_net[l].bias
        a = _neural_net[l].act_fun[0](z)

        out.append((z, a))

    if _train:
        # Backward pass (Back propagation)
        deltas = []
        for l in reversed(range(0, len(_neural_net))):
            z = out[l + 1][0]
            a = out[l + 1][1]

            if l == len(_neural_net) - 1:
                deltas.insert(0, cost_fun[1](a, y) * _neural_net[l].act_fun[1](a))
            else:
                deltas.insert(0, deltas[0] @ _W.T * _neural_net[l].act_fun[1](a))

            _W = _neural_net[l].weight

            # Gradient decent
            _neural_net[l].bias = _neural_net[l].bias - np.mean(deltas[0], axis=0, keepdims=True) * learn_r
            _neural_net[l].weight = _neural_net[l].weight - out[l][1].T @ deltas[0] * learn_r
    return out[-1][1]


# Activation Functions
sigmoid = (lambda x: 1 / (1 + np.e ** (-x)),  # Sigmoid function
           lambda x: x * (1 - x))  # Derived from function

# Cost Function
e2_mean = (lambda yp, yr: np.mean((yp - yr) ** 2),
           lambda yp, yr: 2 * (yp - yr))

if __name__ == "__main__":
    # Create Dataset
    n = 250  # Dates
    p = 2  # Characteristics

    X, Y = make_circles(n_samples=n, factor=0.5, noise=0.09)  # Generate the dataset
    Y = Y[:, np.newaxis]

    topology = [p, 4, 8, 1]  # Topology of NeuralNet
    neural_net = create_nn(topology, sigmoid)

    loss = []
    for i in range(1000):
        pY = train(neural_net, X, Y, e2_mean, learn_r=0.09, _train=True)

        if i % 5 == 0:
            # print(pY)
            loss.append(e2_mean[0](pY, Y))

            res = 60

            _x0 = np.linspace(-1.5, 1.5, res)
            _x1 = np.linspace(-1.5, 1.5, res)

            _Y = np.zeros((res, res))

            for i0, x0 in enumerate(_x0):
                for i1, x1 in enumerate(_x1):
                    _Y[i0, i1] = train(neural_net, np.array([[x0, x1]]), Y, e2_mean)[0][0]

            plt.pcolormesh(_x0, _x1, _Y, cmap="coolwarm")
            plt.axis("equal")

            plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c="skyblue")  # Visualize the data
            plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c="salmon")

            plt.draw()
            plt.pause(0.1)
            plt.clf()
