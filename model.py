import os
import numpy as np
import loss
import random



def one_hot_encode(targets, num_classes) -> int:
    """convert target into one-hot encoding

    Example:
    0 -> [1, 0, 0]
    1 -> [0, 1, 0]
    2 -> [0, 0, 1]

    Arguments:
        num_classes: number of classes in the dataset
    """
    one_hot_targets = np.zeros((targets.shape[0], num_classes))
    for i, target in enumerate(targets):
        try:
            one_hot_targets[i, int(target)] = 1
        except:
            raise ValueError("label for CE loss must be interger")

    return one_hot_targets


def data_loader(data, targets, batch_size):
    """generate batches of data points and targets

    """
    # number of batches in each epoch
    num_batch = (len(data) + 1) // batch_size

    # shuffle the dataset
    idxs = np.arange(len(data))
    np.random.shuffle(idxs)

    for i in range(num_batch):
        begin_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data))

        # get batches of data points and targets with batch_idxs
        batch_idxs = idxs[begin_idx:end_idx]
        batch_data = data[batch_idxs]
        batch_targets = targets[batch_idxs][:]

        yield batch_data, batch_targets


def tanh(x):
    return np.tanh(x)


def grad_tanh(x):
    # gradient of tanh function
    return 1 - (tanh(x) ** 2)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def grad_sigmoid(x):
    # gradient of sigmoid function
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x):
    # softmax function used for CE loss with num_classes > 2
    return np.exp(x) / np.tile(np.sum(np.exp(x), axis=1), (3, 1)).transpose()


def grad_softmax(x, target):
    # gradient of softmax function
    return softmax(x) * (target - softmax(x))


def identity(x):
    return x


def grad_identity(x):
    return 1


class NN:
    def __init__(self, layers, lr, g, output_act_fnc):
        """The entire neural network model used for forward propagation and backward propagation

        Arguments:
            layers: layers (ex: [2, 2, 3, 1] -- two hidden layers with 2 and 3 nodes
                                [2, 2, 1] -- one hidden layer with 2 nodes
            lr: learning rate
        """

        self.W = []
        self.grad_W = []
        self.b = []
        self.grad_b = []
        # self.deltas = []
        self.layers = layers
        self.lr = lr

        # hidden layer activations
        if g == 'tanh':
            self.g = tanh
            self.grad_g = grad_tanh
        elif g == 'sigmoid':
            self.g = sigmoid
            self.grad_g = grad_sigmoid
        else:
            # raise value error if hidden activation function is not "tanh" or "sigmoid"
            raise ValueError("Hidden activation function should be either \"tanh\" or \"sigmoid\".")

        # output layer activation function
        if output_act_fnc == 'identity':
            self.g_output = identity
            self.grad_g_output = grad_identity
        elif output_act_fnc == 'sigmoid':
            self.g_output = sigmoid
            self.grad_g_output = grad_sigmoid
        elif output_act_fnc == 'softmax':
            self.g_output = softmax
            self.grad_g_output = grad_softmax
        else:
            # raise value error if output activation function is not "tanh" or "sigmoid"
            raise ValueError("Output activation function should be either \"identity\", \"sigmoid\", or \"tanh\".")

        # self.z stores the values of z in each layer;
        # z = wx + b
        self.zs = None
        # self.a stores the output/activation of g in each layer;
        # a = inputs or a = g(z)
        self.avs = None

        # initialize the  weights and bias
        for i in range(len(layers) - 1):
            self.W.append(np.random.randn(layers[i], layers[i + 1]) * 0.1)
            self.grad_W.append(np.zeros((layers[i], layers[i + 1])))
            self.b.append(np.zeros((1, layers[i + 1])))
            self.grad_b.append(np.zeros((1, layers[i + 1])))

    def forward(self, x):
        """forward propagation

        Arguments:
            x: features
        """

        # self.z stores the values of z in each layer;
        # z = wx + b
        self.zs = [x]
        # self.avs stores the output/activation of g in each layer;
        # g = inputs or g = a(z)
        self.avs = [x]

        a = x
        for i in range(len(self.layers)-2):
            z = a @ self.W[i] + self.b[i]
            # print(a.shape) # (batch_size, features)
            # print(z.shape) # (batch_size, features)
            self.zs.append(z)
            a = self.g(z)
            self.avs.append(a)

        # output layer
        z = self.avs[-1] @ self.W[-1] + self.b[-1]
        self.zs.append(z)
        # output layer activation
        a = self.g_output(z)
        self.avs.append(a)

        return z

    def backprop(self, loss):
        # output layer
        # print(f"loss * self.grad_g_output(self.zs[-1]):{loss * self.grad_g_output(self.zs[-1])}")
        # print(f"loss:{loss}")
        deltas = [loss * self.grad_g_output(self.zs[-1])]
        # deltas = [loss]
        # deltas.insert(0, loss * self.grad_g_output(self.zs[-1]))
        self.grad_W[-1] = self.avs[-2].T @ deltas[-1]
        # self.grad_W[-1] = np.dot(self.avs[-2].T, deltas[-1])
        self.grad_b[-1] = np.sum(deltas[-1], axis=0)

        # hidden layer gradients
        for i in range(len(self.layers) - 3, 0, -1):
            # print(self.W[i+1].shape)
            # print(self.zs[i + 1].shape)
            deltas.insert(0, self.grad_g(self.zs[i + 1]) * (deltas[0] @ self.W[i + 1].T))
            # deltas.insert(0, self.grad_g(self.zs[i+1]) * np.dot(deltas[0], self.W[i+1].T))
            self.grad_W[i] = np.dot(self.avs[i].T, deltas[0])
            self.grad_b[i] = np.sum(deltas[0], axis=0)

        # print(self.zs[1].shape)
        # print(deltas[0].shape)
        # print(self.W[1].T.shape)
        # print((self.grad_g(self.zs[1]) * np.dot(deltas[0], self.W[1].T)).shape)

        # input layer
        deltas.insert(0, self.grad_g(self.zs[1]) * np.dot(deltas[0], self.W[1].T))
        self.grad_W[0] = np.dot(self.avs[0].T, deltas[0])
        self.grad_b[0] = np.sum(deltas[0], axis=0)

        # for i in range(len(self.W)):
        #     print(f"self.W[{i}]: ", self.W[i].shape)
        #
        # for i in range(len(self.zs)):
        #     print(f"self.zs[{i}]: ", self.zs[i].shape)
        #
        # for i in range(len(deltas)):
        #     print(f"deltas[{i}]: ", deltas[i].shape)

        self.update()

    def update(self):
        for i in range(len(self.W)):
            self.W[i] -= self.lr * self.grad_W[i]
            self.b[i] -= self.lr * self.grad_b[i]

            # set the gradient of each layer as 0
            self.grad_W[i] *= 0
            self.grad_b[i] *= 0


if __name__ == '__main__':
    random.seed(15)

    CWD = os.path.dirname(os.getcwd())
    TRAIN_FILE = "data/train_data1.txt"
    TRAIN_PATH = os.path.join(CWD, TRAIN_FILE)
    TARGET_FILE = "data/train_target1.txt"
    TARGET_PATH = os.path.join(CWD, TARGET_FILE)

    # train_x = np.random.rand(150, 5)
    # train_y = np.random.rand(150, 1)

    train_x = np.genfromtxt(TRAIN_FILE)
    train_y = np.genfromtxt(TARGET_FILE).reshape(-1, 1)

    net = NN([5, 4, 2, 1], 0.01, 'sigmoid', 'identity')

    # print(net.W[1].shape)

    criterion = loss.Loss('sse', 1)

    history_loss = []
    for epoch in range(100):

        epoch_loss = []

        # batch training
        for batch_data, batch_targets in data_loader(train_x, train_y, 4):
            # get the prediction of forward propagation
            pred = net.forward(batch_data)

            # calculate the loss and gradient of loss
            loss, grad_loss = criterion.forward(pred, batch_targets)
            # do backward propagation and update network weights
            net.backprop(grad_loss)

            # save the loss
            sum_loss = np.sum(loss)
            epoch_loss.append(sum_loss)

            # print the average loss in the batch
            # print("loss: %.4f" % (sum_loss / 5))

        # print the average loss in the entire dataset
        avg_loss = sum(epoch_loss) / len(train_x)
        history_loss.append(avg_loss)

        if avg_loss < 0.01:
            print("early stopping at epoch %d because loss = %.4f < %.4f" % (epoch, avg_loss, 0.01))
            break

        print("num epoch: %d, loss: %.5f" % (epoch, avg_loss))

    print("finish training. Doing inference")

    output = net.forward(train_x)

    loss, _ = criterion.forward(output, train_y)

    print("Final loss: %.4f" % (loss / len(train_y)))
