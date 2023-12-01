import loss
import os
from model import model
import numpy as np
import random

if __name__ == '__main__':
    random.seed(15)

    CWD = os.path.dirname(os.getcwd())
    TRAIN_FILE = "data/train_data1.txt"
    TRAIN_PATH = os.path.join(CWD, TRAIN_FILE)
    TEST_FILE = "data/train_target1.txt"
    TEST_PATH = os.path.join(CWD, TEST_FILE)

    # train_x = np.random.rand(150, 5)
    # train_y = np.random.rand(150, 1)

    train_x = np.genfromtxt(TRAIN_PATH)
    train_y = np.genfromtxt(TEST_PATH).reshape(-1, 1)

    net = NN([5, 4, 2, 1], 0.01, 'sigmoid', 'identity')

    # print(net.W[1].shape)

    criterion = Loss('sse', 1)

    history_loss = []
    for epoch in range(100):

        epoch_loss = []

        # batch training
        for batch_data, batch_targets in data_loader(train_x, train_y, 4):
            # print("batch")
            # print("batch data: ", batch_data)
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
