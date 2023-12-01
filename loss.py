import numpy as np
from model import sigmoid, grad_sigmoid, softmax, grad_softmax
import pdb


def sse(preds, target):
    # the value of SSE loss
    return np.sum(1 / 2.0 * (preds - target)**2)


def grad_sse(input, target):
    # return the grad of SSE
    # print("======================================================================")
    # print((input - target).shape)
    # print("======================================================================")
    return input - target


def bce(preds, target):
    # the value of CE loss for binary classification
    loss = -(target * np.log(preds + 1e-12) + (1 - target) * np.log(1 - preds + 1e-12))
    return np.sum(loss)


def ce(preds, target):
    # the value of CE loss for multi-class classification
    loss = -(target * np.log(preds + 1e-12))
    return np.sum(loss)


def grad_bce(input, target):
    # return the grad of CE for binary classification
    sigmoid_input = sigmoid(input)
    target = np.expand_dims(target, axis=1)

    ret = - (target / sigmoid_input -
             (1 - target) / (1 - sigmoid_input)) * grad_sigmoid(input)
    if np.isnan(ret).any(): pdb.set_trace()

    return ret


def grad_ce(input, target):
    # return the grad of CE for multi-class classification
    softmax_input = softmax(input)
    ret = - 1 / softmax_input * grad_softmax(input, target)
    return ret


class Loss:
    def __init__(self, loss_type, num_classes):
        """the loss class used to calculate the loss and gradient of loss

        Argument:
            type_ls: the type of loss function "CE" or "SSE"
            num_classes: number of classes
        """
        self.loss_type = loss_type
        self.num_classes = num_classes

    def forward(self, preds, target):
        # need last layer of gradients here as well
        # grad_bce and grad_ce use grad_sigmoid and grad_softmax, respectively
        # because the last layer (preds) was not transformed
        # calculate the values and gradients of loss function
        if self.loss_type == 'sse':
            loss = sse(preds, target)
            grad_loss = grad_sse(preds, target)
        elif self.loss_type == 'ce':
            if self.num_classes > 2:
                # multi-class classification
                loss = ce(preds, target)
                grad_loss = grad_ce(preds, target)
            else:
                # binary classification
                loss = bce(preds, target)
                grad_loss = grad_bce(preds, target)

        return loss, grad_loss
