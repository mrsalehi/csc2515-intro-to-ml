import sys
sys.path.append("..")

from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from matplotlib import pyplot as plt

import numpy as np
import torch


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Runnnig on', DEVICE)

def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1|| + ||W^2||.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2)
        h_w_norm = torch.norm(self.h.weight, 2)
        return g_w_norm ** 2. + h_w_norm ** 2.

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        out = torch.sigmoid(self.h(torch.sigmoid(self.g(inputs))))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function. 
    
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    val_accs = []
    train_losses = []

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0).to(DEVICE)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)
            output = output

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            #nan_mask = torch.isnan(train_data[user_id]).to(DEVICE)
            #nan_mask = torch.tensor(nan_mask).to(DEVICE)
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)
            loss += 0.5 * lamb * model.get_weight_norm()
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        val_accs.append(valid_acc)
        train_losses.append(train_loss)

        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
    
    return train_losses, val_accs
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs.to(DEVICE)).cpu()

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    # ks = [10, 50, 100, 200, 500]
    
    # I got the best result with k=100
    # k = 100
    
    # Set optimization hyperparameters.
    # lr = 1e-2
    # num_epoch = 40
    # lamb = 0.01
    
    # best_model = None
    # best_k = None
    # best_val_acc = 0.

    # val_accs = []
    # train_losses = []

    # for k in ks:
    # print(k)
    # model = AutoEncoder(num_question=1774, k=k)
    # model.to(DEVICE)
    # train_losses, val_accs = train(
    #     model, lr, lamb, train_matrix, zero_train_matrix,
    #     valid_data, num_epoch)

    # fig, axes = plt.subplots(2, figsize=(20, 20))
    
    # axes[0].plot(train_losses)
    # axes[0].set_xlabel('Epoch')
    # axes[0].set_ylabel('Train loss')

    # axes[1].plot(val_accs)
    # axes[1].set_xlabel('Epoch')
    # axes[1].set_ylabel('Val Acc')

    # plt.savefig('nn.png')

    # test_acc = evaluate(model, zero_train_matrix, test_data)
    # print('Test acc', test_acc)
    
    # acc = evaluate(model, zero_train_matrix, valid_data)
    # if acc > best_val_acc:
    #     best_k = k
    #     best_val_acc = acc
    #     best_model = model

    # lr = 1e-3
    # num_epoch = 40

    # lambs = [1e-3, 1e-2, 1e-1, 1]
    # best_model = None
    # best_lamb = None
    # best_val_acc = 0.
    
    # for lamb in lambs:
    #     print(lamb)
    #     model = AutoEncoder(num_question=1774, k=100)
    #     model.to(DEVICE)
    #     train_losses, val_accs = train(
    #         model, lr, lamb, train_matrix, zero_train_matrix,
    #         valid_data, num_epoch)
        
    #     if val_accs[-1] > best_val_acc:
    #         best_lamb = lamb
    #         best_val_acc = val_accs[-1]
    #         best_model = model
    
    # print(best_lamb)

    lamb = 1e-3
    lr = 1e-2
    num_epoch = 40

    model = AutoEncoder(num_question=1774, k=100)
    model.to(DEVICE)
    train_losses, val_accs = train(
        model, lr, lamb, train_matrix, zero_train_matrix,
        valid_data, num_epoch)
    
    fig, axes = plt.subplots(2, figsize=(20, 20))
    
    axes[0].plot(train_losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Train loss')

    axes[1].plot(val_accs)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Val Acc')

    plt.savefig('nn-lamb.png')

    test_acc = evaluate(model, zero_train_matrix, test_data)
    print('Test acc', test_acc)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
