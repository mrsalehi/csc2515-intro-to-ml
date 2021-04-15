import sys
sys.path.append("..")

from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset

import numpy as np
import torch
from time import time
import random
import csv

import wandb


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


def data_loader(zero_train_matrix, train_matrix, batch_size, shuffle=True):
    if shuffle:
        idxs = np.random.permutation(len(train_matrix))
        zero_train_matrix, train_matrix = zero_train_matrix[idxs], train_matrix[idxs]
    
    for i in range(0, len(zero_train_matrix) // batch_size):
        yield zero_train_matrix[i * batch_size: (i+1)*batch_size], train_matrix[i * batch_size: (i+1)*batch_size]


class AutoEncoder(nn.Module):
    def __init__(self, num_question, n_hidden_units, activation='sigmoid'):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.

        n_units = [num_question] + n_hidden_units + [num_question]

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        
        self.net = []

        for i in range(len(n_units) - 1):
            self.net.append(nn.Linear(n_units[i], n_units[i+1]))

            if i < len(n_units) - 2:
                self.net.append(self.activation)
            else:
                self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)


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
        out = self.net(inputs)
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, train_data, zero_train_data, valid_data, 
eval_input_matrix, cfg):
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
    if cfg.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.lamb)
    elif cfg.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.lamb)
    

    num_student = train_data.shape[0]

    val_accs = []
    train_losses = []

    best_val_acc = 0.

    for epoch in range(0, cfg.num_epoch):
        train_loss = 0.

        for X_zero, X in data_loader(
            zero_train_data, train_data, batch_size=cfg.batch_size, shuffle=False):
            
            X_zero = X_zero.to(DEVICE)
            target = X_zero.clone()

            optimizer.zero_grad()
            output = model(X_zero)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = torch.isnan(X)
            target[nan_mask] = output[nan_mask]

            loss = torch.mean(torch.sum((output - target) ** 2., dim=-1))
            #loss += 0.5 * cfg.lamb * model.get_weight_norm()

            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        # if epoch == 300:
        #     print('Reducing learning rate')
        #     for g in optimizer.param_groups:
        #         g['lr'] *= 0.1

        valid_acc = evaluate(model, eval_input_matrix, valid_data)
        wandb.log({'Epoch': epoch, 'Val Acc': valid_acc, 'Train Loss': train_loss})
        
        if valid_acc > best_val_acc:
            model.cpu()
            best_val_acc = valid_acc
            torch.save({'model_state_dict': model.state_dict()}, cfg.chkpt_name)
            model.to(DEVICE)
        
        
        val_accs.append(valid_acc)
        train_losses.append(train_loss)

        print("Epoch: {} \tTraining Cost: {:.6f}\t "
        "Valid Acc: {}".format(epoch, train_loss, valid_acc))
    
    return train_losses, val_accs, best_val_acc


def build_evaluation_input_matrix(data, train_data):
    matrix = []
    for u in data["user_id"]:
        matrix.append(train_data[u])
    
    return torch.stack(matrix, dim=0)


def build_train_data_dict(train_matrix):
    train_data = {'user_id': None, 'question_id': None, 'is_correct': None}

    not_nan_idxs = np.where(1 - np.isnan(train_matrix))
    
    train_data['user_id'] = not_nan_idxs[0].tolist()
    train_data['question_id'] = not_nan_idxs[1].tolist()
    train_data['is_correct'] = [train_matrix[i, j] for i, j in zip(train_data['user_id'], train_data['question_id'])]

    return train_data


def evaluate(model, val_input_matrix, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    model.eval()

    question_ids = torch.tensor(valid_data["question_id"])
    is_correct = torch.tensor(valid_data["is_correct"])

    batch_size = 128
    corrects = 0

    for i in range(len(question_ids) // batch_size):
        inputs = val_input_matrix[i*batch_size:(i+1)*batch_size]
        qids = question_ids[i*batch_size:(i+1)*batch_size]
        is_cor = is_correct[i*batch_size:(i+1)*batch_size]
        
        output = model(inputs.to(DEVICE)).cpu()
        guesses = output[list(range(batch_size)), qids] >= 0.5
        corrects += torch.sum(guesses == is_cor).item()

    return corrects / len(question_ids)


def main():
    # is_nan = torch.isnan(train_matrix)
    # print(torch.mean(torch.sum(is_nan, dim=-1).float()))  # out: 1669.4

    TRAIN = True

    n_hidden_units = [50]
    lamb = 0.
    num_epoch = 1000
    lr = 1e-3
    batch_size = 128
    chkpt_name = "50"
    activation = "sigmoid"
    optim = "adam"


    if TRAIN:
        wandb.init(project='csc2515-proj', name=chkpt_name)


    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    eval_input_matrix = build_evaluation_input_matrix(valid_data, zero_train_matrix)
    train_data = build_train_data_dict(train_matrix)

    if TRAIN:
        wandb.config.num_epoch = num_epoch
        wandb.config.lr = lr
        wandb.config.activation = activation
        wandb.config.lamb = lamb
        wandb.config.n_hidden_units = n_hidden_units
        wandb.config.chkpt_name = chkpt_name
        wandb.config.batch_size = batch_size
        wandb.config.optim = optim

        cfg = wandb.config

        model = AutoEncoder(num_question=1774, activation=activation, n_hidden_units=n_hidden_units)

        model.to(DEVICE)
        
        train_losses, val_accs, best_val_acc = train(
            model=model, train_data=train_matrix, zero_train_data=zero_train_matrix, 
            valid_data=valid_data, eval_input_matrix=eval_input_matrix, cfg=cfg)


    #model = AutoEncoder(num_question=1774, k=k)
    model = AutoEncoder(num_question=1774, n_hidden_units=n_hidden_units)
    checkpoint = torch.load(chkpt_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)

    test_input_matrix = build_evaluation_input_matrix(test_data, zero_train_matrix)
    
    print('Train acc:', evaluate(model,build_evaluation_input_matrix(train_data, zero_train_matrix), train_data))
    print('Val acc:', evaluate(model, eval_input_matrix, valid_data))
    print('Test acc:', evaluate(model, test_input_matrix, test_data))


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Runnnig on', DEVICE)
    
    main()
