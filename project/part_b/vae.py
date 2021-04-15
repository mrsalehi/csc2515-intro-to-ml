import torch
from torch import nn
import wandb
from torch import optim
import numpy as np
import random

import sys
sys.path.append("..")

from utils import *


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


class VAE(nn.Module):
    def __init__(self, num_question, n_hidden_units=[50, 10]):
        
        self.dim_z = n_hidden_units[:-1] / 2 
        n_units = [num_question] + n_hidden_units

        self.enc = []
        for i in range(len(n_units) - 1):
            self.enc.append(nn.Linear(n_units[i], n_units[i+1]))
            if i < len(n_units) - 2:
                self.enc.append(nn.Tanh())
        
        self.enc = nn.Sequential(*self.enc)

        self.dec = []
        for i in range(len(n_units) - 1, 1):
            self.enc.append(nn.Linear(n_units[i], n_units[i-1]))
            if i > 0:
                self.dec.append(nn.Tanh())

        self.dec = nn.Sequential(*self.dec)

    def forward(self, x):
        batch_size = x.size(0)
        out_enc = self.encoder(x)
        mu, log_var = out_enc[:self.dim_z], out_enc[self.dim_z:]
        z = torch.randn_like(log_var) * torch.exp(0.5 * log_var) + mu
        x_recon = self.dec(z)

        return mu, log_var, x_recon


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


def data_loader(zero_train_matrix, train_matrix, batch_size, shuffle=True):
    if shuffle:
        idxs = np.random.permutation(len(train_matrix))
        zero_train_matrix, train_matrix = zero_train_matrix[idxs], train_matrix[idxs]
    
    for i in range(0, len(zero_train_matrix) // batch_size):
        yield zero_train_matrix[i * batch_size: (i+1)*batch_size], train_matrix[i * batch_size: (i+1)*batch_size]


def train(model, train_data, zero_train_data, valid_data, eval_input_matrix, cfg):
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
            X_recon, log_var, mu = model(X_zero)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = torch.isnan(X)
            target[nan_mask] = X_recon[nan_mask]

            kl_loss = 0.5 * torch.sum(- 1. - log_var + mu**2 + torch.exp(log_var), axis=1)
            recon_loss = torch.mean(nn.BCEWithLogitsLoss(X_recon, target))
            
            loss = recon_loss + kl_loss
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

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


def main():
    TRAIN = True

    n_hidden_units = [50]
    lamb = 0.
    num_epoch = 1000
    lr = 1e-3
    batch_size = 128
    chkpt_name = "50"
    activation = "sigmoid"
    optim = "adam"

    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    eval_input_matrix = build_evaluation_input_matrix(valid_data, zero_train_matrix)
    train_data = build_train_data_dict(train_matrix)

    if TRAIN:
        wandb.init(project='csc2515-proj', name=chkpt_name)
        wandb.config.num_epoch = num_epoch
        wandb.config.lr = lr
        wandb.config.activation = activation
        wandb.config.lamb = lamb
        wandb.config.n_hidden_units = n_hidden_units
        wandb.config.chkpt_name = chkpt_name
        wandb.config.batch_size = batch_size
        wandb.config.optim = optim

        cfg = wandb.config

        model = VAE(num_question=1774, activation=activation, n_hidden_units=n_hidden_units)

        model.to(DEVICE)


def main():
    pass


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Runnnig on', DEVICE)
    
    main()