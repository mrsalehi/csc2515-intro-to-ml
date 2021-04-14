# TODO: complete this file.
from neural_network import AutoEncoder, load_data
from torch.autograd import Variable
import torch
from torch import optim
import numpy as np


DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def bootstrap(train_matrix, zero_train_matrix):
    """
    data: numpy 2d array
    """
    idxs = np.random.choice(len(train_matrix), len(train_matrix))
    return train_matrix[idxs], zero_train_matrix[idxs]


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch, chkpt_name):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :param k: int  (just to be used in the checkpoint file name)

    :return: None
    """
    # TODO: Add a regularizer to the cost function. 
    
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    for epoch in range(0, num_epoch):
        print(epoch)

        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0).to(DEVICE)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)
            loss += 0.5 * lamb * model.get_weight_norm()
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        # valid_acc = evaluate(model, zero_train_data, valid_data)

    model.cpu()
    torch.save({
    'model_state_dict': model.state_dict(),
    }, f'net_{chkpt_name}.pt')
    model.to(DEVICE)

        # print("Epoch: {} \tTraining Cost: {:.6f}\t "
        #       "Valid Acc: {}".format(epoch, train_loss, valid_acc))
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


def ensemble_evaluate(models, train_data, valid_data):
    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        
        outputs = []
        
        for model in models:
            outputs.append(model(inputs.to(DEVICE)).cpu())

        output = torch.mean(torch.cat(outputs, dim=0), dim=0)

        guess = output[valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1

    return correct / float(total)

def main():
    TRAIN = False
    EVAL = True

    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    lamb = 0.
    k = 50
    lr = 1e-2
    num_epoch = 50

    models = []

    if TRAIN:
        for i in range(3):
            print('model', i)
            model = AutoEncoder(num_question=1774, k=k)
            model.to(DEVICE)
            train_matrix_bstrap, zero_train_matrix_bstrap = bootstrap(
                train_matrix, zero_train_matrix)

            train(model, lr, lamb, train_matrix_bstrap, zero_train_matrix_bstrap,
                valid_data, num_epoch, chkpt_name=f'ensemble_{i}')

    if EVAL:
        for i in range(3):
            k = 50 
            model = AutoEncoder(num_question=1774, k=k)
            checkpoint = torch.load(f'net_ensemble_{i}.pt')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(DEVICE)
            model.eval()
            models.append(model)

        

        print('Val acc:', ensemble_evaluate(models, zero_train_matrix, valid_data))
        print('Test acc:', ensemble_evaluate(models, zero_train_matrix, test_data))


if __name__ == '__main__':
    main()
