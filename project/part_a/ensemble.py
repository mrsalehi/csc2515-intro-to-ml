# TODO: complete this file.
from neural_network import DEVICE, AutoEncoder, train, load_data
from torch.autograd import Variable
import torch
import numpy as np


def bootstrap(train_matrix, zero_train_matrix):
    """
    data: numpy 2d array
    """
    idxs = np.random.choice(len(train_matrix), len(train_matrix))
    return train_matrix[idxs], zero_train_matrix[idxs]


def evaluate(models, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param models: Modules
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.

    total = 0
    correct = 0

    for model in models:
        model.eval()

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        outputs = []
        
        for model in models:
            outputs.append(model(inputs.to(DEVICE)).cpu())

        
        outputs = torch.stack(outputs)
        output = torch.mean(outputs, dim=0)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1

    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    lamb = 0.
    lr = 1e-2
    num_epoch = 40

    models = []

    for i in range(3):
        print('model', i)
        model = AutoEncoder(num_question=1774, k=100)
        model.to(DEVICE)
        train_matrix_bstrap, zero_train_matrix_bstrap = bootstrap(
            train_matrix, zero_train_matrix)

        train_losses, val_accs = train(
            model, lr, lamb, train_matrix_bstrap, zero_train_matrix_bstrap,
            valid_data, num_epoch)

        models.append(model)

    print('Val acc:', evaluate(models, zero_train_matrix, valid_data))
    print('Test acc:', evaluate(models, zero_train_matrix, test_data))


if __name__ == '__main__':
    main()
