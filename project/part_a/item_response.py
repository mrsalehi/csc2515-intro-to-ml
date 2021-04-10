import sys
sys.path.append("..")

from utils import *

from scipy.special import logsumexp
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    
    for uid, qid, is_cor in zip(data['user_id'], data['question_id'], data['is_correct']):
        x = theta[uid] - beta[qid]
        log_lklihood += -logsumexp([0, -x]) if is_cor else -logsumexp([0, x])
        
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    grad_theta = np.zeros_like(theta)
    grad_beta = np.zeros_like(beta)

    for uid, qid, is_cor in zip(data['user_id'], data['question_id'], data['is_correct']):
        x = theta[uid] - beta[qid]
        #log_sig = -logsumexp([0, -x])
        sig = sigmoid(x)

        grad_theta[uid] += -(1 - sig) if is_cor else sig
        grad_beta[qid] += (1 - sig) if is_cor else -sig

    theta -= lr * grad_theta
    beta -= lr * grad_beta
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros((542,))
    beta = np.zeros((1774,))

    val_acc_lst = []
    train_lls = []
    val_lls = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_lls.append(neg_lld)
        neg_lld_val = neg_log_likelihood(val_data, theta=theta, beta=beta)
        val_lls.append(neg_lld_val)

        score = evaluate(data=val_data, theta=theta, beta=beta)
        
        if not val_acc_lst:
            val_acc_lst = [score]
        elif score > max(val_acc_lst):
            best_theta = theta.copy()
            best_beta = beta.copy()
        
        val_acc_lst.append(score)
        
        print("Iteration {}, NLLK: {} \t Score: {}".format(i, neg_lld, score))
        update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return best_theta, best_beta, train_lls, val_lls, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    best_theta, best_beta, train_lls, val_lls, val_acc_lst = irt(train_data, val_data, lr=1e-2, iterations=3)
    print('Val Acc:', max(val_acc_lst))

    plt.plot(train_lls, label="Train log-likelihood")
    plt.plot(val_lls, label='Val log-likelihood')
    plt.legend()
    plt.savefig('curve-itr.png')
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (c)                                                #
    #####################################################################
    print('Test Acc:', evaluate(data=test_data, theta=best_theta, beta=best_beta))


    # part d
    plt.clf()
    sample_betas = np.random.choice(best_beta, size=5, replace=False)
    
    for beta in sample_betas:
        probs = []
        for theta in sorted(best_theta):
            probs.append(sigmoid(theta - beta))
        
        plt.plot(list(sorted(best_theta)), probs, label=f'beta={beta}')
    
    plt.xlabel('theta')
    plt.ylabel('Probability')
    plt.title('Probability of correct answer given thetas and 5 sample betas')
    plt.legend()
    plt.savefig('partd.png')

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
