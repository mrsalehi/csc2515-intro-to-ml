'''
Question 2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from math import pi

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for k in range(train_labels):
        digits_k = data.get_digits_by_label(train_data, train_labels, k)
        means[k] = np.sum(digits_k, axis=0) / len(digits_k)
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    for k in range(train_labels):
        digits_k = data.get_digits_by_label(train_data, train_labels, k)
        covariances[k] = (1 / len(digits_k)) * (digits_k.T @ digits_k)
        covariances[k] += np.eye(64) * 0.01

    return covariances

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    n = len(digits)
    log_likelihoods = np.zeros((n, 10))
    for k in range(10):
        log_const_term = -32 * np.log(2 * pi) - 0.5 * np.log(np.linalg.det(covariances[k])) 
        dev = digits - means[k]  # shape: (n ,64)
        log_exp_term = -0.5 * np.sum((dev  @ covariances[k]) * dev.T, axis=1)   # shape: (n,)
        log_likelihoods[:, k] = log_const_term + log_exp_term
    
    return log_likelihoods

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    cond_ll = np.zeros((n, 10))
    cond_ll = np.logsumexp(generative_likelihood, 1/10)
    cond_ll /= np.sum(cond_ll, axis=1, keepdims=True)

    return cond_ll

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    cond_likelihood = cond_likelihood[:, label]
    # Compute as described above and return
    return np.mean(cond_likelihood)

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    return np.argmax(cond_likelihood, axis=1)

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation
    preds_train = classify_data(train_data, means, covariances)
    preds_test = classify_data(test_data, means, covariances)

    print('Train acc: %.2f' % (np.sum(preds_train == train_labels) / len(train_data)) * 100)
    print('Test acc: %.2f' % (np.sum(preds_test == test_labels) / len(test_data)) * 100)

if __name__ == '__main__':
    main()
