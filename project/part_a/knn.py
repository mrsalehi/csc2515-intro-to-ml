from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt

import sys
sys.path.append("..")

from utils import *

def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # print("Sparse matrix:")
    # print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    # fig, axes = plt.subplots(1)
    # fig.tight_layout()

    ks = [1, 6, 11, 16, 21, 26]
    accs = []

    # for i, k in enumerate([1, 6, 11, 16, 21, 26]):
    #     print(f'K={k},', end=' ')
    #     accs.append(knn_impute_by_user(sparse_matrix, val_data, k))
    
    # plt.plot(ks, accs)
    # plt.xlabel('k')
    # plt.ylabel('Acc')
    # plt.title('Validation Accuracy')

    # plt.savefig('fig-user.png')

    # test accuracy 
    # knn_impute_by_user(sparse_matrix, test_data, 11)

    plt.clf()

    # for i, k in enumerate([1, 6, 11, 16, 21, 26]):
    #     print(f'K={k},', end=' ')
    #     accs.append(knn_impute_by_item(sparse_matrix, val_data, k))

    # plt.plot(ks, accs)
    # plt.xlabel('k')
    # plt.ylabel('Acc')
    # plt.title('Validation Accuracy')

    # plt.savefig('fig-item.png')

    # test accuracy 
    # knn_impute_by_user(sparse_matrix, test_data, 21)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
