#########################################
# Johns Hopkins University              #
# 601.455 Computer Integrated Surgery 2 #
# Spring 2018                           #
# Query by Video For Surgical Activities#
# Felix Yu                              #
# JHED: fyu12                           #
# Gianluca Silva Croso                  #
# JHED: gsilvac1                        #
#########################################

import matplotlib
matplotlib.use("agg")

import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy import stats
from pylab import savefig
from sklearn.metrics import confusion_matrix
import json
import pickle
import sys

baseline_data_dir = '/home-4/zsong17@jhu.edu/work/zsong17/data/baseline/'

def print_error():
    """
    Prints usage instructions
    """
    print("ERROR! Improper command line input detected.")
    print("Usage for nearest neighbor analysis: python result_analysis.py <model> <num neighbors>")
    print("-"*40)
    print("Model legend")
    print("0: Temporal Averaging")
    print("1: GAP_RNN_TL")
    print("2: GAP_RNN_TOOL\n")
    return


def load_key(key_file_path, database_size):
    """
    Loads txt file for correspondence between database indexes and specific
    clips
    :param key_file_path: txt file displaying correspondence
    :param database_size: number of clip feature vectors in the database
    :return: two vectors with database_size elements.
    labels contains the activity of each clip, clip_names contains the names.
    Both indexed to match the indexing of the database.
    """
    text = open(key_file_path)
    clip_names = list()
    labels = np.zeros(database_size) 
    ind = 0

    for line in text:
        split_line = line.strip().split()
        labels[ind] = split_line[2]
        clip_names.append(split_line[0] + "_" + split_line[2] + "_" + split_line[1])
        ind += 1
    return labels, clip_names


def analysis(which_model, num_neighbors):
    """
    Using train.npy as database, analyses accuracy of model in test database,
    producing accuracy statistics, precision and recall for each phase, and a
    confusion matrix.
    Prediction is based on majority of num_neighbors nearest neighbors.
    If there is a tie, closest neighbor gets preference.
    Also outputs a text file that matches each phase clip from the test
    data with one most similar clip from the training data.
    :param which_model: Model under consideration. 0 for mean, 1 for GAP_TNN
    without tool labels, 2 for GAP_RNN with tool labels
    :param num_neighbors: number of neighbors used for classification
    """
    # DIRECTORY WHERE CONFUSION MATRIX AND MATCHINGS FILE WILL BE SAVED - MODIFY IF NECESSARY
    out_dir = '/home-2/fyu12@jhu.edu/work3/fyu/CATARACT/analysis/'
    # DIRECTORY CONTAINING NPY DATABASES - MODIFY IF NECESSARY
    base_dir = '/home-2/fyu12@jhu.edu/work3/fyu/CATARACT/databases/'
    if which_model == "0":
        model_type = "mean"
    elif which_model == "1":
        model_type = "GAP_RNN_TL"
    elif which_model == "2":
        model_type = "GAP_RNN_TOOL"
    out_file_name = model_type + '_' + str(num_neighbors) + '.txt'
    out_fig_name = model_type + "_" + str(num_neighbors) + '_test.png'
    out_file = open(out_dir + out_file_name, 'w')

    database_dir = base_dir + model_type + "/"
    train_database = np.load(database_dir + "train.npy")
    test_queries = np.load(database_dir + "test.npy")
    train_labels, train_clip_names = load_key(database_dir + "train_key.txt", train_database.shape[0])
    test_labels, test_clip_names = load_key(database_dir + "test_key.txt", test_queries.shape[0])

    nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree', metric='euclidean').fit(train_database)
    distances, indices = nbrs.kneighbors(test_queries)

    confusion_matrix = np.zeros((10, 10))
    success = 0
    failure = 0
    ind = 0
    for neighs in indices:
        out_file.write(str(test_clip_names[ind]) + "\t\t")
        for neigh in neighs:
            out_file.write(str(train_clip_names[neigh]) + "\t\t")
        out_file.write("\n")
        neigh_labels = train_labels[neighs]
        m = stats.mode(neigh_labels)
        if m[1] == 1:
            label_guess = int(train_labels[neighs[0]]) - 1
        else:
            label_guess = int(m[0][0]) - 1
        true_label = int(test_labels[ind]) - 1
        if true_label == label_guess:
            success += 1
        else:
            failure += 1
        confusion_matrix[true_label][label_guess] += 1
        ind += 1
    precision_str = "Precision:\t"
    recall_str = "Recall:\t\t"
    for i in range(confusion_matrix.shape[0]):
        TP = confusion_matrix[i, i]
        FP_with_TP = np.sum(confusion_matrix[:, i])
        FN_with_TP = np.sum(confusion_matrix[i, :])
        precision_str += "{:0.3f}\t".format(TP/FP_with_TP)
        recall_str += "{:0.3f}\t".format(TP/FN_with_TP)
    print(precision_str)
    print(recall_str)
    confusion_matrix = confusion_matrix/np.reshape(np.sum(confusion_matrix, axis=1), (confusion_matrix.shape[0], 1))
    confusion_matrix = np.nan_to_num(confusion_matrix)
    print_results(success, failure, confusion_matrix, model_type, num_neighbors, out_dir + out_fig_name)
    out_file.close()

def calculate_metrics():
    # testing
    test_path = baseline_data_dir + str(10) + "/data.json"
    test_set = json.load(open(test_path, "r"))
    test_X, test_y = list(zip(*test_set))
    test_X = np.array(test_X)
    test_y = np.array(test_y)
    model = pickle.load(open("baseline.pickle", "rb"))
    prediction = model.predict(test_X)
    conf_mat = confusion_matrix(test_y, prediction)
    # write confusion matrix
    precision_str = "Precision:\t"
    recall_str = "Recall:\t\t"
    model_type = "svm"
    out_dir = "/home-4/zsong17@jhu.edu/work/zsong17/query_by_video_code/Deliverables/utils/"
    out_fig_name = model_type + "_test.png" 
    out_file = open(out_dir + out_fig_name, 'w')
    for i in range(conf_mat.shape[0]):
        TP = conf_mat[i, i]
        FP_with_TP = np.sum(conf_mat[:, i])
        FN_with_TP = np.sum(conf_mat[i, :])
        precision_str += "{:0.3f}\t".format(TP / FP_with_TP)
        recall_str += "{:0.3f}\t".format(TP / FN_with_TP)
    print(precision_str)
    print(recall_str)
    conf_mat = conf_mat / np.reshape(np.sum(conf_mat, axis=1), (conf_mat.shape[0], 1))
    conf_mat = np.nan_to_num(conf_mat) 
    # modify print_results
    success = np.count_nonzero(test_y == prediction)
    failure = len(prediction) - success
    print_results(success, failure, conf_mat, model_type, out_dir + out_fig_name)
    out_file.close()


def print_results(success, failure, confusion_matrix, model_type, out_fig_name):
    """
    Saves the confusion matrix as a figure
    :param success: number of successfully classified examples
    :param failure: number of mistakes in classification
    :param confusion_matrix: NxN confusion matrix. N is the number of classes
    :param model_type: String with which model is being used
    :param num_neighs: number of nearest neighbors being considered
    :param out_fig_name: filename for the image to be saved as
    """
    axes = np.array(range(1, 11))
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xticks(np.arange(len(axes)))
    ax.set_yticks(np.arange(len(axes)))
    ax.set_xticklabels(axes)
    ax.set_yticklabels(axes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    ax.set_title("Confusion Matrix for {} with 14 tools".format(model_type))
    im = ax.imshow(confusion_matrix, cmap="YlGn")
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
           text = ax.text(j, i, '{:0.3f}'.format(confusion_matrix[i, j]),
                       ha="center", va="center")
    text = ax.text(-4, 0, 'Analysis of Results\n' + '_'*20)
    text = ax.text(-4, 0.7, 'Successes: \n{:d}'.format(success))
    text = ax.text(-4, 1.4, 'Failures: \n{:d}'.format(failure))
    text = ax.text(-4, 2.1, "Accuracy:\n {:0.3f}".format(float(success)/(float(success+failure))))
    # plt.show()
    savefig(out_fig_name, bbox_inches='tight')


if __name__ == "__main__":
    '''
    if len(sys.argv) < 3:
        print_error()
    else:
        model_used = sys.argv[1]
        num_neighbors_considered = eval(sys.argv[2])
        if not (model_used == "0" or model_used == "1" or model_used == "2"):
            print_error()
        else:            
            analysis(model_used, num_neighbors_considered)
    '''
    calculate_metrics() 
