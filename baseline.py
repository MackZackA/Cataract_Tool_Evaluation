import math
import os
import sys
import random
import json
import pickle
import numpy as np
from statistics import stdev
from sklearn.svm import SVC

# DIRECTORY WHERE FULL SURGERY VIDEO FILES ARE LOCATED - MODIFY IF NECESSARY
vid_dir = "/home-4/zsong17@jhu.edu/QueryByVideo/query_by_video_data/videos/"
# DIRECTORY WHERE ANNOTATION FILES ARE LOCATED - MODIFY IF NECESSARY
anno_dir = "/home-4/zsong17@jhu.edu/QueryByVideo/query_by_video_data/annotations/tasks/"
# DIRECTORY WHERE TOOL ANNOTATIONS ARE STORED
tool_dir = "/home-4/zsong17@jhu.edu/work/zsong17/data/tool_annotations/"
# DIRECTORY WHERE PHASE CLIPS WILL BE WRITTEN - MODIFY IF NECESSARY
# phase_dir = "/home-4/zsong17@jhu.edu/work/zsong17/data/cataract_phase_separated/"
phase_dir = "/home-4/zsong17@jhu.edu/work/zsong17/data/baseline_svm/"
# DIRECTORY WHERE FEATURE MATRICES OF EACH VIDEO ARE STORED
vid_matrix_dir = "/home-4/zsong17@jhu.edu/work/zsong17/data/video_matrix_unseparated/"
# DIRECTORY WHERE TEN FOLDS OF DATA FOR THE BASELINE ARE STORED
baseline_data_dir = '/home-4/zsong17@jhu.edu/work/zsong17/data/baseline/'
GLOBAL_MAXIMUM = 11
MEAN = -1


def sampling_preparation():
    '''
    This function saves frame numbers of all phase clips into a dictionary
    and takes the global maximum and mean (total divided by 10).
    '''
    global GLOBAL_MAXIMUM, MEAN
    sum_length = 0
    frame_dict = {}
    video_matrices = os.listdir(vid_matrix_dir)
    print("Start preparation:")
    for matrix in video_matrices:
        vid_id = matrix[-7:-4]
        full_name = vid_matrix_dir + matrix
        read_matrix = np.load(full_name)
        length = read_matrix.shape[0]
        frame_dict[vid_id] = length
        sum_length += length
    MEAN = sum_length / 10
    print("MEAN = {}".format(MEAN))
    return frame_dict

def sampling():
    '''
    This function will sample the 100 videos 100,000 times and pick 
    the distribution with lowest standard deviation.

    Return: a list of 10 sublists of IDs.
    '''
    stand_dev = float('inf')
    frame_dict = sampling_preparation()
    best_distribution = None
    best_ids = None
    for i in range(10000000):
        samples = random.sample(frame_dict.items(), 100)
        ids, lengths = list(zip(*samples))
        cluster_lengths = [sum(lengths[x:x+10]) for x in range(0, len(lengths), 10)]
        value = stdev(lengths)
        if value < stand_dev:
            best_distribution = samples
            best_ids = ids
            stand_dev = value
        if i % 100000 == 0:
            print("Booooooo")
    print("Sampling done.\n {}\nIDs: {}".format(stand_dev, ids))
    with open('best_split.json', 'w') as outfile:
        json.dump(ids, outfile)
    print("Save samples to 'best_split.json'.")

def count_global():
    global GLOBAL_MAXIMUM
    phases = os.listdir(phase_dir)
    for phase_matrix in phases:
        read_matrix = np.load(phase_dir + phase_matrix)
        if read_matrix.shape[0] == 0:
            continue
        uniques, indices = np.unique(read_matrix, axis=0, return_index=True)
        uniques_count = len(uniques)
        if uniques_count > GLOBAL_MAXIMUM:
            GLOBAL_MAXIMUM = uniques_count 
    print("GLOBAL_MAXIMUM = {}".format(GLOBAL_MAXIMUM)) 

def extract_example_frames():
    data = []
    ###
    for i in range(1, 11):
        fold_dir = baseline_data_dir + str(i) + "/"
        phases = os.listdir(fold_dir)
        for phase_matrix in phases:
            if phase_matrix == "data.json":
                continue
            label = phase_matrix[-6:-4] 
            read_matrix = np.load(fold_dir + phase_matrix)
            length = read_matrix.shape[0]
            if read_matrix.shape[0] == 0:
                continue
            uniques = np.unique(read_matrix, axis=0).tolist()
            uniques_count = len(uniques)
            for uniq in uniques:
                data.append((uniq, label))
            # print(length, GLOBAL_MAXIMUM - uniques_count)
            random_indices = random.choices(range(0, length), k=GLOBAL_MAXIMUM-uniques_count)
            for idx in random_indices:
                data.append((read_matrix[idx, :].tolist(), label))
        with open(fold_dir + 'data.json', 'w') as data_tuples:
            json.dump(data, data_tuples)
    ###
    print("Example Frames and labels for this fold are collected.")

def training_svm():
    # loading data from 8 folds for training, 1 for validating, and 1 for testing
    train_set = [] 
    for i in range(1, 9):
        data_path = baseline_data_dir + str(i) + '/data.json'
        data = json.load(open(data_path, 'r'))
        train_set += data
    print("Training set collected.")
    # training
    X, y = list(zip(*train_set))
    X = np.array(X)
    y = np.array(y)
    model = SVC()
    model.fit(X, y)
    pickle.dump(model, open('baseline.pickle', 'wb'))


def testing():
    # val_path = os.path.join(baseline_data_dir, str(9), '/data.json')
    # val_set = json.load(val_path)
    # validation
    # val_X, val_y = list(zip(*val_set))
    # val_X = np.array(val_X)
    # val_y = np.array(val_y)
    # model.predict() 
    # test
    test_path = baseline_data_dir + str(10) + '/data.json'
    test_set = json.load(open(test_path, 'r'))
    print('Validation, test sets collected.')
    test_X, test_y = list(zip(*test_set))
    test_X = np.array(test_X)
    test_y = np.array(test_y)
    model = pickle.load(open('baseline.pickle', 'rb'))
    performance = model.score(test_X, test_y)
    print(performance)
    print("Complete testing.")


if __name__ == "__main__":
    # sampling()
    # count_global()
    # extract_example_frames()
    # training_svm()
    testing()



