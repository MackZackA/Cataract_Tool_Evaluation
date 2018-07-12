#########################################
# Johns Hopkins University              #
# Spring 2018                           #
# Ziang Song                            #
# JHED: zsong17                         #
#########################################

import cv2
import sys
import os
import math
import numpy as np

# DIRECTORY WHERE FULL SURGERY VIDEO FILES ARE LOCATED - MODIFY IF NECESSARY
vid_dir = "/home-4/zsong17@jhu.edu/QueryByVideo/query_by_video_data/videos/"
# DIRECTORY WHERE ANNOTATION FILES ARE LOCATED - MODIFY IF NECESSARY
anno_dir = "/home-4/zsong17@jhu.edu/QueryByVideo/query_by_video_data/annotations/tasks/"
# DIRECTORY WHERE TOOL ANNOTATIONS ARE STORED
tool_dir = "/home-4/zsong17@jhu.edu/work/zsong17/data/tool_annotations/"
# tool_dir = "/home-4/zsong17@jhu.edu/QueryByVideo/query_by_video_data/annotations/tools/"

##### Not used outside segment_video()
# DIRECTORY WHERE PHASE CLIPS WILL BE WRITTEN - MODIFY IF NECESSARY
phase_dir = "/home-4/zsong17@jhu.edu/work/zsong17/data/cataract_phase_separated/"

# DIRECTORY WHERE FEATURE MATRICES OF EACH VIDEO WILL BE WRITTEN
vid_matrix_dir = "/home-4/zsong17@jhu.edu/work/zsong17/data/video_matrix_unseparated/"

# DIRECTORTY WHERE PHASE ENCODINGS ARE STORED
# phase_encoding_dir = "/home-4/zsong17@jhu.edu/work/zsong17/data/phase_matrices/"
# phase_encoding_dir = "/home-4/zsong17@jhu.edu/work/zsong17/data/baseline_svm/" 
phase_encoding_dir = "/home-4/zsong17@jhu.edu/work/zsong17/data/baseline/raw/"

NUM_TOOLS = 14
NUM_PHASES = 10
tool_indices = {'01':0,
               '02':1,
               '04':2,
               '05':3,
               '07':4,
               '08':5,
               '09':6,
               '10':7,
               '11':8,
               '12':9,
               '13':10,
               '16':11,
               '20':12,
               '25':13,    
               }
fold_1 = ['262', '138', '280', '621', '123', '051', '330', '121', '034', '015']
fold_2 = ['327', '181', '120', '286', '124', '158', '258', '290', '293', '353']
fold_3 = ['270', '085', '348', '303', '162', '314', '042', '052', '149', '325']
fold_4 = ['324', '255', '375', '169', '366', '267', '208', '173', '351', '037']
fold_5 = ['119', '137', '365', '217', '341', '274', '077', '039', '235', '302'] 
fold_6 = ['518', '128', '212', '147', '279', '127', '296', '504', '558', '278'] 
fold_7 = ['371', '220', '332', '224', '249', '062', '157', '321', '282', '146'] 
fold_8 = ['252', '026', '342', '117', '159', '014', '161', '148', '130', '038'] 
fold_9 = ['053', '248', '008', '133', '334', '243', '155', '320', '359', '058'] 
fold_10 = ['313', '150', '023', '170', '122', '145', '175', '118', '154', '055']

def extract_tool_frames():
    '''
    This function reads each video from vid_dir, creates a frame-tool matrix 
    for that video and initializes values based on the legend (start or end) of its
    associate tools. It will then save all feature matrices into vid_matrix_dir.
    '''
    tool_annotations = os.listdir(tool_dir)
    existing_videos = os.listdir(vid_dir)
    feature_matrix = None
    length = None
    fps = None
    missing_number = []
    processed_count = 0
    for tool in tool_annotations:
        chrono_frame_list = [] # key: frame, value: tool label
        frame_dict = {} # key: tool label, value: frame 
        ta_num = tool[-7:-4]
        print("Processing video number: {}".format(ta_num))
        # Obtain frame number
        video_name = 'vid_' + ta_num + '.mp4'
        video_ext = vid_dir + video_name
        if video_name not in existing_videos:
            print("Video under this number is missing.")
            missing_number.append(ta_num)
            continue
        processed_count += 1
        cap = cv2.VideoCapture(video_ext)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) 
        feature_matrix = np.zeros((length, NUM_TOOLS), dtype=int)
        cap.release()

        # obtaining phases 060-061
        index_change = []
        encoding_ext = anno_dir + 'vid_' + ta_num + '.txt'
        # print("Line 85: {}".format(encoding_ext))
        # load_encoding = np.load(encoding_ext)
        load_encoding = open(encoding_ext, 'r')
        rows = load_encoding.readlines()
        line_count = len(rows)
        if line_count % 2 != 0:
            line_count -= 1
        for i in range(0, line_count, 2):
            line1_split = rows[i].strip().split()
            line2_split = rows[i + 1].strip().split()
            t1 = int(math.ceil(float(line1_split[0]) * fps))
            t2 = int(math.ceil(float(line2_split[0]) * fps))
            phase = line1_split[1][:-1]
            if phase == '06':
                index_change.append((t1, t2))
        # Initialize feature matrice
        tool_file = open(tool_dir + tool, 'r')
        lines = tool_file.readlines()
        for j in range(len(lines)):
            line_list = lines[j].strip().split()
            if not line_list:
                continue
            # print("Print line_list:", line_list)
            # if ta_num == '161':
            #     print("Line 109:", line_list)
            time, tool_label = tuple(line_list)
            time = float(time)
            frame_num = math.ceil(time * fps)
            # index = int(tool_label[:-1]) - 1
            index = None
            # print("Line 116: tool_label = {}".format(tool_label))
            if tool_label[:-1] in tool_indices:
                index = tool_indices[tool_label[:-1]]
                # print("Line 119: Caught!!!!!!!")
                last_digit = int(tool_label[-1])
                if last_digit == 0:
                    '''
                    if tool_label != '180':
                        feature_matrix[frame_num:, index] = 1
                    else:
                        feature_matrix[frame_num:, index] = 0
                    '''
                    feature_matrix[frame_num:, index] = 1
                else:
                    feature_matrix[frame_num:, index] = 0
            ###### dealing with tool 170 within any phase 6
            elif tool_label[:-1] == '17':
                # print("Line 132: Caught!!!!!!!")
                for i in index_change:
                    if time > i[0] and time < i[1]:
                        last_digit = tool_label[-1]
                        tool_label = '10' + last_digit
                        index = tool_indices[tool_label[:-1]]
                        if int(last_digit) == 0:
                            feature_matrix[frame_num:, index] = 1
                        else:
                            feature_matrix[frame_num:, index] = 0
            # else:
            #    print("Line 143: Didn't catch anything.")        
            #####################################################
        save_address = vid_matrix_dir + "vid_" + ta_num + ".npy"  
        np.save(save_address, feature_matrix)
        print("Sample:", feature_matrix[60:65, :])
        print("Shape: {}".format(feature_matrix.shape))
        print("Save video matrix as {}".format(save_address))
        feature_matrix = None
    print("Encodings for full videos complete.\nNumber of files processed: {}\nMissing {} videos are: \n{}\n".format(processed_count, len(missing_number), missing_number))
 

def extract_phase_matrices():
    '''
    This function accesses video matrices stored by extract_tool_frames() 
    and segments them by phase.
    The outputs are files with chunked matrices and their respective phases.`
    '''
    global phase_encoding_dir
    phase_encoding_dir = "/home-4/zsong17@jhu.edu/work/zsong17/data/baseline/10/" #4. uncomment to use absolute path
    unseparated_encodings = os.listdir(vid_matrix_dir)
    phase_annotations = os.listdir(anno_dir)
    existing_videos = os.listdir(vid_dir)
    length = None
    fps = None
    # legends = [] # list of tuples of beginnings and ends
    missing_files = []
    # for encoding in unseparated_encodings: #1. uncomment to use absolute path
    for encoding in fold_10:
        # encoding_num = encoding[-7:-4] #2. uncomment to use absolute path
        encoding_num = encoding
        print("Current encoding number is: {}".format(encoding_num))
        anno_fn = "vid_" + encoding_num + ".txt"
        annotation_ext = anno_dir + anno_fn
        video_fn = 'vid_' + encoding_num + '.mp4'
        video_ext = vid_dir + video_fn
        if anno_fn not in phase_annotations:
            # raise ValueError("Annotation file under this number is missing")
            print("Annotation file under this number is missing")
            missing_files.append(encoding_num)
            continue
        if video_fn not in existing_videos:
            # raise ValueError("Video under this number is missing")
            print("Video under this number is missing")
            missing_files.append(encoding_num)
            continue
        # encoding_ext = vid_matrix_dir + encoding #3. uncomment to use absolute path
        encoding_ext = vid_matrix_dir + "vid_" + encoding_num + ".npy"
        load_encoding = np.load(encoding_ext)
        cap = cv2.VideoCapture(video_ext)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        clip_count = 1
        annotation_file = open(annotation_ext, 'r')
        lines = annotation_file.readlines()
        line_count = len(lines)
        if line_count % 2 != 0:
            line_count -= 1
        for i in range(0, line_count, 2):
            line1_split = lines[i].strip().split()
            line2_split = lines[i + 1].strip().split()
            t1 = int(math.ceil(float(line1_split[0]) * fps))
            t2 = int(math.ceil(float(line2_split[0]) * fps))
            # phase = int(line1_split[1][:-1])
            phase = line1_split[1][:-1]
            # legends.append((t1, t2, phase))
            phase_encoding = load_encoding[t1:t2, :]
            # print("Shape of Phase Encoding: {}".format(phase_encoding.shape))
            filename = "vid_" + encoding_num + "_" + str(clip_count) + "_p_" + phase
            path = phase_encoding_dir + filename
            np.save(path, phase_encoding)
            clip_count += 1
    print("\nNumbers of missing files: {}".format(missing_files)) 
    print("Phase matrices collected.\n")


def segment_vid():
    """
    Read each video from vid_dir and segment it into phases based on corresponding
    annotations in anno_dir. Write out resulting phase clips into phase dir under
    directory corresponding to the phase (1 to 10).
    In this script, annotation format is:
    time  legend  phase
    where time is a time in seconds, legend is start or end and phase is the
    corresponding phase
    """
    annotations = os.listdir(anno_dir)
    existing_videos = os.listdir(vid_dir)

    for annotation in annotations:
        vid_num = annotation[-7:-4]
        print("Currently on video number: " + vid_num)
        sys.stdout.flush()
        annotation_file = open(anno_dir + annotation, 'r')

        cap = cv2.VideoCapture(vid_dir + 'vid_' + vid_num + '.mp4')
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        num_phase_vids = [0] * (NUM_PHASES + 1)
        cur_frame = 0
        dx = 320
        dy = 240

        cut_boundaries = list()
        lines = annotation_file.readlines()
        for i in range(0, len(lines), 2):
            line1_split = lines[i].strip().split()
            line2_split = lines[i + 1].strip().split()
            t1 = int(math.ceil(float(line1_split[0]) * fps))
            t2 = int(math.ceil(float(line2_split[0]) * fps))
            activity = int(line1_split[1][:-1])
            cut_boundaries.append((activity, t1, t2))
            print('{:d} {:d} {:d}'.format(activity, t1, t2))
        print("Done reading annotations")
        sys.stdout.flush()
        for triple in cut_boundaries:
            start_frame = triple[1]
            end_frame = triple[2]
            activity_num = triple[0]
            num_phase_vids[activity_num] += 1
            vid_name = phase_dir + str(activity_num) + "/p" + str(activity_num) + "_n" +\
                str(num_phase_vids[activity_num]) + '_vid_' + vid_num + '.avi'
            print("Will write phase " + str(activity_num))
            sys.stdout.flush()
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            '''
            video_out = cv2.VideoWriter(vid_name,
                                        fourcc,
                                        fps,
                                        (640, 480))
            '''
            video_out = cv2.VideoWriter(vid_name,
                                        fourcc,
                                        fps,
                                        (int(cap.get(3)), int(cap.get(4))))
            while cur_frame < start_frame:
                ret, img = cap.read()
                cur_frame += 1
                if not ret:
                    break
            while cur_frame < end_frame:
                ret, img = cap.read()
                if not ret:
                    break
                cur_frame += 1
                video_out.write(img)
            print("Done writing phase " + str(activity_num))
            sys.stdout.flush()
            video_out.release()
            print("Released " + vid_name)
            sys.stdout.flush()
        cap.release()


if __name__ == "__main__":
    # extract_tool_frames()
    extract_phase_matrices() 
