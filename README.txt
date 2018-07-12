Baseline Model and processing for Cataract Project

Data Preparation:
1. Pick 100 out of 105 videos.
Omit - vid_114, vid_125, vid_164, vid_292, vid_297

2. Obtain mean frame number from the total number of frames.
Alternative - use python-opencv2 to get the duration - see TK's utility for existing code

*** Doesn't this come after Steps 4 and 5?) *** 3. Extract video matrices and store them in folds 1 - 10 under subdirectory "full_video_matrices".

(Steps 4 and 5: Run the functions video_encoder.py separately for videos in each fold)

For each of 100 videos - 
	- compute sum of duration of phases 1-9 and 13 -- call this video duration
Compute mean of the 100 video durations

4. Sampling: Write a Python script to 
- set the random seed
- initialize "variance" as 10000000 or another large number
repeat 100,000 times --
	- randomly split them into 10 clusters, each contains 10 videos
	- compute sum of squared deviation from mean of the 100 video durations
	- if sum of squared deviation from mean of the 100 video durations < 		initialized large number
		- keep the 10 clusters/folds
	- else
		- continue
Output: a list of ten lists of video IDs.

5. With ten clusters sampled, use video_encoder.py to obtain the phase matrices. Store them in folds 1 - 10 under subdirectory "phases".

6. Obtain N, the global maximum of unique encodings in all phases.

7. For each phase, obtain N examples.
    N =  all unique encodings + the rest randomly sampled from the phase.
    Output: a json list of (encoding, phase label) pairs for each fold.

Model:
    Train Multi-Class SVM on 8 folds, with 1 fold as validation and 1 fold as test sets. Save the model as "baseline_svm.model".

Analysis:
    Substitute KNN model with the SVM model in analysis.py in util/ folder

