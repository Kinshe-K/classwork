#import clf
#from methods import sliding_window, orb, histmean60, pilsum2, percenterrorrms
#from glob import glob
#from progressbar import progressbar as pb
#from sklearn.neural_network import MLPClassifier
#import pickle
#import random

from methods import sliding_window, orb, histmean60, pilsum2, percenterrorrms
from glob import glob
from progressbar import progressbar as pb
from sklearn.neural_network import MLPClassifier
import pickle
import random
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

with open("my_neuralnetwork.pkl", 'rb') as nn:
    detection_nn = pickle.load(nn)

# Create an MLPClassifier object
clf = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=100, alpha=0.0001,
                    solver='adam', verbose=10, random_state=42, tol=0.0001)

def load_data_from_directory(directory):
    data = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):  # Assuming the images are in PNG format
            img = load_img(os.path.join(directory, filename), target_size=(64, 64))  # Adjust target_size as needed
            img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
            flattened_img_array = img_array.flatten()  # Flatten the image array
            data.append(flattened_img_array)
            if "match" in filename:  # Assuming file names indicate matches or mismatches
                labels.append(1)  # Positive label
            else:
                labels.append(0)  # Negative label
    return np.array(data), np.array(labels)

# Load and preprocess training data
X_train, y_train = load_data_from_directory("train")
# Train the classifier using the training data
clf.fit(X_train, y_train)
    
def sum_confidence(p1, p2):
    results = [ sliding_window.compare_prints(p1, p2),
                orb.compare_prints(p1,p2), 
                histmean60.compare_prints(p1,p2),
                pilsum2.compare_prints(p1, p2),
                percenterrorrms.compare_prints(p1, p2) ]
    #print(sum(results))
    return(sum(results)>=3.5)

def simple_vote(p1, p2):
    results = [ sliding_window.compare_prints(p1, p2),
                orb.compare_prints(p1,p2), 
                histmean60.compare_prints(p1,p2),
                pilsum2.compare_prints(p1, p2),
                percenterrorrms.compare_prints(p1, p2) ]
    
    votes = [ round(conf) for conf in results]
    return(sum(votes)>3)

def nn_weighted(p1, p2):
    features = []
    features.append(sliding_window.compare_prints(p1, p2))
    features.append(orb.compare_prints(p1, p2))
    features.append(histmean60.compare_prints(p1, p2))
    features.append(pilsum2.compare_prints(p1, p2))
    features.append(percenterrorrms.compare_prints(p1, p2))
    
    # Concatenate the feature vectors into a single feature vector
    combined_features = np.concatenate(features)
    
    return bool(clf.predict([combined_features])[0])


def nn_thresh(p1, p2):
    results = [[ sliding_window.compare_prints(p1, p2),
                orb.compare_prints(p1,p2), 
                histmean60.compare_prints(p1,p2),
                pilsum2.compare_prints(p1, p2),
                percenterrorrms.compare_prints(p1, p2) ]]
    confidence = clf.predict_proba(results)[0][0]
    print(confidence)

def gen_statistics(match_method, iters=50, debug = False):
    fnames = glob("test/f*.png")
    match = []
    mismatch = []
    for i in pb(range(iters)):
        f1_test = random.choice(fnames)
        f1_num = int(f1_test[6:10])
        f2_num = f1_num + random.randint(-1,1)
        f2_test_glob = "test/s" + str(f2_num).zfill(4) + "*.png"
        try:
            f2_test = glob(f2_test_glob)[0]
            match_val = match_method(f1_test, f2_test)
        except Exception as e: 
            continue
        if f1_num == f2_num:
            match.append(match_val)
        else:
            mismatch.append(match_val)
    #if debug:
        #print(f"Matches: {match}")
        #print(f"Mismatches: {mismatch}")
    false_reject = [ a for a in match if a is False ]
    false_accept = [ a for a in mismatch if a is True]
    false_reject_rate = (len(false_reject)/len(match))
    false_accept_rate = (len(false_accept)/len(mismatch))
    print(f"False acceptance rate: {false_accept_rate}")
    print(f"False reject rate: {false_reject_rate}")
    return((false_accept_rate, false_reject_rate))

nn_weighted("train/f0252_01.png", "train/s0252_01.png")
print("SUM CONFIDENCE")
for i in range(5): gen_statistics(sum_confidence)
print("SIMPLE VOTE")
for i in range(5): gen_statistics(simple_vote)
print("NEURAL NETWORK")
for i in range(5): gen_statistics(nn_weighted)