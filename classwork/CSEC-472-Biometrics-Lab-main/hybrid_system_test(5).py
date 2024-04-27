from methods import sliding_window, orb, histmean60, pilsum2, percenterrorrms
from glob import glob
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
import pickle
import random
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

with open("my_neuralnetwork.pkl", 'rb') as nn:
    detection_nn = pickle.load(nn)

# Create an MLPClassifier object
clf = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=500, alpha=0.0001,
                    solver='adam', verbose=10, random_state=42, tol=0.0001)

def load_data_from_directory(directory):
    data = []
    labels = []
    filenames = os.listdir(directory)
    for filename in filenames:
        if filename.endswith(".png"):  
            if filename.startswith("f"):  # Check if the file is a fingerprint image (fXXX.png)
                matching_filename = "s" + filename[1:]  # Form the corresponding matching filename (sXXX.png)
                if matching_filename in filenames:  # Check if the matching file exists in the directory
                    img1 = load_img(os.path.join(directory, filename), target_size=(64, 64))
                    img2 = load_img(os.path.join(directory, matching_filename), target_size=(64, 64))
                    img_array1 = img_to_array(img1) / 255.0  
                    img_array2 = img_to_array(img2) / 255.0  
                    flattened_img_array1 = img_array1.flatten()  
                    flattened_img_array2 = img_array2.flatten()  
                    data.append((flattened_img_array1, flattened_img_array2))
                    labels.append(1)  # Positive label for matching pair
                else:
                    print(f"Matching file not found for {filename}")
            elif filename.startswith("s"):  # Check if the file is a non-matching fingerprint image (sXXX.png)
                img = load_img(os.path.join(directory, filename), target_size=(64, 64))
                img_array = img_to_array(img) / 255.0  
                flattened_img_array = img_array.flatten()  
                data.append((flattened_img_array, flattened_img_array))  # Use the same image for comparison
                labels.append(0)  # Negative label for non-matching pair
            else:
                print(f"Ignoring file: {filename} - Filename format unrecognized")
    return np.array(data), np.array(labels)

# Load and preprocess training data
X_train_pairs, y_train = load_data_from_directory("train")
# Reshape X_train_pairs to have only two dimensions
X_train = X_train_pairs.reshape(X_train_pairs.shape[0], -1)
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
    
    # Check each feature array
    for feature in features:
        if np.ndim(feature) == 0:  # Check if the array is zero-dimensional
            return False  # Return False if any feature is zero-dimensional
    
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

def gen_statistics(match_method, iters=50, debug = True):
    fnames = glob("test/f*.png")
    match = []
    mismatch = []
    for i in tqdm(range(iters)):
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
    if debug:
        print(f"Matches: {match}")
        print(f"Mismatches: {mismatch}")
    false_reject = [ a for a in match if a is False ]
    false_accept = [ a for a in mismatch if a is True]
    false_reject_rate = (len(false_reject)/len(match))
    false_accept_rate = (len(false_accept)/len(mismatch))
    print(f"False acceptance rate: {false_accept_rate}")
    print(f"False reject rate: {false_reject_rate}")
    return((false_accept_rate, false_reject_rate))

nn_weighted("train/f0252_01.png", "train/s0252_01.png")

# Print SUM CONFIDENCE results
print("SUM CONFIDENCE")
for i in range(1):
    false_accept_rate, false_reject_rate = gen_statistics(sum_confidence)
    print(f"False acceptance rate: {false_accept_rate}")
    print(f"False reject rate: {false_reject_rate}")

# Print SIMPLE VOTE results
print("SIMPLE VOTE")
for i in range(1):
    false_accept_rate, false_reject_rate = gen_statistics(simple_vote)
    print(f"False acceptance rate: {false_accept_rate}")
    print(f"False reject rate: {false_reject_rate}")

# Print NEURAL NETWORK results
print("NEURAL NETWORK")
for i in range(1):
    false_accept_rate, false_reject_rate = gen_statistics(nn_weighted)
    print(f"False acceptance rate: {false_accept_rate}")
    print(f"False reject rate: {false_reject_rate}")