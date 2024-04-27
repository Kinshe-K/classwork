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


with open("my_neuralnetwork.pkl", 'rb') as nn:
    detection_nn = pickle.load(nn)
    
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
    results = [[ sliding_window.compare_prints(p1, p2),
                orb.compare_prints(p1,p2), 
                histmean60.compare_prints(p1,p2),
                pilsum2.compare_prints(p1, p2),
                percenterrorrms.compare_prints(p1, p2) ]]
    return(bool(clf.predict(results)[0]))

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