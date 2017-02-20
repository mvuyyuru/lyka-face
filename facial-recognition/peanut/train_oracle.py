# peanut - v 0.3

import os
import pickle
import pandas as pd
from operator import itemgetter

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
# from sklearn.mixture import GMM

print "peanut - v 0.3"

video = 'lfw'

base_path = os.path.dirname(os.path.realpath(__file__))
generated_representation_labels = os.path.join(base_path, 'features/{}/labels.csv'.format(video))
generated_representations = os.path.join(base_path, 'features/{}/reps.csv'.format(video))
classifier_pickle = os.path.join(base_path, 'features/{}/classifier.pkl'.format(video))

def train_classifier():
    print("Loading embeddings.")
    labels = pd.read_csv(generated_representation_labels, header=None).as_matrix()[:, 1]
    labels = map(itemgetter(1),
                 map(os.path.split,
                     map(os.path.dirname, labels)))  # Get the directory.
    embeddings = pd.read_csv(generated_representations, header=None).as_matrix()
    label_encoder = LabelEncoder().fit(labels)
    labels_num = label_encoder.transform(labels)
    n_classes = len(label_encoder.classes_)
    print("Training for {} classes.".format(n_classes))
    clf = SVC(C=1, kernel='linear', probability=True)  # linear svm
    # clf = GMM(n_components=nClasses) #GMM

    clf.fit(embeddings, labels_num)

    print("Saving classifier to '{}'".format(classifier_pickle))
    with open(classifier_pickle, 'w') as f:
        pickle.dump((label_encoder, clf), f)

raw_input("train classifier with processed {}. press enter to continue ...".format(video))
train_classifier()