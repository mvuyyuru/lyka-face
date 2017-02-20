import os
import glob
import time
import shlex
import pickle
import subprocess
import numpy as np
import pandas as pd
from operator import itemgetter

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
# from sklearn.mixture import GMM

# http://opencv.org/
import cv2

# https://github.com/cmusatyalab/openface
import openface
import openface.helper
from openface.data import iterImgs

base_path = os.path.dirname(os.path.realpath(__file__))
raw_faces_path = os.path.join(base_path, 'images/raw')
faces_dump_path = os.path.join(base_path, 'images/dump')
aligned_faces_path = os.path.join(base_path, 'images/aligned')
dlib_face_predictor = os.path.join(base_path, 'models/dlib/shape_predictor_68_face_landmarks.dat')
openface_torch_network = os.path.join(base_path, 'models/openface/nn4.small2.v1.t7')
generate_representations_cache = os.path.join(base_path, 'images/aligned/cache.t7')
generated_representation_labels = os.path.join(base_path, 'features/labels.csv')
generated_representation_representations = os.path.join(base_path, 'features/reps.csv')
classifier_pickle = os.path.join(base_path, 'features/classifier.pkl')

aligned_image_size = 96
align_images_with_multiple_faces = False
align_landmark_indices = openface.AlignDlib.OUTER_EYES_AND_NOSE  # or openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP


def align_faces_dlib():
    openface.helper.mkdirP(aligned_faces_path)

    images = list(iterImgs(raw_faces_path))

    align_dlib = openface.AlignDlib(dlib_face_predictor)

    for image in images:
        aligned_person_face_path = os.path.join(aligned_faces_path, image.cls)
        openface.helper.mkdirP(aligned_person_face_path)
        aligned_numbered_person_face_path = os.path.join(aligned_person_face_path, image.name)
        aligned_image_name = aligned_numbered_person_face_path + ".png"

        image_rgb = image.getRGB()
        if image_rgb is None:
            print("Unable to load image {}").format(image.name)
            aligned_image_rgb = None
        else:
            aligned_image_rgb = align_dlib.align(aligned_image_size, image_rgb, landmarkIndices=align_landmark_indices, skipMulti = align_images_with_multiple_faces)
            if aligned_image_rgb is None:
                print("Unable to align image {}").format(image.name)

        if aligned_image_rgb is not None:
            aligned_image_bgr = cv2.cvtColor(aligned_image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(aligned_image_name, aligned_image_bgr)


def generate_representations_openface():
    # if your dataset has changed, delete the cache file
    if os.path.isfile(generate_representations_cache):
        os.remove(generate_representations_cache)

    generate_representations_lua = subprocess.Popen(shlex.split('./batch-represent/main.lua -outDir features -data images/aligned'))
    generate_representations_lua.wait()


def train_classifier():
    print("Loading embeddings.")
    labels = pd.read_csv(generated_representation_labels, header=None).as_matrix()[:, 1]
    labels = map(itemgetter(1),
                 map(os.path.split,
                     map(os.path.dirname, labels)))  # Get the directory.
    embeddings = pd.read_csv(generated_representation_representations, header=None).as_matrix()
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


def infer_classifier():
    with open(classifier_pickle, 'r') as f:
        (le, clf) = pickle.load(f)

    images = glob.glob(os.path.join(faces_dump_path, '*.*'))
    align = openface.AlignDlib(dlib_face_predictor)
    net = openface.TorchNeuralNet(openface_torch_network, aligned_image_size)

    for img in images:
        print("\n=== {} ===".format(img))
        rep = getRep(img, align, net).reshape(1, -1)
        start = time.time()
        predictions = clf.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)
        person = le.inverse_transform(maxI)
        confidence = predictions[maxI]
        print("Prediction took {} seconds.".format(time.time() - start))
        print("Predict {} with {:.2f} confidence.".format(person, confidence))
        # if isinstance(clf, GMM):
        #     dist = np.linalg.norm(rep - clf.means_[maxI])
        #     print("  + Distance from the mean: {}".format(dist))


def getRep(imgPath, align, net):
    start = time.time()
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    print("  + Original size: {}".format(rgbImg.shape))
    print("Loading the image took {} seconds.".format(time.time() - start))

    start = time.time()

    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        raise Exception("Unable to find a face: {}".format(imgPath))
    print("Face detection took {} seconds.".format(time.time() - start))

    start = time.time()
    alignedFace = align.align(aligned_image_size, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if alignedFace is None:
        raise Exception("Unable to align image: {}".format(imgPath))
    print("Alignment took {} seconds.".format(time.time() - start))

    start = time.time()
    rep = net.forward(alignedFace)
    print("Neural network forward pass took {} seconds.".format(time.time() - start))
    return rep

align_faces_dlib()
generate_representations_openface()
train_classifier()
infer_classifier()
