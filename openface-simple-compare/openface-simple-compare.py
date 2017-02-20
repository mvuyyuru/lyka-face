#estimates similarity between faces

import numpy as np
import os
#http://opencv.org/
import cv2
#https://github.com/cmusatyalab/openface
import openface
import time
import itertools
import argparse

base_path = os.path.dirname(os.path.realpath(__file__))
dlib_face_predictor_model_path = os.path.join(base_path, 'models/dlib/shape_predictor_68_face_landmarks.dat')
openface_network_model_path = os.path.join(base_path, 'models/openface/nn4.small2.v1.t7')

img_dimension = 96

parser = argparse.ArgumentParser()
parser.add_argument('images', type=str, nargs="+", help="Images to compare.")
input_arguments = parser.parse_args()

start_time = time.time()
align = openface.AlignDlib(dlib_face_predictor_model_path)
network = openface.TorchNeuralNet(openface_network_model_path, img_dimension)
print ("loading align and network model took {} seconds".format(time.time() - start_time))

#get representation by feeding img into neural net
def openface_network_forward_pass(img_path):

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise Exception("Unable to load image {}".format(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    start_time = time.time()
    face_bounding_box = align.getLargestFaceBoundingBox(img_rgb)
    if face_bounding_box is None:
        raise Exception("Unable to find face in image {}".format(img_path))
    print ("found face in {} in {} seconds").format(img_path, time.time() - start_time)

    start_time = time.time()
    aligned_face = align.align(img_dimension, img_rgb, face_bounding_box, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if aligned_face is None:
        raise Exception("Unable to align face in image {}".format(img_path))
    print ("aligned face in {} in {} seconds").format(img_path, time.time() - start_time)

    start_time = time.time()
    representation = network.forward(aligned_face)
    print ("openface forward pass took {} seconds".format(time.time()-start_time))
    return representation

for (image_1, image_2) in itertools.combinations(input_arguments.images, 2):
    distance = openface_network_forward_pass(image_1) - openface_network_forward_pass(image_2)
    squared_l2_distance = np.dot(distance, distance)
    print("Compare representations of {} with {}.".format(image_1, image_2))
    print(" -> " + "Squared l2 distance: {:0.3f}".format(squared_l2_distance))
