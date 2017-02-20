# peanut - v 0.3

import os
import time
import pickle
import glob
import csv
import numpy as np
from skvideo.io import VideoWriter

# http://opencv.org/
import cv2

# https://github.com/cmusatyalab/openface
import openface

print "peanut - v 0.3"

video = 'lfw'

base_path = os.path.dirname(os.path.realpath(__file__))
classifier_pickle = os.path.join(base_path, 'features/{}/classifier.pkl'.format(video))
dlib_face_predictor = os.path.join(base_path, 'models/dlib/shape_predictor_68_face_landmarks.dat')
openface_network = os.path.join(base_path, 'models/openface/nn4.small2.v1.t7')
img_employ_dump_path = os.path.join(base_path, 'img/{}/employ/dump'.format(video))
img_employ_dump_path_relative = 'img/{}/employ/dump'.format(video)
img_employ_check_path = os.path.join(base_path, 'img/{}/employ/check'.format(video))
img_employ_check_path_relative = 'img/{}/employ/check'.format(video)
vid_employ_path = os.path.join(base_path, 'vid/{}/{}-test.mp4'.format(video, video))
vid_output = 'vid/{}/{}-oracle.mp4'.format(video, video)
memory_dump_path = os.path.join(base_path, 'memories/{}'.format(video))
memory_dump = os.path.join(memory_dump_path, 'memory.csv')
img_test_dump = os.path.join(base_path, 'img/{}/test/dump'.format(video))
img_test_check_path = os.path.join(base_path, 'img/{}/test/check'.format(video))
img_test_check_path_relative = 'img/{}/test/check'.format(video)

aligned_image_size = 96
chain = True
predictions = []

align = openface.AlignDlib(dlib_face_predictor)
net = openface.TorchNeuralNet(openface_network, aligned_image_size)
with open(classifier_pickle, 'r') as f:
        (le, clf) = pickle.load(f)


def employ_classifier(img):
    print("\n=== {} ===".format(img))
    net_out = getRep(img)
    if type(net_out) is not int:
        rep = net_out.reshape(1, -1)
        start = time.time()
        predictions = clf.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)
        person = le.inverse_transform(maxI)
        confidence = predictions[maxI]
        print("Prediction took {} seconds.".format(time.time() - start))
        print("Predict {} with {:.2f} confidence.".format(person, confidence))
        return person, confidence
        # if isinstance(clf, GMM):
        #     dist = np.linalg.norm(rep - clf.means_[maxI])
        #     print("  + Distance from the mean: {}".format(dist))
    else:
        return 'err', 0


def getRep(imgPath):
    start = time.time()
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        print ("Unable to load image: {}".format(imgPath))
        return 0

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    print("  + Original size: {}".format(rgbImg.shape))
    print("Loading the image took {} seconds.".format(time.time() - start))

    start = time.time()

    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        print ("Unable to find a face: {}".format(imgPath))
        return 0
    print("Face detection took {} seconds.".format(time.time() - start))

    start = time.time()
    alignedFace = align.align(aligned_image_size, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if alignedFace is None:
        print ("Unable to align image: {}".format(imgPath))
        return 0
    print("Alignment took {} seconds.".format(time.time() - start))

    start = time.time()
    rep = net.forward(alignedFace)
    print("Neural network forward pass took {} seconds.".format(time.time() - start))
    return rep


def identify_faces_image():
    people = glob.glob(os.path.join(img_test_dump, '*'))
    print people
    for person in people:
        person_base = os.path.basename(os.path.normpath(person))
        test_images = glob.glob(os.path.join(person, '*.*'))
        for image in test_images:
            print("\n=== {} ===".format(image))
            net_out = getRep(image)
            if type(net_out) is not int:
                rep = net_out.reshape(1, -1)
                start = time.time()
                predictions = clf.predict_proba(rep).ravel()
                maxI = np.argmax(predictions)
                person = le.inverse_transform(maxI)
                confidence = predictions[maxI]
                print("Prediction took {} seconds.".format(time.time() - start))
                print("Predict {} with {:.2f} confidence.".format(person, confidence))
            else:
                print("Prediction failed.")
                person = 'err'
                confidence = '1'

            check_label = '{},{}'.format(person, confidence)
            person_check_path = os.path.join(img_test_check_path_relative, person_base)
            if not os.path.exists(person_check_path):
                os.makedirs(person_check_path)
            tmp_image = cv2.imread(image)
            if not cv2.imwrite(person_check_path + '/{}.png'.format(check_label), tmp_image):
                print "Failed to write {}.png".format(check_label)


def identify_faces_video():
    stream = cv2.VideoCapture(vid_employ_path)
    video_writer = VideoWriter(vid_output, frameSize=(int(stream.get(3)), int(stream.get(4))))
    video_writer.open()
    if not os.path.exists(memory_dump_path):
        os.makedirs(memory_dump_path)

    if not os.path.exists(img_employ_dump_path):
        os.makedirs(img_employ_dump_path)

    if not os.path.exists(img_employ_check_path):
        os.makedirs(img_employ_check_path)

    while stream.isOpened():

        ret, frame = stream.read()
        if ret:
            bounding_boxes = align.getAllFaceBoundingBoxes(frame)

            for bounding_box in bounding_boxes:
                bounding_box_convert = frame[bounding_box.top():bounding_box.bottom() + 1, bounding_box.left():bounding_box.right() + 1]
                label = '{}-{},{},{},{}'.format(int(stream.get(1)), bounding_box.top(), bounding_box.bottom() + 1, bounding_box.left(), bounding_box.right() + 1)
                if not cv2.imwrite(img_employ_dump_path_relative + '/{}.png'.format(label), bounding_box_convert):
                    print "Failed to write {}.png".format(label)

                guess_person, guess_confidence = employ_classifier(os.path.join(img_employ_dump_path, '{}.png'.format(label)))

                check_label = '{},{}'.format(guess_person, guess_confidence)
                if not cv2.imwrite(img_employ_check_path_relative + '/{}.png'.format(check_label), bounding_box_convert):
                    print "Failed to write {}.png".format(check_label)
				
                cv2.rectangle(frame, (bounding_box.left(), bounding_box.top()),
                              (bounding_box.right() + 1, bounding_box.bottom() + 1), (255, 0, 0), 2)
                cv2.putText(frame, '{}'.format(guess_person),
                            (bounding_box.left(), bounding_box.bottom() + 1), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
                cv2.putText(frame, '{0:.5f}'.format(guess_confidence),
                            (bounding_box.left(), bounding_box.top()), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
                predictions.append([stream.get(1), stream.get(7), guess_person, guess_confidence, bounding_box.left(), bounding_box.top(), bounding_box.right(), bounding_box.bottom()])				

            print "==========\n{}%\n==========".format((stream.get(1)/stream.get(7))*100)
            # cv2.imshow('vid', frame)
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    with open(memory_dump, "wb") as memory_csv:
        writer = csv.writer(memory_csv)
        writer.writerows(predictions)
    stream.release()
    video_writer.release()

raw_input("employ classifier to recognize {}. press enter to continue ...".format(video))
# identify_faces_video()
identify_faces_image()
