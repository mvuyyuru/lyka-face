# peanut - v 0.3

import os

# http://opencv.org/
import cv2

# https://github.com/cmusatyalab/openface
import openface

print "peanut - v 0.3"

video = 'fri'

base_path = os.path.dirname(os.path.realpath(__file__))
vid_train_path = os.path.join(base_path, 'vid/{}/{}-train.avi'.format(video, video))
img_train_dump_path = os.path.join(base_path, 'img/{}/train/dump'.format(video))
img_train_dump_path_relative = 'img/{}/train/dump'.format(video)
dlib_face_predictor = os.path.join(base_path, 'models/dlib/shape_predictor_68_face_landmarks.dat')


def extract_faces_dlib(vid_path):
    stream = cv2.VideoCapture(vid_path)
    align = openface.AlignDlib(dlib_face_predictor)
    dump_count = 0

    if not os.path.exists(img_train_dump_path):
        os.makedirs(img_train_dump_path)

    while stream.isOpened():

        ret, frame = stream.read()
        if ret:
            bounding_boxes = align.getAllFaceBoundingBoxes(frame)

            for bounding_box in bounding_boxes:
                # bounding_box_resized = cv2.resize(frame[bounding_box.top():bounding_box.bottom() + 1,
                #                   bounding_box.left():bounding_box.right() + 1], (96, 96),
                #                   interpolation=cv2.INTER_CUBIC)
                if not cv2.imwrite(img_train_dump_path_relative + '/{}.png'.format(dump_count), frame[bounding_box.top():bounding_box.bottom() + 1, bounding_box.left():bounding_box.right() + 1]):
                    print "Failed to write {}.png".format(dump_count)
                dump_count += 1

            # cv2.imshow('vid', frame)
            print "{}%".format((stream.get(1)/stream.get(7))*100)
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    stream.release()

raw_input("extract faces from {}. press enter to continue ...".format(video))
extract_faces_dlib(vid_train_path)
