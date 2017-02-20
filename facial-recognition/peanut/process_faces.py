# peanut - v 0.3

import os
import shlex
import subprocess

# http://opencv.org/
import cv2

# https://github.com/cmusatyalab/openface
import openface
from openface.data import iterImgs

print "peanut - v 0.3"

video = 'lfw'

base_path = os.path.dirname(os.path.realpath(__file__))
img_train_raw_path_relative = 'img/{}/train/raw'.format(video)
img_train_raw_path = os.path.join(base_path, 'img/{}/train/raw'.format(video))
img_train_aligned_path = os.path.join(base_path, 'img/{}/train/aligned'.format(video))
img_train_aligned_path_relative = 'img/{}/train/aligned'.format(video)
img_train_features_path_relative = 'features/{}'.format(video)
dlib_face_predictor = os.path.join(base_path, 'models/dlib/shape_predictor_68_face_landmarks.dat')
generate_representations_cache = os.path.join(base_path, 'images/aligned/cache.t7')

aligned_image_size = 96
align_images_with_multiple_faces = False
align_landmark_indices = openface.AlignDlib.OUTER_EYES_AND_NOSE  # or openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP


def rename_raw_faces():

    face_count = 0
    for person in os.listdir(img_train_raw_path_relative):
        raw_face_path_prefix = img_train_raw_path_relative + '/{}/'.format(person)
        for raw_face in os.listdir(img_train_raw_path_relative + '/{}'.format(person)):
            os.rename(raw_face_path_prefix + raw_face, raw_face_path_prefix + 'tmp-' + str(face_count) + '.png')
            face_count += 1

    for person in os.listdir(img_train_raw_path_relative):
        label_count = 1
        raw_face_path_prefix = img_train_raw_path_relative + '/{}/'.format(person)
        for raw_face in os.listdir(img_train_raw_path_relative + '/{}'.format(person)):
            os.rename(raw_face_path_prefix + raw_face, raw_face_path_prefix + '{}-'.format(person) + str(label_count) + '.png')
            label_count += 1

    print "renamed {} raw faces".format(face_count)


def align_faces_dlib():
    openface.helper.mkdirP(img_train_aligned_path)

    images = list(iterImgs(img_train_raw_path))

    align_dlib = openface.AlignDlib(dlib_face_predictor)

    for image in images:
        aligned_person_face_path = os.path.join(img_train_aligned_path, image.cls)
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
    batch_process_command = './batch-represent/main.lua -outDir ' + img_train_features_path_relative + ' -data ' + img_train_aligned_path_relative
    generate_representations_lua = subprocess.Popen(shlex.split(batch_process_command))
    generate_representations_lua.wait()

raw_input("process faces from {}. press enter to continue ...".format(video))
rename_raw_faces()
align_faces_dlib()
generate_representations_openface()
