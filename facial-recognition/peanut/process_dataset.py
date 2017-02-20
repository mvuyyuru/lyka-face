# peanut - v 0.3

import os
import glob
import shutil
import random

print "peanut - v 0.3"

dataset = 'lfw'

base_path = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(base_path, 'img/{}/{}'.format(dataset, dataset))
dataset_dump = os.path.join(base_path, 'img/{}/train/dump'.format(dataset))
dataset_raw = os.path.join(base_path, 'img/{}/train/raw'.format(dataset))
dataset_test = os.path.join(base_path, 'img/{}/test/dump'.format(dataset))

minimum_images_count = 16
train_images_count = 15
# remainder will be test images

def thin_by_minimum_images():

    if not os.path.exists(dataset_dump):
        os.makedirs(dataset_dump)

    if not os.path.exists(dataset_raw):
        os.makedirs(dataset_raw)

    if not os.path.exists(dataset_test):
        os.makedirs(dataset_test)

    people = glob.glob(os.path.join(dataset_path, '*'))
    progress_count = 0.0
    for person in people:
        progress_count += 1
        print '{}%'.format(((progress_count)/(len(people)))*100)


        images = glob.glob(os.path.join(person, '*.*'))
        person_base = os.path.basename(os.path.normpath(person))

        if len(images) >= minimum_images_count:
            shutil.copytree(person, os.path.join(dataset_dump, person_base))
            train_images_set = random.sample(images, train_images_count)

            dataset_raw_person = os.path.join(dataset_raw, person_base)
            dataset_test_person = os.path.join(dataset_test, person_base)
            if not os.path.exists(dataset_raw_person):
                os.makedirs(dataset_raw_person)
            if not os.path.exists(dataset_test_person):
                os.makedirs(dataset_test_person)

            for image in train_images_set:
                shutil.copy(image, dataset_raw_person)

            for image in images:
                if image not in train_images_set:
                    shutil.copy(image, dataset_test_person)




raw_input("process images from {} dataset into train and test sets. press enter to continue ...".format(dataset))
thin_by_minimum_images()