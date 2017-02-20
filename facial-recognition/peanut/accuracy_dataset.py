# peanut - v 0.3

import os
import glob

print "peanut - v 0.3"

dataset = 'lfw'

base_path = os.path.dirname(os.path.realpath(__file__))
img_check_path = os.path.join(base_path, 'img/{}/test/check'.format(dataset))

confidence_threshold = 0.3

def infer_accuracy():
    correct_count = 0
    total_count = 0
    people = glob.glob(os.path.join(img_check_path, '*'))
    for person in people:
        person_short = os.path.basename(os.path.normpath(person))
        images = glob.glob(os.path.join(person, '*.*'))
        for image in images:
            label = os.path.basename(os.path.normpath(image))
            if label.split(',')[0] != 'err':
                total_count += 1
            if person_short == label.split(',')[0]:
                correct_count += 1
            # print (label.split(',')[1])[:-4]
    print 'correct:{}'.format(correct_count)
    print 'total:{}'.format(total_count)
    total_count *= 1.0
    print '%correct:{:.2f}'.format((correct_count/total_count)*100)


def infer_accuracy_confidence_limited():
    correct_count = 0
    total_count = 0
    people_count = 0
    processed_people_count = 0
    people = glob.glob(os.path.join(img_check_path, '*'))
    for person in people:
        people_count += 1
        processed_person = False
        person_short = os.path.basename(os.path.normpath(person))
        images = glob.glob(os.path.join(person, '*.*'))
        for image in images:
            label = os.path.basename(os.path.normpath(image))
            if float((label.split(',')[1])[:-4]) >= confidence_threshold:
                if label.split(',')[0] != 'err':
                    total_count += 1
                if person_short == label.split(',')[0]:
                    if not processed_person:
                        processed_person = True
                        processed_people_count += 1
                    correct_count += 1
        if not processed_person:
            print person_short + ' has 0 correct predictions ... '
    print 'correct:{}'.format(correct_count)
    print 'total:{}'.format(total_count)
    total_count *= 1.0
    print '%correct:{:.2f}'.format((correct_count/total_count)*100)
    print 'number of people:{}'.format(people_count)
    print 'number of people with at least 1 correct prediction:{}'.format(processed_people_count)


raw_input("determine accuracy of recognition of {} dataset. press enter to continue ...".format(dataset))
# infer_accuracy()
infer_accuracy_confidence_limited()