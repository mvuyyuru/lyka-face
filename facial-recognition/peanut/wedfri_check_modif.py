import os
import glob
import shutil

base_path = os.path.dirname(os.path.realpath(__file__))
check_path = os.path.join(base_path, 'img/wedfri/employ/check')
basic_check_path = os.path.join(base_path, 'img/wedfri/employ/basic_check')
conf_check_path_base = os.path.join(base_path, 'img/wedfri/employ/')

def move_to_folders():
    dump_images = glob.glob(os.path.join(check_path, '*.*'))

    if not os.path.exists(basic_check_path):
            os.makedirs(basic_check_path)

    for image in dump_images:
        label = os.path.basename(os.path.normpath(image))
        guess_person = label.split(',')[0]
        basic_check_person_path = os.path.join(basic_check_path, guess_person)
        if not os.path.exists(basic_check_person_path):
            os.makedirs(basic_check_person_path)
        shutil.copy(image, basic_check_person_path)



def move_to_folders_restrict_by_conf(conf_thres):
    dump_images = glob.glob(os.path.join(check_path, '*.*'))

    conf_check_path = os.path.join(conf_check_path_base, str(conf_thres) + '_conf_check')
    if not os.path.exists(conf_check_path):
        os.makedirs(conf_check_path)

    for image in dump_images:
        label = os.path.basename(os.path.normpath(image))
        guess_person = label.split(',')[0]
        guess_conf = (label.split(',')[1])[:-4]
        if float(guess_conf) >= conf_thres:
            conf_check_person_path = os.path.join(conf_check_path, guess_person)
            if not os.path.exists(conf_check_person_path):
                os.makedirs(conf_check_person_path)
            shutil.copy(image, conf_check_person_path)



#move_to_folders()
move_to_folders_restrict_by_conf(0.6)
move_to_folders_restrict_by_conf(0.65)
move_to_folders_restrict_by_conf(0.7)
move_to_folders_restrict_by_conf(0.75)
move_to_folders_restrict_by_conf(0.85)
move_to_folders_restrict_by_conf(0.9)
move_to_folders_restrict_by_conf(0.95)



