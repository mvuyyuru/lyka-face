# peanut - v 0.3

import os
import csv

print "peanut - v 0.3"

video = 'lfw'

base_path = os.path.dirname(os.path.realpath(__file__))
memory_dump_path = os.path.join(base_path, 'memories/{}'.format(video))
memory_dump = os.path.join(memory_dump_path, 'memory.csv')

def render_memories():
    with open(memory_dump, 'rb') as memory_csv:
        reader = csv.reader(memory_csv)
        predictions = list(reader)

    for prediction in predictions:
        print prediction

# raw_input("write video visually indicating recognitions. press enter to continue ...")
render_memories()
