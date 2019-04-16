import os
import sys
import random

import cv2
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from settings import app

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-IMAGES_INPUT_FOLDER', '--IMAGES_INPUT_FOLDER', action = 'store', dest = 'IMAGES_INPUT_FOLDER', help = "train dir", default='/data/home/jdegange/vision/rvlcdip_densenet/rvlcdip_densenet/dataset/all/val/')
parser.add_argument('-OUTPUT_FILENAME', '--OUTPUT_FILENAME', action = 'store', dest = 'OUTPUT_FILENAME', help = "OUTPUT_FILENAME dir", default='/datadrive1/rvl_cdip_tf_record/val/')
parser.add_argument('-NUMBER_OF_SHARDS', '--NUMBER_OF_SHARDS', action = 'store', dest = 'NUMBER_OF_SHARDS', help = "NUMBER_OF_SHARDS dir", default=100)
parser.add_argument('-TRAINING_EXAMPLES_SPLIT', '--TRAINING_EXAMPLES_SPLIT', action = 'store', dest = 'TRAINING_EXAMPLES_SPLIT', help = "train TRAINING_EXAMPLES_SPLIT", default=0)
parser.add_argument('-SEED', '--SEED', action = 'store', dest = 'SEED', help = "SEED", default=123)
parser.add_argument('-FILENAME_PREFIX', '--FILENAME_PREFIX', action = 'store', dest = 'FILENAME_PREFIX', help = "train dir", default='VAL')


args = parser.parse_args()
print(args)

IMAGES_INPUT_FOLDER = str(args.IMAGES_INPUT_FOLDER)
IMAGES_INPUT_FOLDER = str(args.IMAGES_INPUT_FOLDER)
TRAINING_EXAMPLES_SPLIT = float(args.TRAINING_EXAMPLES_SPLIT)
OUTPUT_FILENAME = str(args.OUTPUT_FILENAME)
NUMBER_OF_SHARDS = int(args.NUMBER_OF_SHARDS)
SEED = int(args.SEED)
FILENAME_PREFIX = str(args.FILENAME_PREFIX)

def _load_image(path):
    image = cv2.imread(path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype(np.float32)
    return None

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _build_examples_list(input_folder, seed):
    examples = []
    for classname in tqdm(os.listdir(input_folder)):
        class_dir = os.path.join(input_folder, classname)
        if (os.path.isdir(class_dir)):
            for filename in os.listdir(class_dir):
                filepath = os.path.join(class_dir, filename)
                example = {
                    'classname': classname, 
                    'path': filepath
                }
                print(str(filepath))
                examples.append(example)

    random.seed(seed)
    random.shuffle(examples)
    print(str(len(examples)), " found for conversion." )
    return examples

def _split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]

def _get_examples_share(examples, training_split):
    examples_size = len(examples)
    len_training_examples = int(examples_size * training_split)
    return np.split(examples, [len_training_examples])

def _write_tfrecord(examples, output_filename):
    writer = tf.python_io.TFRecordWriter(output_filename)
    for example in tqdm(examples):
        try:
            image = _load_image(example['path'])
            if image is not None:
                print(str(example['path']))
                encoded_image_string = cv2.imencode('.jpg', image)[1].tostring()
                feature = {
                    'train/label': _bytes_feature(tf.compat.as_bytes(example['classname'])),
                    'train/image': _bytes_feature(tf.compat.as_bytes(encoded_image_string))
                }

                tf_example = tf.train.Example(features = tf.train.Features(feature=feature))
                writer.write(tf_example.SerializeToString())
        except Exception as inst:
            print(inst)
            #pass
    writer.close()

def _write_sharded_tfrecord(examples, number_of_shards, base_output_filename,FILENAME_PREFIX):
    sharded_examples = _split_list(examples, number_of_shards)
    for count, shard in tqdm(enumerate(sharded_examples, start = 1)):
        output_filename = '{0}_{1}_{2:02d}of{3:02d}.tfrecord'.format(
            base_output_filename,
            FILENAME_PREFIX,
            count,
            number_of_shards 
        )
        _write_tfrecord(shard, output_filename)


examples = _build_examples_list(IMAGES_INPUT_FOLDER, SEED)
training_examples, test_examples = _get_examples_share(examples, TRAINING_EXAMPLES_SPLIT) # pylint: disable=unbalanced-tuple-unpacking

print("Creating training shards", flush = True)
_write_sharded_tfrecord(training_examples, NUMBER_OF_SHARDS, OUTPUT_FILENAME, FILENAME_PREFIX)
#print("\nCreating test shards", flush = True)
#_write_sharded_tfrecord(test_examples, app['NUMBER_OF_SHARDS'], app['OUTPUT_FILENAME'], False)
print("\n", flush = True)
