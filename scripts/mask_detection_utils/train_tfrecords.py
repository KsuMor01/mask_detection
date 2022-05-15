import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
import tensorflow as tf

from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

def make_labels(label_map, custom):
    print('Creating label_map')

    #  labels = [{'name': 'with_mask', 'id': 1}, {'name': 'without_mask', 'id': 2},
    #  {'name': 'mask_weared_incorrect', 'id': 3}]

    labels = []
    if custom:
        print('Input num of labels:')
        n = int(input())
        print('Input custom labels:')
        for lbl in range(n):
            labels.append(input())
    else:
        labels = ['with_mask', 'without_mask', 'mask_weared_incorrect']

    with open(label_map, 'w') as f:
        for i, label in enumerate(labels):
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label))
            f.write('\tid:{}\n'.format(i+1))
            f.write('}\n')
    print(labels)

def split_data(): #TODO
    pass

def load_data(data_path):
    #TODO from path or from archive
    print('loading data...')
    max_imgs = 20
    dir = data_path + 'train/'
    for a,d, files in os.walk(dir):
        plt.figure(figsize=(10, 10))
        i = 0
        for f in files:
            if f.endswith('jpg'):
                # print(f, i)
                img = Image.open(dir+f)
                plt.subplot(5, 4 , i+1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)

                tree = ET.parse(dir + f[:-4] + '.xml')
                root = tree.getroot()

                for obj in root.findall('object'):
                    bndbox = obj.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)

                    out_img = ImageDraw.Draw(img)
                    out_img.rectangle([xmin,ymin,xmax, ymax], outline ="yellow", width=6)

                plt.imshow(img, cmap=plt.cm.gray)
                i += 1
                if i == max_imgs:
                    break
        plt.show()
        print('data loaded!')
def load_pretrained_model(name, dir):
    if not os.path.exists(dir + name):
        print('loading pretrained model ...')
        archive_name = name + '.tar.gz'
        command = 'cd ../pretrained_models/' \
                  + ' && wget http://download.tensorflow.org/models/object_detection/tf2/20200711/' \
                  + archive_name + ' && tar -zxvf ' + archive_name
        #print(command)
        #print(os.system(command))
        if os.system(command):
            print('Error! Something went wrong.')
        else:
            print('pretrained model loaded!')
    else:
        print('Pretrained model already exists')

def create_tfrecords(paths):
    print('creating tfrecords...')
    command = 'python ' + paths['TFRECORDS_SCRIPT'] + \
              ' -x ' + paths['DATA_PATH'] + 'train/' + \
              ' -l ' + paths['LABEL_MAP'] + \
              ' -o ' + paths['ANNOTATIONS'] + 'train.record'
    #print(command)
    os.system(command)

    command = 'python ' + paths['TFRECORDS_SCRIPT'] + \
              ' -x ' + paths['DATA_PATH'] + 'test/' + \
              ' -l ' + paths['LABEL_MAP'] + \
              ' -o ' + paths['ANNOTATIONS'] + 'test.record'
    #print(command)
    os.system(command)

def configure_pipeline(paths, num_train_steps):
    # pass
    print('configuring pipeline ...')
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(paths['PIPELINE'], "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)
    config = config_util.get_configs_from_pipeline_file(paths['PIPELINE'])
    pipeline_config.eval_config.max_evals = 1
    pipeline_config.model.ssd.num_classes = 3 #TODO
    pipeline_config.train_config.batch_size = 16
    pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINDED_DIR'],
                                                                     paths['PRETRAINED_MODEL'], 'checkpoint', 'ckpt-0')
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_config.num_steps = num_train_steps
    pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.learning_rate_base = 0.008
    pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.warmup_learning_rate = 0.001
    pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.total_steps = 5000
    pipeline_config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.warmup_steps = 300
    pipeline_config.train_input_reader.label_map_path = paths['LABEL_MAP']
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [
        os.path.join(paths['ANNOTATIONS'], 'train.record')]
    pipeline_config.eval_input_reader[0].label_map_path = paths['LABEL_MAP']
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [
        os.path.join(paths['ANNOTATIONS'], 'test.record')]
    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(paths['PIPELINE'], "wb") as f:
        f.write(config_text)
    print('pipeline updated!')


def train_model(paths, num_train_steps):
    #  !python {TRAINING_SCRIPT} --model_dir={paths['CHECKPOINT_PATH']} --pipeline_config_path={files['PIPELINE_CONFIG']} --num_train_steps={num_train_steps}
    if not os.path.exists(paths['MODEL_DIR']):
        os.mkdir(paths['MODEL_DIR'])
        os.mkdir(paths['TRAIN_DIR'])
    print('=============training model start==============')
    command = 'python ' + paths['TRAINING_SCRIPT'] + \
              ' --model_dir=' + paths['TRAIN_DIR'] + \
              ' --pipeline_config_path=' + paths['PIPELINE'] + \
              ' --num_train_steps=' + str(num_train_steps)
    print(command)
    os.system(command)


def freeze_graph(paths):
    #TODO create freeze dir
    #!python {FREEZE_SCRIPT} --input_type=image_tensor --pipeline_config_path={files['PIPELINE_CONFIG']} --trained_checkpoint_dir={paths['CHECKPOINT_PATH']} --output_directory={paths['OUTPUT_PATH']}
    print('freezing graph')
    if not os.path.exists(paths['FREEZE_DIR']):
        os.mkdir(paths['FREEZE_DIR'])
    command = 'python ' + paths['FREEZE_GRAPH_SCRIPT'] + \
              ' --input_type=image_tensor ' + \
              '--pipeline_config_path=' + paths['PIPELINE'] + \
              ' --trained_checkpoint_dir=' + paths['TRAIN_DIR'] + \
              ' --output_directory=' + paths['FREEZE_DIR']
    print(command)
    os.system(command)

def convert_model(paths, model_name):
#!python {TFLITE_SCRIPT} --max_detections=10 --pipeline_config_path={files['PIPELINE_CONFIG']} --trained_checkpoint_dir={paths['CHECKPOINT_PATH']} --output_directory={paths['TFLITE_PATH']}
    print('converting tflite graph')
    if not os.path.exists(paths['TFLITE_DIR']):
        os.mkdir(paths['TFLITE_DIR'])
    command = 'python ' + paths['TFLITE_GRAPH_SCRIPT'] + \
              ' --max_detections=10' + \
              ' --pipeline_config_path=' + paths['PIPELINE'] + \
              ' --trained_checkpoint_dir=' + paths['TRAIN_DIR'] + \
              ' --output_directory=' + paths['TFLITE_DIR']
    print(command)
    os.system(command)

    images_paths = []
    for currentFile in Path(paths['DATA_PATH'] + 'train/').glob('*.jpg'):
        currentFileStr = str(currentFile)
        images_paths.append(currentFileStr)
        print(currentFile)
    images_paths.sort()
    image_size = 320

    def rep_data_gen():
        a = []
        for i in range(len(images_paths)):
            file_name = images_paths[i]
            img = cv2.imread(file_name)
            if img is None:
                continue
            img = cv2.resize(img, (image_size, image_size))
            img = img
            img = img.astype(np.float32) / 255
            a.append(img)
        a = np.array(a)
        img = tf.data.Dataset.from_tensor_slices(a).batch(1)
        for i in img.take(16):
            #print(i)
            yield [i]
    # print(paths['TFLITE_DIR'])

    converter = tf.lite.TFLiteConverter.from_saved_model(paths['TFLITE_DIR'] + 'saved_model')
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data_gen
    converter.inference_output_type = tf.float32
    tflite_model = converter.convert()

    print('\nsaving model')

    with open(paths['TFLITE_DIR'] + model_name + '.tflite', "wb") as f:
        f.write(tflite_model)
