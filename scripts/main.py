'''
This is main script for mask_detection project

You can train, convert and test mask_detection tflite model
'''

from mask_detection_utils.train_tfrecords import *

import argparse


parser = argparse.ArgumentParser(description='Settings for mask_detection')
parser.add_argument('-d', '--data_path', type=str, default='../data/dataset2_split/',
                    help='path to the data')
parser.add_argument('--model_name', type=str, default='test_model_7',
                    help='name of the model')
parser.add_argument('--pretrained_model', type=str, default='ssd_mobilenet_v2_320x320_coco17_tpu-8',
                    help='name of the model of the TensorFlow 2 Detection Model Zoo')
parser.add_argument('--labels_custom', type=bool, default=False,
                    help='make this True to put your own labels list')
parser.add_argument('--num_train_steps', type=bool, default=10000,
                    help='num of train steps')

args = parser.parse_args()
print('\n=======ARGS=========\n')
print(args)

paths = {
    'ANNOTATIONS': '../annotations/',
    'UTILS': 'od_lib_utils/',
    'PRETRAINDED_DIR': '../pretrained_models/'
}

paths['PRETRAINED_MODEL'] = args.pretrained_model
paths['MODEL_DIR'] = '../models/' + args.model_name + '/'

paths['DATA_PATH'] = args.data_path
paths['LABEL_MAP'] = paths['ANNOTATIONS'] + 'label_map.pbtxt'
paths['PIPELINE'] = paths['ANNOTATIONS'] + 'pipeline.config'

paths['TFRECORDS_SCRIPT'] = paths['UTILS'] + 'generate_tfrecord.py'
paths['TRAINING_SCRIPT'] = paths['UTILS'] + 'model_main_tf2.py'
paths['FREEZE_GRAPH_SCRIPT'] = paths['UTILS'] + 'exporter_main_v2.py'
paths['TFLITE_GRAPH_SCRIPT'] = paths['UTILS'] + 'export_tflite_graph_tf2.py'

paths['TRAIN_DIR'] = paths['MODEL_DIR'] + 'train/'
paths['FREEZE_DIR'] = paths['MODEL_DIR'] + 'freeze/'
paths['TFLITE_DIR'] = paths['MODEL_DIR'] + 'tflite/'


print('\n=======PATHS=========\n')
for path in paths.items():
    print(path)

print('\n=======LABELS=========\n')
# if(args.labels_custom):
    # make_labels(paths['LABEL_MAP'], args.labels_custom)

print('\n=======DATA=========\n')
# load_data(args.data_path)

print('\n=======PRETRAINED MODEL=========\n')
# load_pretrained_model(paths['PRETRAINED_MODEL'], paths['PRETRAINDED_DIR'])

print('\n=======TFRECORDS=========\n')
# create_tfrecords(paths)

print('\n=======CONFIG=========\n')
# configure_pipeline(paths, args.num_train_steps)

print('\n=======TRAIN=========\n')
# train_model(paths, 1000)

print('\n=======FREEZE=========\n')
# freeze_graph(paths)

print('\n=======CONVERT=========\n')
convert_model(paths, args.model_name)
print('\n=======TEST=========\n')

print('\n=======PLOT=========\n')



