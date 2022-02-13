import numpy as np
import os

from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf

from absl import logging

spec = model_spec.get('efficientdet_lite0')

#os.command('unzip ../data/dataset_prop.zip')

data_path = '../data/dataset_prop/'
train_data = object_detector.DataLoader.from_pascal_voc( \
    data_path + 'train/images', data_path + 'train/annotations', \
    label_map={1: "without_mask", 2: "with_mask", 3: "mask_weared_incorrect"})

test_data = object_detector.DataLoader.from_pascal_voc( \
    data_path + 'test/images', data_path + 'test/annotations', \
    label_map={1: "without_mask", 2: "with_mask", 3: "mask_weared_incorrect"})
print('data loaded')
model = object_detector.create(train_data, model_spec=spec, epochs=50)

model.export(export_dir='.')