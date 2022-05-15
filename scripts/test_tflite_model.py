import cv2
import pathlib
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

model_path = "../models/test_model_320/tflite/test_model_320.tflite"
test_data_path = '../data/dataset_prop/test/images/'
interpreter = tflite.Interpreter(model_path=model_path, num_threads=None)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
floating_model = input_details[0]['dtype'] == np.float32
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]


def test():
    batch = 0
    plt.figure(figsize=(10, 10))
    print(test_data_path)
    for path, dir, files in os.walk(test_data_path):
        for i, filename in enumerate(files):
            print(filename)

            image = cv2.imread(path + filename)
            ht, wt = image.shape[:2]
            #print(ht, wt)
            colors_cnv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(colors_cnv)
            img = img.resize((width, height))
            #print(img.size)
            input_data = np.expand_dims(img, axis=0)
            #print(input_data)

            if floating_model:
                input_data = (np.float32(input_data) - 127.5) / 127.5

            interpreter.set_tensor(input_details[0]['index'], input_data)

            interpreter.invoke()
            #print('invoked')
            boxes = interpreter.get_tensor(output_details[1]['index'])
            classes = interpreter.get_tensor(output_details[2]['index'])
            scores = interpreter.get_tensor(output_details[0]['index'])
            num = interpreter.get_tensor(output_details[3]['index'])

            # print("BOXES\n", boxes)
            # print("CLASSES\n", classes)
            # print("SCORES\n", scores)
            # print("NUM\n", num)

            for j in range(0, len(boxes[0])):
                if scores[0, j] > 0.3:
                    x1 = boxes[0, j, 1] * wt
                    x2 = boxes[0, j, 3] * wt
                    y1 = boxes[0, j, 0] * ht
                    y2 = boxes[0, j, 2] * ht
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            # cv2.imshow("img" + str(batch), image)
            # cv2.moveWindow("img" + str(batch), 300*batch, 0)
            # cv2.waitKey(0)

            colors_cnv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(colors_cnv)
            img = img.resize((width, height))
            cv2.imshow("out", image)
            cv2.waitKey(0)


            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)

            plt.imshow(img,cmap=plt.cm.gray)
            batch += 1
            if batch == 10:
                break

        plt.show()
test()

