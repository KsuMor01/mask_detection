import cv2
import pathlib
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from time import time

model_path = "/home/ksumor/Downloads/ssd_resnet50.tflite"
test_data_path = '/home/ksumor/Downloads/test_photos/'
interpreter = tflite.Interpreter(model_path=model_path, num_threads=None)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
floating_model = input_details[0]['dtype'] == np.float32
if floating_model:
    print('Floating model')
else:
    print('Uint8 model')

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

print('input size ', width, height)


def test():
    batch = 25
    conf_filter = 0.5
    plt.figure(figsize=(10, 10))
    print(test_data_path)
    color = (255, 255, 255)
    for path, dir, files in os.walk(test_data_path):
        for i, filename in enumerate(files):
            print(filename)
            if filename.endswith('xml'):
                continue
            if filename.endswith('mp4'):
                continue

            image = cv2.imread(path + filename)
            # orig_img = image.copy()
            ht, wt = image.shape[:2]
            #ht, wt = (300,300)
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

            st = time()
            interpreter.invoke()
            print('invoke ', (time()-st)*1000)
            boxes = interpreter.get_tensor(output_details[1]['index'])
            classes = interpreter.get_tensor(output_details[2]['index'])
            scores = interpreter.get_tensor(output_details[0]['index'])
            num = interpreter.get_tensor(output_details[3]['index'])

            print("BOXES\n", boxes)
            # print("CLASSES\n", classes)
            print("SCORES\n", scores)
            # print("NUM\n", num)
            # image = cv2.resize(image, (300, 300))

            # for j in range(0, len(boxes[0])):
            #     score = scores[0, j]
            #     if score > conf_filter:
            #         x1 = boxes[0, j, 1] * wt
            #         x2 = boxes[0, j, 3] * wt
            #         y1 = boxes[0, j, 0] * ht
            #         y2 = boxes[0, j, 2] * ht
            #         coords = (int(x1), int(y1)), (int(x2), int(y2))
            #         pred_class = int(classes[0, j])
            #         label = 'with_mask'
            #         if pred_class == 0:
            #             color = (0, 255, 0)
            #             label = 'with_mask'
            #         if pred_class == 1:
            #             color = (0, 0, 255)
            #             label = 'without_mask'
            #         if pred_class == 2:
            #             color = (255, 0, 0)
            #         # image = cv2.resize(image, (300, 300))
            #         cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 15)
            #         cv2.putText(image, label + ' ' + str(int(score*100)) + '%', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
            #         print('OUT ')
            #         print((int(x1), int(y1)), (int(x2), int(y2)))
            #         print(pred_class)
            #         print(score)

            # cv2.imshow("img" + str(batch), image)
            # cv2.moveWindow("img" + str(batch), 300*batch, 0)
            # cv2.waitKey(0)

            # colors_cnv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # img = Image.fromarray(colors_cnv)
            # img.save('../output2/img1'+ str(i) + '.jpg')
            # img = img.resize((width, height))
            # image = cv2.resize(image, (400,400))

            # cv2.imshow("out", image)
            # cv2.waitKey(0)

            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)

            plt.imshow(img, cmap=plt.cm.gray)
            print('\n')
            # batch += 1
            if i == batch:
                break

    plt.show()
test()

