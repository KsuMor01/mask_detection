from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

import cv2

########################################## SET ##########################################################

parser = argparse.ArgumentParser(description="Mask detection")

# parser.add_argument("--model_det", type=str, default="test_model_7.tflite", help="model for object detection tflite")
parser.add_argument("--model_det", type=str, default="../tflite_models/test_model_7.tflite", help="model for object detection tflite")
parser.add_argument("--video", type=str, default=2, help="video port number")
args = parser.parse_args()

interpreter_det = tflite.Interpreter(model_path=args.model_det)
interpreter_det.allocate_tensors()
input_details_det = interpreter_det.get_input_details()
output_details_det = interpreter_det.get_output_details()
floating_model = input_details_det[0]['dtype'] == np.float32
height_det = input_details_det[0]['shape'][1]
width_det = input_details_det[0]['shape'][2]

######################################## CAMERA OPEN #####################################################
# vs = cv2.VideoCapture(1, cv2.CAP_V4L2)
# if vs.isOpened():
#     print("Camera is opened on video", args.video)
# else:
#     for i in range(0, 5):
#         vs = cv2.VideoCapture(i, cv2.CAP_V4L2)
#         if vs.isOpened():
#             print("Camera is opened on video", i)
#             break

out_size = (640,  480)

# vs = cv2.VideoCapture('test_video.mp4')
vs = cv2.VideoCapture('/home/ksumor/Downloads/test_photos/test_video.mp4')


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output4.avi', fourcc, 30.0, out_size)

##########################################################################################################

fps = " "
# start reading frames
frame_num = 0
while True:
    ret, frame = vs.read()
    print(ret)

    if ret == False:
        continue

    start_time = time.time()

    ht, wt = frame.shape[:2]

    frame_num += 1
    # if frame_num % 10 != 0:
    #     continue
    # if frame_num == 100:
    #     break


########################################## NN WORKING ###################################################
    # convert to PIL
    colors_cnv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(colors_cnv)

    img = img.resize((width_det, height_det))
    input_data_det = np.expand_dims(img, axis=0)

    if floating_model:
        input_data_det = (np.float32(input_data_det) - 127.5) / 127.5

    interpreter_det.set_tensor(input_details_det[0]['index'], input_data_det)

    st = time.time()
    interpreter_det.invoke()
    print('invoke ', (time.time()-st)*100)

    boxes = interpreter_det.get_tensor(output_details_det[1]['index'])
    classes = interpreter_det.get_tensor(output_details_det[3]['index'])
    scores = interpreter_det.get_tensor(output_details_det[0]['index'])
    num_boxes = interpreter_det.get_tensor(output_details_det[2]['index'])

###################################### PERSON DETECTION #################################################
    x1 = x2 = y1 = y2 = None
    color = (0, 0, 0)
    for j in range(0, len(boxes[0])):
        score = scores[0, j]
        if score > 0.5:
            x1 = boxes[0, j, 1] * wt
            x2 = boxes[0, j, 3] * wt
            y1 = boxes[0, j, 0] * ht
            y2 = boxes[0, j, 2] * ht
            coords = (int(x1), int(y1)), (int(x2), int(y2))
            print(coords)
            pred_class = int(classes[0, j])
            label = 'with_mask'
            if pred_class == 0:
                color = (0, 255, 0)
                label = 'with_mask'
            if pred_class == 1:
                color = (0, 0, 255)
                label = 'without_mask'
            if pred_class == 2:
                color = (255, 0, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            cv2.putText(frame, label + ' ' + str(int(score*100)) + '%', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color,
                        2)

    ################################################ FPS ##############################################################
    stop_time=time.time()
    fps = "FPS: " + str(int(1/(stop_time - start_time)))
    # out_frame = cv2.resize(frame, out_size)
    # out.write(out_frame)

    cv2.putText(frame, fps, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    print('FPS ', fps, '\nFull', (stop_time - start_time)*1000, 'ms')
#########################################################################################
    # cv2.imshow("Window", frame)
    # cv2.waitKey(0)
    # k = cv2.waitKey(1) & 0xff
    # if k==27:
    #     break
cv2.destroyAllWindows()
vs.release()

























