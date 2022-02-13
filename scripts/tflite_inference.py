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
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Face detection")
    parser.add_argument("--model_det", type=str, default="od.tflite", help="model for object detection tflite")
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
vs = cv2.VideoCapture(args.video, cv2.CAP_V4L2)
if vs.isOpened():
    print("Camera is opened on video", args.video)
else:
    for i in range(0, 5):
        vs = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if vs.isOpened():
            print("Camera is opened on video", i)
            break

##########################################################################################################

fps = " "
# start reading frames
while True:
    ret, frame = vs.read()

    if ret == False:
        continue

    start_time = time.time()


########################################## NN WORKING ###################################################
    # convert to PIL
    colors_cnv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(colors_cnv)

    img = img.resize((width_det, height_det))
    input_data_det = np.expand_dims(img, axis=0)

    if floating_model:
        input_data_det = (np.float32(input_data_det) - 127.5) / 127.5

    interpreter_det.set_tensor(input_details_det[0]['index'], input_data_det)

    interpreter_det.invoke()
   
    boxes = interpreter_det.get_tensor(output_details_det[0]['index'])
    classes = interpreter_det.get_tensor(output_details_det[1]['index'])
    scores = interpreter_det.get_tensor(output_details_det[2]['index'])
    num_boxes = interpreter_det.get_tensor(output_details_det[3]['index'])

###################################### PERSON DETECTION #################################################
    x1 = x2 = y1 = y2 = None
    for i in range(int(num_boxes)):
      if (scores[0, i] > .5 and int(classes[0, i]) == 0):
           x1 = boxes[0, i, 1] * 640
           x2 = boxes[0, i, 3] * 640
           y1 = boxes[0, i, 0] * 480
           y2 = boxes[0, i, 2] * 480
           cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
           cv2.putText(frame, 'person', (int(x1), int(y1)), cv2.LINE_AA, 0.45, (0, 0, 255), 2)

################################################ FPS ##############################################################
    stop_time=time.time()
    fps = "FPS: " + str(int(1/(stop_time - start_time)))
    cv2.putText(frame, fps, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2) 
    print('FPS ',fps, '\nFull', (stop_time - start_time)*1000, 'ms')
#########################################################################################
    cv2.imshow("Window", frame)
    k = cv2.waitKey(1) & 0xff
    if k==27:
        break
cv2.destroyAllWindows()
vs.release()

























