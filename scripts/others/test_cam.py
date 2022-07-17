# from io import BytesIO
# from time import sleep
# from picamera import PiCamera
# from PIL import Image
#
# # Create the in-memory stream
# stream = BytesIO()
# camera = PiCamera()
# camera.start_preview()
# sleep(2)
# camera.capture(stream, format='jpeg')
# # "Rewind" the stream to the beginning so we can read its content
# stream.seek(0)
# image = Image.open(stream)
# filename = "img{timestamp:%Y-%m-%d-%H-%M}.jpg"
# # image.save(filename)
# print(filename)
#
#
# for filename in camera.capture_continuous(stream, 'jpeg'):
#     # print('Captured %s' % filename)
#     # img =
#     sleep(2)

# import io
# import time
# import picamera
# from PIL import Image, ImageDraw, ImageFont
# with picamera.PiCamera() as camera:
#     stream = io.BytesIO()
#     for i, foo in enumerate(camera.capture_continuous(stream, format='jpeg')):
#         st = time.time()
#         # Truncate the stream to the current position (in case
#         # prior iterations output a longer image)
#         print(i)
#         image = Image.open(stream)
#
#         fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)
#         color = (255,255,255)
#         d = ImageDraw.Draw(image)
#         d.multiline_text((10, 10), "Hello\nWorld", font=fnt, fill=color)
#         d.rectangle([10, 10, 20, 20], fill=None, outline=color, width=2)
#
#         image.save(str(i)+'.jpg')
#         stream.truncate()
#         stream.seek(0)
#         print('inference ', time.time()-st)
#         if i == 5:
#             break

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

import cv2

vs = cv2.VideoCapture(1, cv2.CAP_V4L2)
# if vs.isOpened():
#     print("Camera is opened on video", 0)
# else:
#     for i in range(0, 5):
#         vs = cv2.VideoCapture(i, cv2.CAP_V4L2)
#         if vs.isOpened():
#             print("Camera is opened on video", i)
#             break

out_size = (640,  480)

# vs = cv2.VideoCapture('/home/ksumor/Downloads/test_photos/test_video.mp4')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output2.avi', fourcc, 30.0, out_size)

##########################################################################################################

fps = " "
# start reading frames
frame_num = 0
while True:
    ret, frame = vs.read()

    if ret == False:
        continue

    start_time = time.time()


    ################################################ FPS ##############################################################
    stop_time=time.time()
    fps = "FPS: " + str(int(1/(stop_time - start_time)))
    out_frame = cv2.resize(frame, out_size)
    out.write(out_frame)

    cv2.putText(frame, fps, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    print('FPS ', fps, '\nFull', (stop_time - start_time)*1000, 'ms')
#########################################################################################
    # cv2.imshow("Window", frame)
    # cv2.waitKey(1)
    # k = cv2.waitKey(1) & 0xff
    # if k==27:
    #     break
cv2.destroyAllWindows()
vs.release()









































