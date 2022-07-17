from io import BytesIO
import time
import picamera
from picamera import PiCamera
from PIL import Image, ImageDraw, ImageShow, ImageFont
import tflite_runtime.interpreter as tflite
import numpy as np
from datetime import datetime

import io
import socket
import struct

#TODO
'''
1. 

'''

# set camera stream

# camera = PiCamera()
# stream = BytesIO()

# model_path = "test_model_320.tflite"
# model_path = "od.tflite"
model_path = "test_model_4.tflite"
interpreter = tflite.Interpreter(model_path=model_path, num_threads=None)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
floating_model = input_details[0]['dtype'] == np.float32
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

frame_it = 0
# camera.capture(stream, format='jpeg', resize=(320,320))
# camera.capture_continuous(stream, format='jpeg', resize=(300, 300))
# stream.seek(0)

with picamera.PiCamera() as camera:
    stream = io.BytesIO()
    camera.start_preview()
    for i, foo in enumerate(camera.capture_continuous(stream, format='jpeg', resize=(width, height))):
        start_time = time.time()
        # camera.start_preview()
        print(i)
        img = Image.open(stream)

        fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 12)
        color = (255, 255, 255)
        d = ImageDraw.Draw(img)
        d.multiline_text((50, 10), datetime.now().strftime("%H:%M:%S"), font=fnt, fill=color)


        wt, ht = img.size
        print('frame size ', img.size)

        # img = img.resize((width, height))
        print('input size ', img.size)

        input_data = np.expand_dims(img, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5

        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[1]['index'])
        classes = interpreter.get_tensor(output_details[3]['index'])
        scores = interpreter.get_tensor(output_details[0]['index'])
        num = interpreter.get_tensor(output_details[2]['index'])

        print("BOXES\n", boxes)
        print("CLASSES\n", classes)
        print("SCORES\n", scores)
        print("NUM\n", num)

        for j in range(0, len(boxes[0])):
            if scores[0, j] > 0.1:
                x1 = int(boxes[0, j, 1] * wt)
                x2 = int(boxes[0, j, 3] * wt)
                y1 = int(boxes[0, j, 0] * ht)
                y2 = int(boxes[0, j, 2] * ht)

                print('coords ', x1, y1, x2, y2)
                d.rectangle([x1, y1, x2, y2], fill=None, outline=color, width=2)


        fps = str(int(1 / (time.time() - start_time))) + 'FPS'
        d.multiline_text((10, 10), fps, font=fnt, fill=color)
        print('inference ', time.time()-start_time, 's\n', fps)

        # img.show(title='img')
        # v = ImageShow.Viewer()
        # v.show_image(img)

        img.save('test_images/' + str(i)+'.jpg')
        stream.truncate()
        stream.seek(0)

        # if i == 10:
        #     break


#
# while True:
#     start_time = time.time()
#     frame_it += 1
#     print('FRAME # ', frame_it)
#
#     camera.capture_continuous(stream, format='jpeg', resize=(300, 300))
#
#     img = Image.open(stream)
#     wt, ht = img.size
#     print('frame size ', img.size)
#
#     # img = img.resize((width, height))
#     print('input size ', img.size)
#
#     input_data = np.expand_dims(img, axis=0)
#
#     if floating_model:
#         input_data = (np.float32(input_data) - 127.5) / 127.5
#
#     interpreter.set_tensor(input_details[0]['index'], input_data)
#
#     interpreter.invoke()
#
#     # boxes = interpreter.get_tensor(output_details[1]['index'])
#     # classes = interpreter.get_tensor(output_details[3]['index'])
#     # scores = interpreter.get_tensor(output_details[0]['index'])
#     # num = interpreter.get_tensor(output_details[2]['index'])
#
#     boxes = interpreter.get_tensor(output_details[0]['index'])
#     classes = interpreter.get_tensor(output_details[1]['index'])
#     scores = interpreter.get_tensor(output_details[2]['index'])
#     num = interpreter.get_tensor(output_details[3]['index'])
#
#     print("BOXES\n", boxes)
#     print("CLASSES\n", classes)
#     print("SCORES\n", scores)
#     print("NUM\n", num)
#
#     for j in range(0, len(boxes[0])):
#         if scores[0, j] > 0.1:
#             x1 = int(boxes[0, j, 1] * wt)
#             x2 = int(boxes[0, j, 3] * wt)
#             y1 = int(boxes[0, j, 0] * ht)
#             y2 = int(boxes[0, j, 2] * ht)
#
#             print('coords ', x1, y1, x2, y2)
#
#             # draw = ImageDraw.Draw(img)
#
#             # draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)
#             # draw.rectangle([0, 0, 100, 100], outline="yellow", width=3)
#             # img1 = img
#             # o = camera.add_overlay(img1.tobytes(), size=(1024, 768))
#             # o.alpha = 128
#             # o.layer = 3
#             # ImageShow.show(img)
#
#             # cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
#
#     fps = str(int(1 / (time.time() - start_time))) + 'FPS'
#     # camera.annotate_text = fps
#     print('inference ', time.time()-start_time, 's\n', fps)
#
#     fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)
#     color = (255, 255, 255)
#     d = ImageDraw.Draw(img)
#     d.multiline_text((10, 10), "Hello\nWorld", font=fnt, fill=color)
#     d.rectangle([10, 10, 20, 20], fill=None, outline=color, width=2)
#     img.save(str(time.time()) + '.jpg')
#
#
#     stream.truncate()
#     stream.seek(0)
#





