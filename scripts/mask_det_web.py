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
model_path = "od.tflite"
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

import io
import picamera
import logging
import socketserver
from threading import Condition
from http import server

PAGE="""\
<html>
<head>
<title>mask_detection/title>
</head>
<body>
<h1>PiCamera MJPEG Streaming Demo</h1>
<img src="stream.mjpg" width="640" height="480" />
</body>
</html>
"""

class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame, copy the existing buffer's content and notify all
            # clients it's available
            self.buffer.truncate()
            with self.condition:
                self.frame = self.buffer.getvalue()
                self.condition.notify_all()
            self.buffer.seek(0)
        return self.buffer.write(buf)

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


with picamera.PiCamera() as camera:
    stream = io.BytesIO()
    output = StreamingOutput()
    camera.start_recording(output, format='mjpeg')

    try:
        address = ('', 8000)
        server = StreamingServer(address, StreamingHandler)
        server.serve_forever()
    finally:
        camera.stop_recording()


    # for i, foo in enumerate(camera.capture_continuous(stream, format='jpeg', resize=(300,300))):
    #     start_time = time.time()
    #     # Truncate the stream to the current position (in case
    #     # prior iterations output a longer image)
    #     print(i)
    #     img = Image.open(stream)
    #
    #     fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 12)
    #     color = (255,255,255)
    #     d = ImageDraw.Draw(img)
    #     d.multiline_text((10, 10), datetime.now().strftime("%H:%M:%S"), font=fnt, fill=color)
    #
    #
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
    #         if scores[0, j] > 0.5:
    #             x1 = int(boxes[0, j, 1] * wt)
    #             x2 = int(boxes[0, j, 3] * wt)
    #             y1 = int(boxes[0, j, 0] * ht)
    #             y2 = int(boxes[0, j, 2] * ht)
    #
    #             print('coords ', x1, y1, x2, y2)
    #             d.rectangle([x1, y1, x2, y2], fill=None, outline=color, width=2)
    #
    #
    #
    #     # img.save('test_images/' + str(i)+'.jpg')
    #     stream.truncate()
    #     stream.seek(0)
    #
    #     fps = str(int(1 / (time.time() - start_time))) + 'FPS'
    #     # camera.annotate_text = fps
    #     print('inference ', time.time()-start_time, 's\n', fps)

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





