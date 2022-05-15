from io import BytesIO
from time import sleep
from picamera import PiCamera
from PIL import Image
import tflite_runtime.interpreter as tflite
import numpy as np

#TODO
'''
1. 

'''

# set camera stream
stream = BytesIO()
camera = PiCamera()
camera.start_preview()

model_path = "test_model.tflite"
interpreter = tflite.Interpreter(model_path=model_path, num_threads=None)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
floating_model = input_details[0]['dtype'] == np.float32
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

while True:
    camera.capture(stream, format='jpeg')
    stream.seek(0)

    img = Image.open(stream)
    wt, ht = img.size
    img = img.resize((width, height))

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
        if scores[0, j] > 0.3:
            x1 = boxes[0, j, 1] * wt
            x2 = boxes[0, j, 3] * wt
            y1 = boxes[0, j, 0] * ht
            y2 = boxes[0, j, 2] * ht
            #cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)







