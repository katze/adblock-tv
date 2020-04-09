# python3
#
# Pour ce code, je me suis basé sur l'exemple de Tensorflow dispo à cette adresse : 
# https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/raspberry_pi
# Je vous invite à commercer par cet expemple notament pour l'installation de TF sur votre Rasperry Pi

# La configuration de LIRC est une vrai galère depuis Raspian Buster
# Voici le tuto que j'ai suivi pour l'installation https://github.com/raspberrypi/linux/issues/2993

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import time
import numpy as np
import picamera
import os
import sys
import RPi.GPIO as GPIO
import signal

from PIL import Image
from tflite_runtime.interpreter import Interpreter

dirname = os.path.dirname(__file__)


GPIO.setmode(GPIO.BOARD)
GPIO.setup(40, GPIO.OUT, initial = GPIO.LOW)
GPIO.output(40, GPIO.HIGH)


def terminateProcess(signalNumber, frame):
    print ('(SIGTERM) terminating the process')
    sys.exit()


def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]


def main():

  history = []
  state = ''
  previous_state = ''

  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model', help='File path of .tflite file.', default=os.path.join(dirname, 'model/model.tflite'))
  parser.add_argument(
      '--labels', help='File path of labels file.', default=os.path.join(dirname, 'model/labels.txt'))
  args = parser.parse_args()

  labels = load_labels(args.labels)

  interpreter = Interpreter(args.model)
  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']

  with picamera.PiCamera(resolution=(600, 600), framerate=30) as camera:
    camera.iso = 100
    camera.vflip = 1
    camera.hflip = 1
    camera.crop = (0.0, 0.3, 0.7, 0.5)
    #camera.exposure_mode = "sport"
    camera.exposure_compensation = 0
    camera.exposure_mode = 'auto'
    camera.start_preview()
    try:
      stream = io.BytesIO()
      for _ in camera.capture_continuous(
          stream, format='jpeg', use_video_port=True):
        stream.seek(0)
        image = Image.open(stream).convert('RGB').resize((width, height), Image.ANTIALIAS)
        start_time = time.time()
        results = classify_image(interpreter, image)
        elapsed_ms = (time.time() - start_time) * 1000
        label_id, prob = results[0]
        stream.seek(0)
        stream.truncate()

        prob = int(round(prob,2)*100)


        if prob > 70:
            history.append(labels[label_id])
            history = history[-5:]
            print(history)
            if history.count(history[0]) == len(history):
                state = history[0] # on considère que si 10 détections indiquent la meme info on peut s'y fier
                if previous_state != state:

                    if state == 'publicite':
                        print("on coupe le son")
                        os.system('/usr/local/bin/irsend SEND_ONCE TV KEY_MUTE -#2')
                        time.sleep(0.1)
                        os.system('/usr/local/bin/irsend SEND_ONCE TV KEY_VOLUMEDOWN -#2')
                    elif previous_state == 'publicite':
                        print("on remet le son")
                        os.system('/usr/local/bin/irsend SEND_ONCE TV KEY_MUTE -#2')
                        time.sleep(0.1)
                        os.system('/usr/local/bin/irsend SEND_ONCE TV KEY_VOLUMEUP -#2')

                    previous_state = state
         
        sys.stdout.flush()

        camera.annotate_text = '%s\nscore:%s%%' % (labels[label_id], prob)
    finally:
        GPIO.output(40, GPIO.LOW)
        print("fin")
        camera.stop_preview()


if __name__ == '__main__':
  signal.signal(signal.SIGTERM, terminateProcess)
  main()