from __future__ import print_function

import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect, send_file
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow.keras.backend as K
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import base64
import numpy as np
from util import base64_to_pil, img_to_base64
from PIL import Image

# Declare a flask app
app = Flask(__name__)


#print('Model loaded. Check http://127.0.0.1:5000/')


# Model saved with Keras model.save()
MODEL_PATH = 'models/covid_detector.h5'

# Load your own trained model
model = load_model(MODEL_PATH)

labels=['covid','normal']
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')
jpeg_content = 'image/jpeg'

def model_predict(img, model):
    
    img = img.resize((224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/ar', methods=['GET'])
def index_ar():
    # Main page
    return render_template('index_ar.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json).convert('RGB')
        #print(request.files['file'])

        # Save the image to ./uploads
        #img.save("uploads/image.png")

        # Make prediction
        preds = model_predict(img, model)
        gradcam = gradcam_from_img(img)
        # Process your result for human
        if(preds[0][0]>=preds[0][1]):
            pred_class="POSITIVE"
        else:
            pred_class="NEGATIVE"

        pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        
        # Serialize the result, you can add additional fields
        return jsonify(result=pred_class, probability=pred_proba, file = img_to_base64(gradcam))

    return None

def gradcam_from_img(pil_image):
  open_cv_image = np.array(pil_image) 
  obj = open_cv_image[:, :, ::-1].copy() 
  obj = cv2.resize(obj, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
  x = img_to_array(obj)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)

  preds = model.predict(x)
  class_idx = np.argmax(preds[0])
  print(class_idx)
  class_output = model.output[:, class_idx]
  last_conv_layer = model.get_layer("block5_conv3")
  grads = K.gradients(class_output, last_conv_layer.output)[0]
  pooled_grads = K.mean(grads, axis=(0, 1, 2))
  iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

  pooled_grads_value, conv_layer_output_value = iterate([x])

  for i in range(64):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

  heatmap = np.mean(conv_layer_output_value, axis = -1)
  heatmap = np.maximum(heatmap, 0)
  heatmap /= np.max(heatmap)

  heatmap = cv2.resize(heatmap, (obj.shape[1], obj.shape[0]))
  heatmap = np.uint8(255 * heatmap)
  heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
  superimposed_img = cv2.addWeighted(obj, 0.5, heatmap, 0.5, 0)
  obj = cv2.resize(obj, dsize=(300, 300), interpolation=cv2.INTER_CUBIC)
  superimposed_img = cv2.resize(superimposed_img, dsize=(300, 300), interpolation=cv2.INTER_CUBIC)
  superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
  return superimposed_img

if __name__ == '__main__':

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
