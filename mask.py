# web-app for API image manipulation
from flask import Flask, request, render_template, send_from_directory
import tensorflow.contrib.eager as tfe
import os
import pathlib
from PIL import Image
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K
#import segmentation_models
import segmentation_models as sm
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers, layers
import segmentation_models as sm

#https://jinglescode.github.io/datascience/2019/12/02/biomedical-image-segmentation-u-net-nested/
#Unet++

class conv_block_nested(tf.keras.Model):

    def __init__(self, in_ch):
        super(conv_block_nested, self).__init__()
        self.activation = layers.Activation('relu')
        self.conv1 = layers.Conv2D(in_ch, 3, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(in_ch, 3, padding='same')
        self.bn2 = layers.BatchNormalization()

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output

class Nested_UNet(tf.keras.Model):

    def __init__(self, out_ch=3):
        super(Nested_UNet, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = layers.MaxPooling2D(3, strides=2, padding='same') #2 or 3 ?
        self.Up = layers.UpSampling2D(2)

        self.conv0_0 = conv_block_nested(filters[0])
        self.conv1_0 = conv_block_nested(filters[0])
        self.conv2_0 = conv_block_nested(filters[1])
        self.conv3_0 = conv_block_nested(filters[2])
        self.conv4_0 = conv_block_nested(filters[3])

        self.conv0_1 = conv_block_nested(filters[0])
        self.conv1_1 = conv_block_nested(filters[1])
        self.conv2_1 = conv_block_nested(filters[2])
        self.conv3_1 = conv_block_nested(filters[3])

        self.conv0_2 = conv_block_nested(filters[0])
        self.conv1_2 = conv_block_nested(filters[1])
        self.conv2_2 = conv_block_nested(filters[2])

        self.conv0_3 = conv_block_nested(filters[0])
        self.conv1_3 = conv_block_nested(filters[1])

        self.conv0_4 = conv_block_nested(filters[0])

        self.final = tf.keras.layers.Conv2D(out_ch, 3, activation='sigmoid', padding='same')

    def call(self, x):
        
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(tf.keras.layers.concatenate([x0_0, self.Up(x1_0)]))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(tf.keras.layers.concatenate([x1_0, self.Up(x2_0)]))
        x0_2 = self.conv0_2(tf.keras.layers.concatenate([x0_0, x0_1, self.Up(x1_1)]))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(tf.keras.layers.concatenate([x2_0, self.Up(x3_0)]))
        x1_2 = self.conv1_2(tf.keras.layers.concatenate([x1_0, x1_1, self.Up(x2_1)]))
        x0_3 = self.conv0_3(tf.keras.layers.concatenate([x0_0, x0_1, x0_2, self.Up(x1_2)]))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(tf.keras.layers.concatenate([x3_0, self.Up(x4_0)]))
        x2_2 = self.conv2_2(tf.keras.layers.concatenate([x2_0, x2_1, self.Up(x3_1)]))
        x1_3 = self.conv1_3(tf.keras.layers.concatenate([x1_0, x1_1, x1_2, self.Up(x2_2)]))
        x0_4 = self.conv0_4(tf.keras.layers.concatenate([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)]))

        output = self.final(x0_4)
        
        return output

def parse_image(filename, resize = True):
  '''
  Reads an image from a file,
  decodes it into a dense tensor,
  and resizes it to a fixed shape
  '''
  image = tf.io.read_file(filename)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  if resize:
    image = tf.image.resize(image, [256, 256])
  return image

def dice_coef(y_true, y_pred, smooth=K.epsilon()):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
dependencies = {'dice_coef':dice_coef, 'dice_loss':sm.losses.dice_loss }

# get Predictions
def predict(file_path):
  '''
  Takes image path and returns input image and Predicted mask
  '''
  image = parse_image(file_path, resize = True)
  test1 = tf.data.Dataset.from_tensor_slices([image])
  # Loading best model
  model = Nested_UNet(3)
  model.compile(optimizer='adam', loss=sm.losses.dice_loss, metrics=[dice_coef])
  te = test1.batch(1)
  model.predict(te)
  model.load_weights('model/car_damage_unet++.h5')
  #model = tf.keras.models.load_model('C:/Users/Narender Nani/Documents/Python Scripts/upload_file_python-src/car_damage_final_Nested_unet1', custom_objects=dependencies)
  for image in test1.take(1):
    pred_mask = model.predict(image[tf.newaxis, ...])[0]
  return image, pred_mask

# https://pythontic.com/image-processing/pillow/blend
# Function to change the image size
def changeImageSize(maxWidth, 
                    maxHeight, 
                    image):
    
    widthRatio  = maxWidth/image.size[0]
    heightRatio = maxHeight/image.size[1]

    newWidth    = int(widthRatio*image.size[0])
    newHeight   = int(heightRatio*image.size[1])

    newImage    = image.resize((newWidth, newHeight))
    return newImage

# https://github.com/gxercavins/image-api
app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


# default access page
@app.route("/")
def main():
    return render_template('index.html')


# upload selected image and forward to processing page
@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'static/images/')

    # create image directory if not found
    if not os.path.isdir(target):
        os.mkdir(target)

    # retrieve file from html file-picker
    upload = request.files.getlist("file")[0]
    print("File name: {}".format(upload.filename))
    filename = upload.filename

    # file support verification
    ext = os.path.splitext(filename)[1]
    if (ext == ".jpg") or (ext == ".png") or (ext == ".bmp") or (ext == ".jpeg"):
        print("File accepted")
    else:
        return render_template("error.html", message="The selected file is not supported"), 400

    # save file
    destination = "/".join([target, filename])
    print("File saved to to:", destination)
    upload.save(destination)

    # forward to processing page
    return render_template("processing.html", image_name=filename)


# Image segmentation
@app.route("/Predict", methods=["POST"])
def Predict():

    # open and process image
    target = os.path.join(APP_ROOT, 'static/images')
    destination = "/".join([target, filename])
    data_dir = pathlib.Path(target)
    file_path = destination
    
    img = Image.open(destination)

    if mode == 'Predict':
        image1, pred_mask = predict(file_path)
        image = tf.keras.preprocessing.image.array_to_img(image1)
        mask = tf.keras.preprocessing.image.array_to_img(pred_mask)
        pred = Image.blend(image, mask, alpha=.6)
        final = changeImageSize(img.size[0], img.size[1], pred)
    else:
        return render_template("error.html", message="Failed to preprocess the image"), 400

    # save and return image
    destination = "/".join([target, 'temp1.png'])
    if os.path.isfile(destination):
        os.remove(destination)
    final.save(destination)

    return send_image('temp1.png')
    # forward to processing page
    #return render_template("predict.html")


# retrieve file from 'static/images' directory
@app.route('/static/images/<filename>')
def send_image(filename):
    return send_from_directory("static/images", filename)


if __name__ == "__main__":
    app.run(port=5000, debug=True)

