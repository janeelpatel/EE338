from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import random
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import pickle
# import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
# from tqdm import tqdm_notebook
from scipy import signal
from scipy.io.wavfile import read, write
from numpy.fft import fft, ifft
# from google.colab import drive
from torch.autograd import Variable
# from PIL import Image
import soundfile as sf

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# print(torch.__version__)

# Define a flask app
app = Flask(__name__)


class AdaptiveBatchNorm2d(nn.Module):
    def _init_(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(AdaptiveBatchNorm2d, self)._init_()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine)
        self.a = nn.Parameter(torch.FloatTensor(1, 1, 1, 1))
        self.b = nn.Parameter(torch.FloatTensor(1, 1, 1, 1))

    def forward(self, x):
        return self.a * x + self.b * self.bn(x)


class vgg_type(nn.Module):
  def _init_(self):
    super(vgg_type, self)._init_()
    self.conv1 = nn.Conv2d(1, 64, [1,3], padding=[0,1], bias = False)
    self.norm1 = AdaptiveBatchNorm2d(64)
    self.relu = nn.ReLU(inplace = True)
    self.lrelu = nn.LeakyReLU(0.2, inplace = True)

    self.conv2 = nn.Conv2d(64,64,[1,3], padding = [0,1], dilation = 1, bias = False)
    self.conv3 = nn.Conv2d(64,64,[1,3], padding = [0,2], dilation = 2, bias = False)
    self.conv4 = nn.Conv2d(64,64,[1,3], padding = [0,4], dilation = 4, bias = False)
    self.conv5 = nn.Conv2d(64,64,[1,3], padding = [0,8], dilation = 8, bias = False)
    self.conv6 = nn.Conv2d(64,64,[1,3], padding = [0,16], dilation = 16, bias = False)
    self.conv7 = nn.Conv2d(64,64,[1,3], padding = [0,32], dilation = 32, bias = False)
    self.conv8 = nn.Conv2d(64,64,[1,3], padding = [0,64], dilation = 64, bias = False)
    self.conv9 = nn.Conv2d(64,64,[1,3], padding = [0,128], dilation = 128, bias = False)
    self.conv10 = nn.Conv2d(64,64,[1,3], padding = [0,256], dilation = 256, bias = False)
    self.conv11 = nn.Conv2d(64,64,[1,3], padding = [0,512], dilation = 512, bias = False)
    self.conv12 = nn.Conv2d(64,64,[1,3], padding = [0,1024], dilation = 1024, bias = False)
    self.conv13 = nn.Conv2d(64,64,[1,3], padding = [0,2048], dilation = 2048, bias = False)
    self.conv14 = nn.Conv2d(64,64,[1,3], padding = [0,1], bias = False)
 
    self.norm2 = AdaptiveBatchNorm2d(64)
    self.norm3 = AdaptiveBatchNorm2d(64)
    self.norm4 = AdaptiveBatchNorm2d(64)
    self.norm5 = AdaptiveBatchNorm2d(64)
    self.norm6 = AdaptiveBatchNorm2d(64)
    self.norm7 = AdaptiveBatchNorm2d(64)
    self.norm8 = AdaptiveBatchNorm2d(64)
    self.norm9 = AdaptiveBatchNorm2d(64)
    self.norm10 = AdaptiveBatchNorm2d(64)
    self.norm11 = AdaptiveBatchNorm2d(64)
    self.norm12 = AdaptiveBatchNorm2d(64)
    self.norm13 = AdaptiveBatchNorm2d(64)
    self.norm14 = AdaptiveBatchNorm2d(64)

    self.final = nn.Conv2d(64,1, [1,1])

  def forward(self, x):
    x = self.conv1(x)
    x = self.norm1(x)
    x = self.relu(x)

    x = self.conv2(x)
    x = self.norm2(x)
    x = self.lrelu(x)

    x = self.conv3(x)
    x = self.norm3(x)
    x = self.lrelu(x)

    x = self.conv4(x)
    x = self.norm4(x)
    x = self.lrelu(x)

    x = self.conv5(x)
    x = self.norm5(x)
    x = self.lrelu(x)

    x = self.conv6(x)
    x = self.norm6(x)
    x = self.lrelu(x)

    x = self.conv7(x)
    x = self.norm7(x)
    x = self.lrelu(x)

    x = self.conv8(x)
    x = self.norm8(x)
    x = self.lrelu(x)

    x = self.conv9(x)
    x = self.norm9(x)
    x = self.lrelu(x)

    x = self.conv10(x)
    x = self.norm10(x)
    x = self.lrelu(x)

    x = self.conv11(x)
    x = self.norm11(x)
    x = self.lrelu(x)

    x = self.conv12(x)
    x = self.norm12(x)
    x = self.lrelu(x)

    x = self.conv13(x)
    x = self.norm13(x)
    x = self.lrelu(x)

    x = self.conv14(x)
    x = self.norm14(x)
    x = self.lrelu(x)

    x = self.final(x)
    return x

print('yes')

# Load your trained model

checkpoints = torch.load('audio_raw_updated.ckpt.t7', map_location = torch.device('cpu')) ## load the model dict
model = checkpoints['model']  # load model from the dict

# print('lol')

# -------------------------------------------------------

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(aud_path, model):

    y, sr = librosa.load(aud_path) # convert the input audio signal into numpy ndarray
    y = torch.from_numpy(y) # convert numpy into torch tensor
    y = y.unsqueeze_(0).unsqueeze_(0).unsqueeze_(0) #.cuda() # matching the dimensions
    output = model(y)
    output = output.squeeze_(0).squeeze_(0).detach().cpu().numpy() # convert output into 1D numpy array
    output = output.reshape((-1,1))
    # output = output.astype(np.int32)
    # librosa.output.write_wav('results/output1.wav', output, sr)
    print(output.shape)
    sf.write('results/output.wav', output, sr) # , 'PCM_24')
    # print(sr)
    return True


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        return str(preds)
    return None


if __name__ == '__main__':
    app.run(debug=True)