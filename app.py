import io
import json
import os

import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = models.densenet121(pretrained=True)               # Trained on 1000 classes from ImageNet
#model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.eval()                                              # Turns off autograd and

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


