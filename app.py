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

img_class_map = None
mapping_file_path = 'index_to_name.json'                  # Human-readable names for Imagenet classes
if os.path.isfile(mapping_file_path):
    with open (mapping_file_path) as f:
        img_class_map = json.load(f)

# Transform input into the form our model expects
def transform_image(infile):
    input_transforms = [transforms.Resize(255),           # We use multiple TorchVision transforms to ready the image
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],       # Standard normalization for ImageNet model input
            [0.229, 0.224, 0.225])]
    my_transforms = transforms.Compose(input_transforms)
    image = Image.open(infile)                            # Open the image file
    timg = my_transforms(image)                           # Transform PIL image to appropriately-shaped PyTorch tensor
    timg.unsqueeze_(0)                                    # PyTorch models expect batched input; create a batch of 1
    return timg

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


