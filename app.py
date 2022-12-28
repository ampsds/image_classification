import io
import json
import os

import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage

from PIL import Image
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
#model = models.densenet121(pretrained=True)               # Trained on 1000 classes from ImageNet
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.eval()                                              # Turns off autograd and

##ラインボット
CHANNEL_ACCESS_TOKEN = os.getenv('CHANNEL_ACCESS_TOKEN')
CHANNEL_SECRET = os.getenv('CHANNEL_SECRET')

line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

##推論
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

# Get a prediction
def get_prediction(input_tensor):
    outputs = model.forward(input_tensor)                 # Get likelihoods for all ImageNet classes
    _, y_hat = outputs.max(1)                             # Extract the most likely class
    prediction = y_hat.item()                             # Extract the int value from the PyTorch tensor
    return prediction

# Make the prediction human-readable
def render_prediction(prediction_idx):
    stridx = str(prediction_idx)
    class_name = 'Unknown'
    if img_class_map is not None:
        if stridx in img_class_map is not None:
            class_name = img_class_map[stridx][1]
    return prediction_idx, class_name

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.after_request
def after_request(response):
  # response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  return response

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file is not None:
            input_tensor = transform_image(file)
            prediction_idx = get_prediction(input_tensor)
            class_id, class_name = render_prediction(prediction_idx)
            return jsonify({'class_id': class_id, 'class_name': class_name})

#LINE BOTウェブフック
@app.route("/callback", methods=['POST'])
def callback():
  signature = request.headers['X-Line-Signature']
  body = request.get_data(as_text=True)
  app.logger.info("Request body: " + body)

  try:
    handler.handle(body, signature)
  except InvalidSignatureError:
    abort(400)

  return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
  line_bot_api.reply_message(
    event.reply_token,
    TextSendMessage(text=event.message.text))

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    content = line_bot_api.get_message_content(event.message.id)
    img = b""
    for chunk in content.iter_content():
        img += chunk
        
    input_tensor = transform_image(img)
    prediction_idx = get_prediction(input_tensor)
    class_id, class_name = render_prediction(prediction_idx) 

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=class_name))
