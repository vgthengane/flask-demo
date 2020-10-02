import os
import cv2
import urllib
import numpy as np

from app import app
from app.model.infer import predict
from flask import request, jsonify, render_template

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/test", methods=["POST"])
def process():
    data = request.get_json()
    image_URLs = data["s3_URLs"]

    out_dict = {}

    for idx, image_URL in enumerate(image_URLs):
        image = urllib.request.urlopen(image_URL)
        image = np.asarray(bytearray(image.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        preds, category = predict(image)
        out_dict[image_URL] = preds
        print(f"Predicted as \"{category}\" image.")
        print(f"{idx} images are done out of {len(image_URLs)}")
        
    return jsonify(out_dict)

if __name__ == "app.views":

    app.run(host="0.0.0.0", debug=True, port=5000)
