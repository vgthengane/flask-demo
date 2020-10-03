import os
import cv2
import urllib
import numpy as np

from app import app
from app.model.infer import predict
from flask import request, jsonify, render_template, redirect, flash
from werkzeug.utils import secure_filename

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app.secret_key = "secret key"
app.config["max_content_length"] = 16 * 1920 * 1920

path = os.getcwd()
upload_folder = os.path.join("uploads")

if not os.path.isdir(upload_folder):
    os.makedirs(upload_folder)

app.config["upload_folder"] = upload_folder
allowed_extentions = ["png", "jpeg", "jpg", "tiff", "gif"]


def allowed_file(file_name):
    return "." in file_name and file_name.rsplit(".", 
                        1)[1].lower() in allowed_extentions


@app.route("/")
def home():
    return render_template("upload.html")


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



@app.route("/", methods=["POST"])
def upload_file():

    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part.")
            return redirect(request.url)

    file_ = request.files["file"]
    if file_.filename == "":
        flash("No file selected for uploading.")
        return redirect(request.url)

    if file_ and allowed_file(file_.filename):
        file_name = secure_filename(file_.filename)
        file_.save(os.path.join(app.config["upload_folder"], file_name))
        flash("File uploaded successfully.")
        return redirect("/")

    else:
        flash("Allowed file type is ['png', 'jpeg', 'jpg', 'tiff', 'gif']")
        return redirect(request.url)

    

if __name__ == "app.views":

    app.run(host="0.0.0.0", debug=True, port=5000)
