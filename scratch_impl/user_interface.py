import os
import shutil
from flask import Flask, request, send_from_directory, render_template
import tempfile
import cv2
import numpy as np
from similarity import run_similarity_detector
from utility import pyramid_detect, load_from_disk, load_and_preprocess_image, apply_crop
from cascade import *

UPLOAD_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'scratch_impl', 'data', 'uploads'
)

STATIC_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'scratch_impl'
)

try:  # Reset saved files on each start
    shutil.rmtree(UPLOAD_FOLDER, True)
    os.mkdir(UPLOAD_FOLDER)
except OSError:
    pass

app = Flask(__name__, static_folder=STATIC_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    
    """Upload image and display on page"""
    if request.method == 'POST' and 'photo' in request.files:

        filename = save_file(request.files['photo'])
        print(filename)
        
        # load the image uploaded
        input_image = load_and_preprocess_image(os.getcwd() + "/data/uploads/" + filename)
        
        # load the cascade model from disk
        cascade_file = 'sav_files/cascade_lfw_cbcl_t2_3_features_v1.sav'
        test_cascade = load_from_disk(cascade_file)
        
        # detect faces at multiple scales
        bounding_boxes = pyramid_detect(input_image, cascade=test_cascade, cascade_window_size = 24, scale=1.6)
        
        clone = input_image.copy()
        cropped = apply_crop(bounding_boxes, clone)
        
        # filename = photos.save(request.files['photo'])
        celebrity_filename = run_similarity_detector("/data/uploads/cropped.jpg").replace(" ", "%20")

        context = {"original": filename, "celebrity": celebrity_filename}
        
        return render_template('index.html', **context)
    
    return render_template('index.html')


def save_file(file):
    # Save POST request's file object to a temp file
    dummy, temp_filename = tempfile.mkstemp()
    file.save(temp_filename)

    # Compute filename
    hash_filename = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    # Move temp file to permanent location
    shutil.move(temp_filename, hash_filename)

    return file.filename


if __name__ == '__main__':
    app.run()