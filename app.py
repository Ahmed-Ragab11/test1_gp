from io import BytesIO
import cv2
from flask import Flask, render_template, request, jsonify
from matplotlib import pyplot as plt
from src.modeling.run_model_single import (
    load_model, load_inputs, process_augment_inputs, batch_to_tensor
)
from src.optimal_centers.get_optimal_center_single import get_optimal_center_single
from src.cropping.crop_single import crop_single_mammogram
from urllib.request import urlopen
from PIL import Image
import numpy as np
from pydicom import dcmread    
import urllib           

# Initializations for the model
shared_parameters = {
"device_type": "gpu",
"gpu_number": 0,
"max_crop_noise": (100, 100),
"max_crop_size_noise": 100,
"batch_size": 1,
"seed": 0,
"augmentation": True,
"use_hdf5": True,
}
random_number_generator = np.random.RandomState(shared_parameters["seed"])
image_only_parameters = shared_parameters.copy()



app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/mammo/predict/', methods=['POST'])
def predict():
    data = request.get_json()
    imageURL = data['imageURL']
    viewData = data['view']
    # imageURL = request.args.get('imageURL')
    # token = request.args.get('token')
    # viewData = request.args.get('view')
    # Initializations for the model
    image_only_parameters["view"] = viewData[0]+'-'+viewData[2:]
    image_only_parameters["use_heatmaps"] = False
    image_only_parameters["model_path"] = "models/ImageOnly__ModeImage_weights.p"
    model, device = load_model(image_only_parameters)
    # File Paths
    cropped_img_path = 'sample_single_output/' + 'out_file.png'
    metadata_path = 'sample_single_output/' + 'metadata' + '.pkl'
    # Preprocessing
    crop_single_mammogram(imageURL, "NO", viewData, cropped_img_path, metadata_path, 100, 50)
    get_optimal_center_single(cropped_img_path, metadata_path)
    # Load Inputs
    model_input = load_inputs(
    image_path=imageURL,
    metadata_path=metadata_path,
    use_heatmaps=False,
    )
    batch = [
    process_augment_inputs(
        model_input=model_input,
        random_number_generator=random_number_generator,
        parameters=image_only_parameters,
    ),
    ]
    # Classification
    tensor_batch = batch_to_tensor(batch, device)
    y_hat = model(tensor_batch)
    predictions = np.exp(y_hat.cpu().detach().numpy())[:, :2, 1]
    predictions_dict = {
        "benign": float(predictions[0][0]),
        "malignant": float(predictions[0][1]),
    }

    predBen = round(predictions_dict['benign'], 3) * 100
    predMal = round(predictions_dict['malignant'], 3) * 100
    
    result = {
        'benign': predBen,
        'malignant': predMal
    }

    return jsonify(result)

@app.route('/mammo/dicom-to-png/', methods=['POST'])
def dicomToPng():
    data = request.get_json()
    imageURL = data['imageURL']
    image = dcmread(BytesIO(data), force=True)
    # image = (image - image.min()) / (image.max() - image.min()) * 255.0  
    # image = image.astype(np.uint8)
    # cv2.imwrite('hello.png', image)
    return ''


if __name__ == '__main__':
    app.run(port=5000, debug=True)