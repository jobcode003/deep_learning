from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
model=load_model('best_model.h5')

class_names = ['Corn___Common_rust', 'Corn___healthy', 'Corn___Northern_Leaf_Blight', 'Potato___Early_blight'
    , 'Potato___healthy', 'Potato___Late_blight', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
               'Tomato___healthy',
               'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Tomato_mosaic_virus',
               'Tomato___Tomato_Yellow_Leaf_Curl_Virus']
def process_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize just like Rescaling(1./255)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence
@app.route('/')
def index():
  return render_template('input.html')
@app.route('/process',methods=['GET','POST'])
def process():
    if 'image' not in request.files:
        return 'No image uploaded', 400
    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400

    # Save the file
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Process the image
    prediction, confidence = process_image(file_path)

    # Optionally, remove the uploaded file after processing
    os.remove(file_path)

    return render_template('input.html', prediction=prediction, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)