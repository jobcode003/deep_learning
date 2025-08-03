import numpy as np
from keras.models import load_model
from keras.preprocessing import image
model=load_model('best_model.h5')
class_names=['Corn___Common_rust','Corn___healthy','Corn___Northern_Leaf_Blight','Potato___Early_blight'
    ,'Potato___healthy','Potato___Late_blight','Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___healthy',
    'Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Tomato_mosaic_virus','Tomato___Tomato_Yellow_Leaf_Curl_Virus']
def process(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize just like Rescaling(1./255)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence
image__path='C:\\Users\\PC\\Desktop\\computer_vision\\crop_images\\Potato___Early_blight\\image (296).jpg'
predicted_class, confidence = process(image__path)
print(f"Predicted: {predicted_class} ({confidence}% confidence)")
if predicted_class == 'Potato___Early_blight':
    print("what courses it? ",'Common rust is caused by the fungus Puccinia sorghi and occurs every growing season')
