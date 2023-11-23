import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model(r'C:\Users\dest4\Desktop\tomatis\tomate.h5')

class_labels = domates_hastaliklari = [
    'Domates Bakteriyel Leke',
    'Domates Erken Yaprak Yanıklığı',
    'Domates Geç Yaprak Yanıklığı',
    'Domates Yaprak Küfü',
    'Domates Septoria Yaprak Lekesi',
    'Domates Örümcek Akarları - İki Noktalı Örümcek Akarı',
    'Domates Hedef Lekesi',
    'Domates Sarı Yaprak Kıvrım Virüsü',
    'Domates Mozaik Virüsü',
    'Domates Sağlıklı'
]

test_image_path = r"C:\Users\dest4\Desktop\tomatis\tomato dataset\val\Tomato___Bacterial_spot\0ab9c705-f29e-45ac-b786-9549b3c38f16___GCREC_Bact.Sp 3223.JPG"
test_image = image.load_img(test_image_path, target_size=(224,224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

# Preprocess the test image (rescale to [0, 1])
test_image = test_image / 255.0

# Perform inference
predictions = model.predict(test_image)
#print(predictions)
predicted_class = np.argmax(predictions)

# Print the classification label
print(f'Predicted Class: {class_labels[predicted_class]}')