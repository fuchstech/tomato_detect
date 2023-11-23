import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QImage, QPixmap, QIcon
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import tensorflow as tf
import cv2

class DomatesHastalikArayuz(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Domates Hastalıkları Tespit')
        self.setGeometry(100, 100, 600, 400)

        # Set the window icon
        self.setWindowIcon(QIcon(r'C:\Users\dest4\Desktop\tomatis\domato_fox.png'))  # Icon dosyanızın yolu

        # QLabel for displaying the image
        self.image_label = QLabel(self)
        self.image_label.setScaledContents(True)

        # QLabel for displaying the prediction result
        self.prediction_label = QLabel('Tahmin: ', self)

        # QPushButton for opening an image file
        self.open_button = QPushButton('Resim Aç', self)
        self.open_button.clicked.connect(self.open_image)

        # QPushButton for making predictions
        self.predict_button = QPushButton('Tahmin Yap', self)
        self.predict_button.clicked.connect(self.make_prediction)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.open_button)
        layout.addWidget(self.image_label)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.prediction_label)

        self.setLayout(layout)

        self.show()

    def open_image(self):
        # Open a file dialog to get the path of the image file
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(self, 'Resim Aç', '', 'Image Files (*.png *.jpg *.jpeg)')

        # Display the selected image
        pixmap = cv2.imread(image_path)
        pixmap = cv2.cvtColor(pixmap, cv2.COLOR_BGR2RGB)
        height, width, channel = pixmap.shape
        bytes_per_line = 3 * width
        q_image = QImage(pixmap.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_image))

        # Save the path for later use
        self.image_path = image_path

    def make_prediction(self):
        # Load the model
        model = tf.keras.models.load_model(r'C:\Users\dest4\Desktop\tomatis\tomate.h5')

        # Load class labels
        class_labels = [
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

        # Load and preprocess the test image
        test_image = image.load_img(self.image_path, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0

        # Perform inference
        predictions = model.predict(test_image)
        predicted_class = np.argmax(predictions)

        # Print the classification label
        prediction_label = class_labels[predicted_class]
        print(f'Predicted Class: {prediction_label}')

        # Update the QLabel with the prediction result
        self.prediction_label.setText(f'Tahmin: {prediction_label}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DomatesHastalikArayuz()
    sys.exit(app.exec_())
