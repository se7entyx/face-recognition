from flask import Flask, request, jsonify
import cv2
import os
import numpy as np
import base64
from sklearn.svm import SVC
from deepface import DeepFace
from joblib import dump, load
import tempfile

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAKE_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'storage', 'app', 'public', 'palsu'))

def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    equalized = cv2.equalizeHist(blur)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

def extract_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        return image[y:y+h, x:x+w]
    return image

def calculate_lbp(image, radius=2, neighbors=16):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = np.zeros_like(gray)
    for i in range(radius, gray.shape[0] - radius):
        for j in range(radius, gray.shape[1] - radius):
            center = gray[i, j]
            binary_string = ''
            for n in range(neighbors):
                y = i + int(radius * np.sin(2 * np.pi * n / neighbors))
                x = j + int(radius * np.cos(2 * np.pi * n / neighbors))
                binary_string += '1' if gray[y, x] >= center else '0'
            lbp[i, j] = int(binary_string, 2) % 256
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    return hist / (np.sum(hist) + 1e-6)

def load_dataset(dataset_path):
    dataset_features, labels = [], []

    for filename in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, filename)
        image = cv2.imread(img_path)
        if image is not None:
            image = extract_face(preprocess(image))
            image = cv2.resize(image, (100, 100))
            dataset_features.append(calculate_lbp(image))
            labels.append('real')

    for filename in os.listdir(FAKE_PATH):
        img_path = os.path.join(FAKE_PATH, filename)
        image = cv2.imread(img_path)
        if image is not None:
            image = extract_face(preprocess(image))
            image = cv2.resize(image, (100, 100))
            dataset_features.append(calculate_lbp(image))
            labels.append('fake')

    return np.array(dataset_features), np.array(labels)

def train_model(dataset_path, model_path):
    X, y = load_dataset(dataset_path)
    model = SVC(kernel='rbf', probability=True, C=10.0, gamma='scale')
    model.fit(X, y)
    dump(model, model_path)
    return model

@app.route('/verify', methods=['POST'])
def verify_face():
    data = request.get_json()
    dataset_path = data.get('dataset_path')
    base64_image = data.get('image')

    if not dataset_path or not base64_image:
        return jsonify({'result': False, 'error': 'Invalid input'}), 400

    model_path = os.path.join(dataset_path, 'svm_model.joblib')

    try:
        image_data = base64.b64decode(base64_image.split(',')[-1])
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(image_data)
            test_image_path = f.name
    except Exception as e:
        return jsonify({'result': False, 'error': f'Base64 decode failed: {e}'}), 400

    image = cv2.imread(test_image_path)
    if image is None:
        return jsonify({'result': False, 'error': 'Cannot read image'}), 400

    image = extract_face(preprocess(image))
    image = cv2.resize(image, (100, 100))
    lbp_feature = calculate_lbp(image)

    if os.path.exists(model_path):
        model = load(model_path)
    else:
        model = train_model(dataset_path, model_path)

    try:
        prediction = model.predict([lbp_feature])[0]
        confidence = model.predict_proba([lbp_feature]).max()
    except Exception as e:
        return jsonify({'result': False, 'error': f'Prediction error: {e}'}), 500

    try:
        result = DeepFace.find(img_path=test_image_path, db_path=dataset_path, model_name="ArcFace", detector_backend="mtcnn", enforce_detection=False)
        face_match = len(result[0]) > 0
    except Exception as e:
        face_match = False

    os.remove(test_image_path)

    is_valid = confidence < 0.8 and face_match
    return jsonify({'result': is_valid, 'confidence': float(confidence), 'face_match': face_match})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)