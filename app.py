from flask import Flask, render_template, Response, request, redirect, url_for, flash
import cv2
import dlib
import numpy as np
import pickle
import base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)
app.secret_key = 'summa'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('YukeshCNN.dat')
face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

known_face_encodings = []
known_face_names = []
known_face_pins = []

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        pin = request.form['pin']
        image_data = request.form['image']

        image_data = image_data.split(',', 1)[1]

        image_bytes = base64.b64decode(image_data)

        image = Image.open(BytesIO(image_bytes))

        image = np.array(image.convert('RGB'))

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        faces = detector(gray)
        for face in faces:
            shape = predictor(gray, face)
            face_descriptor = face_rec_model.compute_face_descriptor(image, shape)
            known_face_encodings.append(np.array(face_descriptor))
            known_face_names.append(username)
            known_face_pins.append(pin)

        with open('face_encodings.pkl', 'wb') as f:
            pickle.dump((known_face_encodings, known_face_names, known_face_pins), f)

        flash('Registered successfully')
        return redirect(url_for('home'))
    else:
        return render_template('register.html')
@app.route('/pin', methods=['GET', 'POST'])
def pin():
    if request.method == 'POST':
        pin = request.form['pin']

        with open('face_encodings.pkl', 'rb') as f:
            known_face_encodings, known_face_names, known_face_pins = pickle.load(f)

        if pin in known_face_pins:
            return redirect(url_for('access'))
        else:
            flash('Invalid PIN')
            return redirect(url_for('home'))  # Redirect to home page
    else:
        return render_template('pin.html')

@app.route('/access')
def access():
    video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    name = "Unknown"
    for face in faces:
        shape = predictor(gray, face)
        face_descriptor = np.array(face_rec_model.compute_face_descriptor(frame, shape))

        with open('face_encodings.pkl', 'rb') as f:
            known_face_encodings, known_face_names, known_face_pins = pickle.load(f)

        matches = [np.linalg.norm(face_descriptor - known_face, ord=2) for known_face in known_face_encodings]
        name = "Unknown"

        if min(matches) <= 0.6:
            best_match_index = np.argmin(matches)
            name = known_face_names[best_match_index]
        else:
            flash('Unknown face')
            return redirect(url_for('home'))  # Redirect to home page

    return render_template('access.html', name=name)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)