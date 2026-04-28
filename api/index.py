import os
from flask import Flask, render_template, request, send_file, jsonify
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__, template_folder='../templates')

def process_piso_coins(img):
    """Your specific Piso Coin counting logic."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
        param1=50, param2=30, minRadius=20, maxRadius=100
    )
    count = 0
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        count = len(circles)
        for (x, y, r) in circles:
            cv2.circle(img, (x, y), r, (0, 255, 0), 4)
    return img, count

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['image']
    filter_type = request.form.get('filter')
    
    # Convert upload to OpenCV format
    filestr = file.read()
    npimg = np.frombuffer(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    result_img = img.copy()
    message = ""

    if filter_type == 'grayscale':
        result_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif filter_type == 'piso':
        result_img, count = process_piso_coins(img)
        message = f"Found {count} coins!"
    
    # Convert back to send to browser
    is_success, buffer = cv2.imencode(".png", result_img)
    io_buf = io.BytesIO(buffer)
    return send_file(io_buf, mimetype='image/png')

# Necessary for Vercel
app.run()
