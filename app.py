import cv2
import tempfile
import time
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from pillow_heif import HeifFile

import argparse
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import os

import easyocr


import re

import Levenshtein

import boto3
from botocore.config import Config
import pyheif
import io
from dotenv import load_dotenv

import json

assert insightface.__version__ >= '0.3'

app = Flask(__name__)


def reduce_pixel_size(image_path, reduction_factor):
    image = Image.open(image_path)

    width, height = image.size
    new_width = width // reduction_factor
    new_height = height // reduction_factor

    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    return resized_image

def check_idcard(timestamp ,idfile, id_num, th_fname, th_lname, en_fname, en_lname) :

    print(id_num)

    th_fname = th_fname.lower()
    th_lname = th_lname.lower()
    en_fname = en_fname.lower()
    en_lname = en_lname.lower()

    nocr = []
    data = [0, 0, 0, 0, 0]
    # data = [0, 0, 0]

    # model_storage_directory = './models'
    reader = easyocr.Reader(['th','en'], gpu=False)
    # reader = easyocr.Reader(['th','en'], model_storage_directory=model_storage_directory, download_enabled=False)
    # reader = easyocr.Reader(['en'], model_storage_directory=model_storage_directory, download_enabled=False)

    for i in range(4) :
        pocr = reader.readtext(idfile, detail=0, paragraph=True)
        print(pocr)
        # target_words = ['thai', 'national', 'id', 'card']
        target_words = ['thai', 'national', 'id', 'card', 'เลขประจำตัวประชาชน']

        contains_word = False
        for text_result in pocr:
            split_result = re.split("\s", text_result)
            split_result = [x.lower() for x in split_result]
            for word in target_words:
                if word in split_result:
                    contains_word = True
                    cv2.imwrite(f"id_{timestamp}.jpg", idfile)
                    break
            if contains_word :
                break
        if contains_word :
            break
        idfile = cv2.rotate(idfile, cv2.ROTATE_90_CLOCKWISE)
    if contains_word == False :
        return None
    

    ocr = reader.readtext(idfile)

    num_pattern = r'^-?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$'

    for item in ocr :
        coordinates, text, value = item
        text_without_spaces = text.replace(" ", "")
        if not bool(re.match(num_pattern, text_without_spaces)) :
            words = text.strip().split()
            for word in words:
                nocr.append((coordinates, word, value))
        else :
            nocr.append((coordinates, text_without_spaces, value))

    for item in nocr :
        coordinates, text, value = item
        id_distance = Levenshtein.distance(id_num.replace(" ", ""), text)
        th_fname_distance = Levenshtein.distance(th_fname, text)
        th_lname_distance = Levenshtein.distance(th_lname, text)
        en_fname_distance = Levenshtein.distance(en_fname, text)
        en_lname_distance = Levenshtein.distance(en_lname, text)
        if (id_distance <= 3) : 
            data[0] = coordinates
            print(f"{id_num} - {text} = {id_distance}")
        if (th_fname_distance <= 2) : 
            data[1] = coordinates
            print(f"{th_fname} - {text} = {th_fname_distance}")
        if (th_lname_distance <= 2) : 
            data[2] = coordinates
            print(f"{th_lname} - {text} = {th_lname_distance}")
        if (en_fname_distance <= 2) : 
            # data[1] = coordinates
            data[3] = coordinates
            print(f"{en_fname} - {text} = {en_fname_distance}")
        if (en_lname_distance <= 2) : 
            # data[2] = coordinates
            data[4] = coordinates
            print(f"{en_lname} - {text} = {en_lname_distance}")

    return data

def handle_heic(file) :
    if file.filename.endswith('.heic') or file.filename.endswith('.HEIC'):
        # Convert HEIC file to JPEG
        heif_file = HeifFile(file.read())
        image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        image.save(temp_file.name, format='JPEG')
    else:
        # Save the file with its original extension
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(file.read())

    temp_file.flush()
    return temp_file.name

def detect_face(img, name) :
    print("Start detect Face")

    parser = argparse.ArgumentParser(description='insightface app test')
    parser.add_argument('--ctx', default=0, type=int,
                        help='ctx id, <0 means using cpu')
    parser.add_argument('--det-size', default=640, type=int, help='detection size')
    args = parser.parse_args()
    app = FaceAnalysis()
    app.prepare(ctx_id=args.ctx, det_size=(args.det_size, args.det_size))
    faces = app.get(img)
    bounding_box = faces[0].bbox.astype(int)

    rimg = app.draw_on(img, faces)
    cv2.imwrite(f"./{name}.jpg", rimg)
    feats = []
    for face in faces:
        feats.append(face.normed_embedding)
    return np.array(feats, dtype=np.float32), bounding_box

def find_edge(box) :
    x_face = box[2] - box[0]
    y_face = box[3] - box[1]
    print(x_face, y_face)

    if (x_face < 200 or y_face < 200) :
        x_total = x_face * 10.5
        y_total = y_face * 5
    else :
        x_total = x_face * 10
        y_total = y_face * 4.5

    x1 = 0.82 * x_total 
    x_top = box[0] - x1
    print(f"x1: {x1}")

    x2 = 0.08 * x_total
    x_bot = box[2] + x2
    print(f"x2: {x2}")

    y1 = 0.55 * y_total
    print(f"y1: {y1}")
    y_top = box[1] - y1
    y2 = 0.25 * y_total
    print(f"y2: {y2}")
    y_bot = box[3] + y2

    top = (int(x_top), int(y_top))
    bot = (int(x_bot), int(y_bot))

    print(f"first: {x_top, y_top}")
    print(f"sec: {x_bot, y_bot}")

    return top, bot

def check_rectangles(top, bot, detect_word):
    x1_big, y1_big = top
    x2_big, y2_big = bot 
    error = 0
    for rectangle in detect_word:
        if isinstance(rectangle, list) and len(rectangle) == 4:
            for corner in rectangle:
                x, y = corner
                # Accept a little bit error because when find a edges can have some error
                if x < x1_big and x > x2_big and y < y1_big and y > y2_big:
                    return False
        else: 
            error = error + 1
    if (error > 1) :
        return False
    return True

def load_img_s3(key):
    load_dotenv()
    KEY = key
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    BUCKET_NAME = os.getenv('BUCKET_NAME')

    s3 = boto3.client('s3',aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    response = s3.get_object(Bucket=BUCKET_NAME, Key=KEY)
    content_type = response['ContentType']
    image_data = response['Body'].read()
    if (content_type.lower() == 'image/heic' or content_type.lower() == 'image/heif') :
        heif_image = pyheif.read_heif(io.BytesIO(image_data))
        image = Image.frombytes(
        heif_image.mode,
        heif_image.size,
        heif_image.data,
        "raw",
        heif_image.mode,
        heif_image.stride,
        )
    else : 
        image_stream = io.BytesIO(image_data)
        image = Image.open(image_stream)
    return image

@app.route('/valid/front', methods=['POST'])
def valid_front_data():
    print("Start")

    timestamp = int(time.time())    

    img_path = request.form['img']
    id_num = request.form['id_num']
    th_fname = request.form['th_fname']
    th_lname = request.form['th_lname']
    en_fname = request.form['en_fname']
    en_lname = request.form['en_lname']


    image = load_img_s3(img_path)
    print("load")

    file_path = f"id_{timestamp}.jpg"
    image.save(file_path)
    reduction_factor = 2
    resized_image = reduce_pixel_size(file_path, reduction_factor)
    resized_image.save(f"id_{timestamp}.jpg")

    id = cv2.imread(f'./id_{timestamp}.jpg')
    data = check_idcard(timestamp ,id, id_num, th_fname, th_lname, en_fname, en_lname)
    if data == [0,0,0,0,0] :
        response = {'Error': "Don't have data on image"}
        os.remove(f"id_{timestamp}.jpg")
        return jsonify(response), 501
    print(data)

    # img = Image.open(f"id_{timestamp}.jpg")
    # barcode
    # decoded_list = decode(img)
    # print(f"decoded_list:{decoded_list}")
    # for bar in decoded_list :
    #     bar_num = bar.data
    #     print(f"bar:{bar_num}")

    id = cv2.imread(f'./id_{timestamp}.jpg')
    id_feat, box = detect_face(id, "detect1")
    detect_img = cv2.imread('./detect1.jpg')

    top, bot = find_edge(box) 

    rect = cv2.rectangle(detect_img, top, bot, (255, 0, 0), 2)
    cv2.imwrite("rect.jpg", rect)

    result = check_rectangles(top, bot, data)
    if (result) :
        # result1 = collection.insert_one({'userId': 1,'id_feat': id_feat.tolist()})
        # if result1.acknowledged:
        #     response = {'Ok': 'ID features added successfully'}
        #     return jsonify(response), 201
        # else:
        #     response = {'Error': 'Failed to add ID features'}
        #     return jsonify(response), 500
        response = {'feat': id_feat.tolist()}
        os.remove(f"id_{timestamp}.jpg")
        return jsonify(response), 200
    else :
        os.remove(f"id_{timestamp}.jpg")
        return "Not correct Format"

@app.route('/valid/back', methods=['POST'])
def valid_back_data():
    timestamp = int(time.time())  

    img_path = request.form['img']

    image = load_img_s3(img_path)

    file_path = f"back_{timestamp}.jpg"
    image.save(file_path)

    pic = cv2.imread(f'./back_{timestamp}.jpg')

    reader = easyocr.Reader(['en'])
    result = reader.readtext(pic, detail = 0, paragraph=True)
    if not result :
        os.remove(f"back_{timestamp}.jpg")
        return jsonify({'error': 'Cannot find Laser code from id'}), 501

    lasercode_pattern = r'[a-zA-Z]{2}\d{1}-\d{7}-\d{2}'
    for txt in result :
        laser_code = re.search(lasercode_pattern, txt)
    if not laser_code :
        os.remove(f"back_{timestamp}.jpg")
        return jsonify({'error': 'Cannot find Laser code from id'}), 501
    os.remove(f"back_{timestamp}.jpg")
    return jsonify({'laser_code' : laser_code.group()})

@app.route('/face_recognition', methods=['POST'])
def insight_face() :

    timestamp = int(time.time())    

    img_path = request.form['img']
    id_feat_json = request.form['feat']
    id_feat = json.loads(id_feat_json)

    image = load_img_s3(img_path)

    file_path = f"face_{timestamp}.jpg"
    image.save(file_path)

    face = cv2.imread(f'./face_{timestamp}.jpg')
    face_feat, _ = detect_face(face, "detect2")

    if face_feat.shape[0] == 0 :
        face = cv2.rotate(face, cv2.ROTATE_180)
        cv2.imwrite(f"face_{timestamp}.jpg", face)
        face_feat, _ = detect_face(face,"detect2")

    if face_feat.shape[0] != 1 :
        os.remove(f"face_{timestamp}.jpg")
        response = {'Error': "Don't have face on image or have more than 1 person"}
        return jsonify(response), 500

    sims = np.dot(id_feat, face_feat.T)

    os.remove(f"face_{timestamp}.jpg")
    return jsonify({'sims': float(sims)})

@app.route("/")
def index():
    return "Hello World!"


if __name__ == "__main__" :
    app.run(debug=True, port=5000, host="0.0.0.0")