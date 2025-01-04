import uuid
from torch import embedding
import torch.nn.functional as F
from flask import Flask, render_template, request, url_for, redirect, jsonify
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from model.image_embedding import get_image_embedding
from database import insert_photo, insert_user, get_all_biometric_templates, calculate_and_update_biometric_template, get_user_biometric_template, get_keys_by_user_id, insert_key
from model.generate_key import generate_key

app = Flask(__name__)

ACCURACY_THRESHOLD = 0.7
PHOTOS_FOLDER = 'photos'

os.makedirs(PHOTOS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = PHOTOS_FOLDER

@app.route('/')
def index():
    return render_template('a.html')

@app.route('/upload', methods=['POST'])
def upload_photo():
    if 'biometric-image' in request.files:
        #save photo
        file = request.files['biometric-image']
        file.filename = "photo_" + str(uuid.uuid4().hex) + ".jpg"
        #check if file with that name exists and if yes, change it
        while os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], file.filename)):
            file.filename = str(uuid.uuid4().hex) + ".jpg"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        #get photo embedding
        embedding = get_image_embedding(file_path)
        #retrieve all users from the db
        biometric_templates = get_all_biometric_templates()
        #look for users with biometric template similar to photo embedding
        users = []
        for user_id, biometric_template in biometric_templates.items():
            cosine_similarity = F.cosine_similarity(biometric_template, embedding)
            if cosine_similarity >= ACCURACY_THRESHOLD:
                users.append(user_id)
        #if no users found, create new user
        if len(users) == 0:
            user_id = insert_user(embedding)
            insert_photo(user_id, file.filename, embedding)
        #if more than one user found, throw and error couse its dangerous
        elif len(users) > 1:
            user_id = None
        else:
            user_id = users[0]
            insert_photo(user_id, file.filename, embedding)
            calculate_and_update_biometric_template(user_id)

        return redirect(url_for('show_profile', user_id = user_id))

    return "No photo uploaded.", 400

@app.route('/profile')
def show_profile():
    user_id = request.args.get('user_id')
    if user_id:
        keys = get_keys_by_user_id(user_id)
        return render_template('profile.html', keys=keys, user_id=user_id)
    else:
        return "Couldn't identify user"


@app.route('/generate_new_key', methods=['POST'])
def generate_new_key():
    data = request.get_json()
    user_id = data.get('user_id')
    biometric_template = get_user_biometric_template(user_id)
    new_key = generate_key(biometric_template)
    return jsonify({"key": new_key})

if __name__ == '__main__':
    app.run(debug=True)
