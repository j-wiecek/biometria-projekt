import sqlite3 
import torch
import io
import numpy as np

conn = sqlite3.connect('biometric_data.db', check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS user_db (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    biometric_template BLOB
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS photo_db (
    photo_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    photo_name TEXT,
    embedding BLOB,
    FOREIGN KEY (user_id) REFERENCES user_db(user_id)
)
''')

conn.commit()

def insert_photo(user_id, photo_name, embedding_tensor):
    buffer = io.BytesIO()
    torch.save(embedding_tensor, buffer)
    buffer.seek(0)
    serialized_embedding = buffer.read()

    cursor.execute('''
    INSERT INTO photo_db (user_id, photo_name, embedding) 
    VALUES (?, ?, ?)
    ''', (user_id, photo_name, serialized_embedding))
    conn.commit()
    return cursor.lastrowid

def insert_user(biometric_template_tensor):
    buffer = io.BytesIO()
    torch.save(biometric_template_tensor, buffer)
    buffer.seek(0)
    serialized_template = buffer.read()

    cursor.execute('INSERT INTO user_db (biometric_template) VALUES (?)', (serialized_template,))
    conn.commit()
    return cursor.lastrowid

def get_photo_embedding(photo_id):
    cursor.execute('SELECT embedding FROM photo_db WHERE photo_id = ?', (photo_id,))
    result = cursor.fetchone()
    
    if result:
        serialized_embedding = result[0]
        buffer = io.BytesIO(serialized_embedding)
        embedding = torch.load(buffer)
        return embedding
    return None

def get_user_biometric_template(user_id):
    cursor.execute('SELECT biometric_template FROM user_db WHERE user_id = ?', (user_id,))
    result = cursor.fetchone()
    
    if result:
        serialized_template = result[0]
        buffer = io.BytesIO(serialized_template)
        biometric_template = torch.load(buffer)
        return biometric_template
    return None

def get_all_biometric_templates():
    cursor.execute('SELECT user_id, biometric_template FROM user_db')
    results = cursor.fetchall()

    biometric_templates_dict = {}

    for result in results:
        user_id = result[0]
        serialized_template = result[1]
        
        buffer = io.BytesIO(serialized_template)
        biometric_template = torch.load(buffer)
        
        biometric_templates_dict[user_id] = biometric_template

    return biometric_templates_dict

def calculate_and_update_biometric_template(user_id):

    cursor.execute('SELECT embedding FROM photo_db WHERE user_id = ?', (user_id,))
    photo_embeddings = cursor.fetchall()
    embeddings_array = np.array([torch.load(io.BytesIO(embedding[0])).detach().numpy() for embedding in photo_embeddings])
    average_embedding = np.mean(embeddings_array, axis=0)
    buffer = io.BytesIO()
    torch.save(average_embedding, buffer)
    buffer.seek(0)
    serialized_embedding = buffer.read()
    cursor.execute('UPDATE user_db SET biometric_template = ? WHERE user_id = ?', (serialized_embedding, user_id))
    
    conn.commit()