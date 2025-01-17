import sqlite3
import torch
import io

# Połączenie z bazą danych
conn = sqlite3.connect('biometric_data.db', check_same_thread=False)
cursor = conn.cursor()

# Tworzenie tabel, jeśli nie istnieją
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

cursor.execute('''
CREATE TABLE IF NOT EXISTS keys (
    key_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    key TEXT,
    FOREIGN KEY (user_id) REFERENCES user_db(user_id)
)
''')

conn.commit()

# Funkcja do zapisywania zdjęcia i embeddingu
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

# Funkcja do zapisywania użytkownika i jego template
def insert_user(biometric_template_tensor):
    buffer = io.BytesIO()
    torch.save(biometric_template_tensor, buffer)
    buffer.seek(0)
    serialized_template = buffer.read()

    cursor.execute('INSERT INTO user_db (biometric_template) VALUES (?)', (serialized_template,))
    conn.commit()
    return cursor.lastrowid

# Pobranie embeddingu zdjęcia
def get_photo_embedding(photo_id):
    cursor.execute('SELECT embedding FROM photo_db WHERE photo_id = ?', (photo_id,))
    result = cursor.fetchone()
    
    if result:
        buffer = io.BytesIO(result[0])
        return torch.load(buffer)
    return None

# Pobranie biometrycznego template użytkownika
def get_user_biometric_template(user_id):
    cursor.execute('SELECT biometric_template FROM user_db WHERE user_id = ?', (user_id,))
    result = cursor.fetchone()
    
    if result:
        buffer = io.BytesIO(result[0])
        return torch.load(buffer)
    return None

# Pobranie wszystkich biometrycznych template'ów
def get_all_biometric_templates():
    cursor.execute('SELECT user_id, biometric_template FROM user_db')
    results = cursor.fetchall()

    biometric_templates_dict = {}
    for result in results:
        user_id, serialized_template = result
        buffer = io.BytesIO(serialized_template)
        biometric_templates_dict[user_id] = torch.load(buffer)

    return biometric_templates_dict

# Poprawiona funkcja aktualizująca biometryczny template użytkownika
def calculate_and_update_biometric_template(user_id):
    cursor.execute('SELECT embedding FROM photo_db WHERE user_id = ?', (user_id,))
    photo_embeddings = cursor.fetchall()

    if not photo_embeddings:
        print(f"⚠ Brak embeddingów dla użytkownika {user_id}")
        return

    # Przetwarzanie embeddingów i wyrównanie kształtu
    embeddings_list = []
    for embedding in photo_embeddings:
        tensor = torch.load(io.BytesIO(embedding[0])).squeeze().flatten()
        if tensor.shape != (768,):  # Sprawdzamy, czy embedding ma prawidłowy wymiar
            print(f"⚠ Błąd: embedding użytkownika {user_id} ma zły kształt {tensor.shape}")
            continue
        embeddings_list.append(tensor)

    if not embeddings_list:
        print(f"⚠ Wszystkie embeddingi dla użytkownika {user_id} miały niepoprawny kształt!")
        return

    # Obliczenie średniego embeddingu
    average_embedding = torch.stack(embeddings_list).mean(dim=0)

    buffer = io.BytesIO()
    torch.save(average_embedding, buffer)
    buffer.seek(0)
    serialized_embedding = buffer.read()

    # Aktualizacja w bazie
    cursor.execute('UPDATE user_db SET biometric_template = ? WHERE user_id = ?', (serialized_embedding, user_id))
    conn.commit()

# Funkcja do zapisu klucza biometrycznego
def insert_key(user_id, key):
    cursor.execute('''
    INSERT INTO keys (user_id, key)
    VALUES (?, ?)
    ''', (user_id, key))
    conn.commit()
    return cursor.lastrowid

# Pobranie kluczy użytkownika
def get_keys_by_user_id(user_id):
    cursor.execute('''
    SELECT key FROM keys WHERE user_id = ?
    ''', (user_id,))
    return [row[0] for row in cursor.fetchall()]
