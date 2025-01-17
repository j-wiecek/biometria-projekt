import hashlib
import numpy as np
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.hashes import SHA256

def generate_key(embedding, sid):
    try:
        # Hashowanie ID sesji
        sid_hash = hashlib.sha256(sid.encode()).digest()

        # Konwersja tensorów PyTorch do NumPy
        embedding = embedding.cpu().detach().numpy() 

        # Normalizacja embeddingu
        embedding_min = np.min(embedding)
        embedding_max = np.max(embedding)
        embedding_normalized = (embedding - embedding_min) / (embedding_max - embedding_min)

        # Konwersja do bajtów (z poprawką na NumPy)
        embedding_bytes = bytes(int(float(val) * 255) for val in embedding_normalized.flatten().tolist())

        # XOR dla zabezpieczenia klucza
        xor_result = bytes(
            a ^ b for a, b in zip(embedding_bytes, sid_hash * (len(embedding_bytes) // len(sid_hash) + 1))
        )

        # Derivation key używając HKDF
        hkdf = HKDF(algorithm=SHA256(), length=32, salt=None, info=b'biometric-key')
        key = hkdf.derive(xor_result)
        return key

    except Exception as e:
        print(f"Error generating key: {e}")
        raise
