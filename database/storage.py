import pickle
import os
import numpy as np

class FaceRepository:
    def __init__(self, db_path="data/face_db.pkl"):
        self.db_path = db_path
        self.ensure_directory()
        self.database = self.load_data()

    def ensure_directory(self):
        """Tạo folder data nếu chưa có"""
        folder = os.path.dirname(self.db_path)
        if not os.path.exists(folder):
            os.makedirs(folder)

    def load_data(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                    print(f"📂 Đã tải {len(data)} người dùng từ Database.")
                    return data
            except Exception as e:
                print(f"⚠️ Lỗi đọc file DB: {e}")
                return {}
        return {}

    def save_user(self, name, embedding):
        """Lưu hoặc cập nhật vector của một user"""
        self.database[name] = embedding
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.database, f)
        print(f"💾 Đã lưu dữ liệu: {name}")

    def find_closest_match(self, target_embedding, threshold=0.5):
        """
        Tìm người giống nhất trong database
        Input: Vector khuôn mặt (512 chiều)
        Output: (Tên, Điểm số)
        """
        max_score = 0
        identity = "Unknown"

        if len(self.database) == 0:
            return identity, max_score

        # So khớp vector (Cosine Similarity)
        # Lưu ý: Các vector trong DB đã được chuẩn hoá (Length=1) lúc import
        # target_embedding cũng đã được chuẩn hoá bên ngoài
        for name, db_emb in self.database.items():
            score = np.dot(target_embedding, db_emb)
            if score > max_score:
                max_score = score
                identity = name
        
        if max_score > threshold:
            return identity, max_score
        else:
            return "Unknown", max_score