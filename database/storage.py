import psycopg2
import os
import numpy as np

class FaceRepository:
    def __init__(self, db_config):
        """
        db_config: Dict chứa thông tin kết nối (host, user, pass, dbname)
        """
        print("⏳ Đang kết nối PostgreSQL...")
        try:
            self.conn = psycopg2.connect(**db_config)
            self.cursor = self.conn.cursor()
            print("✅ Kết nối DB thành công!")
        except Exception as e:
            print(f"❌ Lỗi kết nối DB: {e}")
            exit()
        
        # Cache dữ liệu lên RAM để so sánh cho nhanh (Real-time)
        # Cấu trúc cache: {"TenUser": numpy_array_vector}
        self.cache = self.load_data_to_ram()

    def load_data_to_ram(self):
        """Load toàn bộ vector từ DB lên RAM khi khởi động"""
        print("📥 Đang tải dữ liệu từ DB lên RAM...")
        self.cursor.execute("SELECT user_code, embedding FROM user_faces")
        rows = self.cursor.fetchall()
        
        data = {}
        for row in rows:
            user_code = row[0]
            emb_list = row[1] # Postgres trả về list
            
            # Convert list thành numpy array để tính toán
            data[user_code] = np.array(emb_list, dtype=np.float32)
            
        print(f"📂 Đã tải {len(data)} người dùng vào bộ nhớ đệm.")
        return data
    
    def save_user(self, user_code, embedding):
        """Lưu người dùng mới vào DB và cập nhật Cache"""
        # 1. Chuyển numpy array thành list python để lưu vào Postgres
        emb_list = embedding.tolist()
        
        try:
            # Upsert: Nếu user_code đã có thì cập nhật, chưa có thì thêm mới
            query = """
                INSERT INTO user_faces (user_code, full_name, embedding)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_code) 
                DO UPDATE SET embedding = EXCLUDED.embedding, created_at = CURRENT_TIMESTAMP;
            """
            self.cursor.execute(query, (user_code, user_code, emb_list))
            self.conn.commit()
            
            # 2. Cập nhật lại Cache trên RAM
            self.cache[user_code] = embedding
            print(f"💾 Đã lưu {user_code} vào PostgreSQL.")
            
        except Exception as e:
            self.conn.rollback() # Rollback nếu lỗi
            print(f"❌ Lỗi lưu DB: {e}")

    def find_closest_match(self, target_embedding, threshold=0.5):
        """
        Tìm kiếm trên RAM (Tốc độ cực nhanh)
        """
        max_score = 0
        identity = "Unknown"

        if len(self.cache) == 0:
            return identity, max_score

        # So khớp vector
        for name, db_emb in self.cache.items():
            score = np.dot(target_embedding, db_emb)
            if score > max_score:
                max_score = score
                identity = name
        
        if max_score > threshold:
            return identity, max_score
        else:
            return "Unknown", max_score
            
    def log_attendance(self, user_code, score):
        """Ghi log điểm danh vào bảng attendance_logs"""
        try:
            query = "INSERT INTO attendance_logs (user_code, score) VALUES (%s, %s)"
            self.cursor.execute(query, (user_code, float(score)))
            self.conn.commit()
            print(f"📝 Đã ghi log điểm danh cho {user_code}")
        except Exception as e:
            print(f"⚠️ Lỗi ghi log: {e}")

    # def ensure_directory(self):
    #     """Tạo folder data nếu chưa có"""
    #     folder = os.path.dirname(self.db_path)
    #     if not os.path.exists(folder):
    #         os.makedirs(folder)

    # def load_data(self):
    #     if os.path.exists(self.db_path):
    #         try:
    #             with open(self.db_path, 'rb') as f:
    #                 data = pickle.load(f)
    #                 print(f"📂 Đã tải {len(data)} người dùng từ Database.")
    #                 return data
    #         except Exception as e:
    #             print(f"⚠️ Lỗi đọc file DB: {e}")
    #             return {}
    #     return {}

    # def save_user(self, name, embedding):
    #     """Lưu hoặc cập nhật vector của một user"""
    #     self.database[name] = embedding
    #     with open(self.db_path, 'wb') as f:
    #         pickle.dump(self.database, f)
    #     print(f"💾 Đã lưu dữ liệu: {name}")

    # def find_closest_match(self, target_embedding, threshold=0.5):
    #     """
    #     Tìm người giống nhất trong database
    #     Input: Vector khuôn mặt (512 chiều)
    #     Output: (Tên, Điểm số)
    #     """
    #     max_score = 0
    #     identity = "Unknown"

    #     if len(self.database) == 0:
    #         return identity, max_score

    #     # So khớp vector (Cosine Similarity)
    #     # Lưu ý: Các vector trong DB đã được chuẩn hoá (Length=1) lúc import
    #     # target_embedding cũng đã được chuẩn hoá bên ngoài
    #     for name, db_emb in self.database.items():
    #         score = np.dot(target_embedding, db_emb)
    #         if score > max_score:
    #             max_score = score
    #             identity = name
        
    #     if max_score > threshold:
    #         return identity, max_score
    #     else:
    #         return "Unknown", max_score