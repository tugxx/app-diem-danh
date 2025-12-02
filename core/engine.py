import numpy as np
from insightface.app import FaceAnalysis

class FaceEngine:
    def __init__(self):
        print("⏳ Đang khởi tạo AI Engine (InsightFace)...")
        # Sử dụng model buffalo_l chuẩn
        self.app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640)) # Có thể xuống 320
        print("✅ AI Engine đã sẵn sàng!")

    def extract_faces(self, img_bgr):
        """
        Input: Ảnh OpenCV (BGR)
        Output: List các khuôn mặt (có bounding box, embedding, keypoints)
        """
        if img_bgr is None: return []
        return self.app.get(img_bgr)