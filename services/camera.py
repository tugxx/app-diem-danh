import cv2
import time
import numpy as np



CONFIG = {
    "FRAME_SKIP": 5,             # Chạy AI mỗi 5 frame
    "PROCESS_SCALE": 0.5,        # Thu nhỏ ảnh 50%
    "SIMILARITY_THRESH": 0.5,    # Ngưỡng nhận diện
    "REQUIRED_CONSECUTIVE": 3,   # Số lần đúng liên tiếp
    "SUCCESS_DURATION": 3.0,     # Thời gian hiện thông báo thành công
    "SPAM_DURATION": 60.0,       # Thời gian cấm check-in lại (giây)
    "FONT": cv2.FONT_HERSHEY_SIMPLEX
}


def run_camera(engine, repository):
    print("\n🚀 Đang khởi động Camera (Real-time Optimized)... Nhấn 'q' để thoát.")
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # --- CẤU HÌNH TỐI ƯU (GIỐNG AUTO CHECK-IN) ---
    FRAME_SKIP = 5        # Chỉ chạy AI mỗi 5 frame
    PROCESS_SCALE = 0.5   # Thu nhỏ ảnh 50% để AI chạy nhanh
    SIMILARITY_THRESHOLD = 0.5 # Ngưỡng nhận diện cho buffalo_s
    
    # Biến lưu trữ kết quả tạm thời (Cache) để vẽ khi AI đang nghỉ
    # Cấu trúc: list các dict {'bbox': ..., 'name': ..., 'score': ..., 'kps': ...}
    cached_results = []
    
    frame_count = 0
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1) # Gương
        display_img = frame.copy()
        h, w = frame.shape[:2]

        # =========================================================
        # 1. LOGIC AI (CHỈ CHẠY ĐỊNH KỲ - MỖI 5 FRAME)
        # =========================================================
        if frame_count % FRAME_SKIP == 0:
            cached_results = [] # Reset cache
            
            # Resize ảnh nhỏ để AI chạy nhanh
            img_small = cv2.resize(frame, (0,0), fx=PROCESS_SCALE, fy=PROCESS_SCALE)
            
            # Gọi Core AI
            faces = engine.extract_faces(img_small)
            
            for face in faces:
                # Quy đổi toạ độ từ ảnh nhỏ về ảnh gốc
                bbox = (face.bbox / PROCESS_SCALE).astype(int)
                kps = (face.kps / PROCESS_SCALE).astype(int) if face.kps is not None else None
                
                # --- TÌM KIẾM TRONG DB ---
                current_emb = face.embedding
                current_emb = current_emb / np.linalg.norm(current_emb)
                
                name, score = repository.find_closest_match(current_emb, threshold=SIMILARITY_THRESHOLD)
                
                # Lưu vào cache để dùng cho các frame sau
                cached_results.append({
                    "bbox": bbox,
                    "name": name,
                    "score": score,
                    "kps": kps
                })

        # =========================================================
        # 2. LOGIC VẼ UI (CHẠY LIÊN TỤC MỖI FRAME -> MƯỢT)
        # =========================================================
        for res in cached_results:
            bbox = res["bbox"]
            name = res["name"]
            score = res["score"]
            kps = res["kps"]
            
            # Chọn màu
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            label = f"{name} ({score:.2f})"

            # Vẽ khung
            cv2.rectangle(display_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Vẽ Landmarks (Mắt mũi miệng)
            if kps is not None:
                for kp in kps:
                    cv2.circle(display_img, (kp[0], kp[1]), 3, (255, 255, 0), -1)

            # Vẽ Header tên
            cv2.rectangle(display_img, (bbox[0], bbox[1]-30), (bbox[2], bbox[1]), color, -1)
            cv2.putText(display_img, label, (bbox[0]+5, bbox[1]-5), font, 0.6, (255, 255, 255), 2)

            # Vẽ Crop View (Góc trái) - Chỉ cần vẽ lại từ bbox đã cache
            try:
                x1, y1 = max(0, bbox[0]-20), max(0, bbox[1]-20)
                x2, y2 = min(w, bbox[2]+20), min(h, bbox[3]+20)
                
                if x2 > x1 and y2 > y1: # Kiểm tra toạ độ hợp lệ
                    face_crop = cv2.resize(frame[y1:y2, x1:x2], (150, 150))
                    
                    # Vẽ viền & ảnh
                    cv2.rectangle(display_img, (10, 10), (160, 160), (255, 255, 255), 2)
                    display_img[10:160, 10:160] = face_crop
                    
                    # Chữ AI Input
                    cv2.putText(display_img, "AI View", (10, 175), font, 0.5, (0, 255, 0), 1)
            except: 
                pass

        # FPS Counter (Màu xanh lá to rõ)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(display_img, f"FPS: {int(fps)}", (w-120, 40), font, 1, (0, 255, 0), 2)

        cv2.imshow("Face Attendance Pro", display_img)
        
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


# =========================================================
# CÁC HÀM HỖ TRỢ (HELPER FUNCTIONS)
# =========================================================

def draw_success_overlay(img, name, start_time):
    """Vẽ màn hình thông báo thành công (Hiệu ứng kính mờ)"""
    h, w = img.shape[:2]
    elapsed = time.time() - start_time
    remaining = int(CONFIG["SUCCESS_DURATION"] - elapsed) + 1
    
    if elapsed > CONFIG["SUCCESS_DURATION"]:
        return False # Hết giờ hiển thị

    # 1. Làm mờ nền
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h//2 - 60), (w, h//2 + 60), (0, 200, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    # 2. Hiện tên
    msg = f"XIN CHAO: {name}"
    text_size = cv2.getTextSize(msg, CONFIG["FONT"], 1.5, 3)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(img, msg, (text_x, h//2 + 10), CONFIG["FONT"], 1.5, (255, 255, 255), 3)

    # 3. Đồng hồ đếm ngược
    cv2.circle(img, (w-50, 50), 30, (0, 255, 0), -1)
    cv2.putText(img, str(remaining), (w-60, 60), CONFIG["FONT"], 1, (255, 255, 255), 2)
    
    return True # Vẫn đang hiển thị


def is_spamming(history, name):
    """Kiểm tra xem người này có vừa check-in xong không"""
    last_time = history.get(name, 0)
    return (time.time() - last_time) < CONFIG["SPAM_DURATION"]


def run_auto_checkin(engine, repository):
    print("\n🤖 CHẾ ĐỘ KIOSK")
    print("👉 Hệ thống chạy liên tục. Nhấn 'q' để thoát.")
    
    cap = cv2.VideoCapture(0)
    
    # State Variables (Biến trạng thái)
    frame_count = 0
    match_streak = 0       # Đếm số lần đúng liên tiếp
    current_candidate = None
    
    # Cache (Lưu kết quả AI để vẽ mượt)
    cache = {"bbox": None, "name": None, "score": 0}
    
    # Logic thành công & Spam
    success_mode = {"active": False, "name": "", "start_time": 0}
    checkin_history = {}
    
    prev_fps_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        display_img = frame.copy()
        curr_time = time.time()

        # -----------------------------------------------------------
        # PHASE 1: XỬ LÝ AI (Chỉ chạy khi không hiện Success & đúng nhịp Frame)
        # -----------------------------------------------------------
        should_run_ai = (not success_mode["active"]) and (frame_count % CONFIG["FRAME_SKIP"] == 0)

        if should_run_ai:
            # Resize để tăng tốc
            img_small = cv2.resize(frame, (0,0), fx=CONFIG["PROCESS_SCALE"], fy=CONFIG["PROCESS_SCALE"])
            faces = engine.extract_faces(img_small)
            
            if len(faces) > 0:
                # Tìm mặt to nhất
                main_face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
                
                # Tính toán lại Bbox trên ảnh gốc
                bbox_orig = (main_face.bbox / CONFIG["PROCESS_SCALE"]).astype(int)
                cache["bbox"] = bbox_orig
                
                # Nhận diện
                emb = main_face.embedding / np.linalg.norm(main_face.embedding)
                name, score = repository.find_closest_match(emb, threshold=CONFIG["SIMILARITY_THRESH"])
                
                cache["name"] = name
                cache["score"] = score

                # Logic Ổn định (3 lần liên tiếp)
                if name != "Unknown":
                    # Kiểm tra Spam
                    if is_spamming(checkin_history, name):
                        cache["name"] = f"{name} (Wait...)"
                        match_streak = 0
                    else:
                        # Logic Streak
                        if name == current_candidate:
                            match_streak += 1
                        else:
                            current_candidate = name
                            match_streak = 1
                else:
                    match_streak = 0
            else:
                cache["bbox"] = None
                match_streak = 0

        # -----------------------------------------------------------
        # PHASE 2: KIỂM TRA ĐIỀU KIỆN CHỐT ĐƠN (TRIGGER SUCCESS)
        # -----------------------------------------------------------
        if match_streak >= CONFIG["REQUIRED_CONSECUTIVE"] and not success_mode["active"]:
            user_name = current_candidate
            
            # Ghi log vào bảng attendance_logs trong Postgres
            current_score = cache["score"] 
            repository.log_attendance(user_name, current_score)

            # Action: Ghi log & Kích hoạt UI thành công
            print(f"✅ [LOG] Check-in: {user_name} at {time.strftime('%H:%M:%S')}")
            checkin_history[user_name] = curr_time
            
            success_mode.update({"active": True, "name": user_name, "start_time": curr_time})
            
            # Reset
            match_streak = 0
            current_candidate = None

        # -----------------------------------------------------------
        # PHASE 3: VẼ GIAO DIỆN (UI RENDERING)
        # -----------------------------------------------------------
        
        # Layer 1: Vẽ khung theo dõi (Tracking Box)
        if cache["bbox"] is not None and not success_mode["active"]:
            bbox = cache["bbox"]
            color = (0, 255, 0) if match_streak > 0 else (0, 255, 255) # Xanh hoặc Vàng
            
            cv2.rectangle(display_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Loading Bar (Visual feedback)
            if match_streak > 0:
                bar_len = int((bbox[2]-bbox[0]) * (match_streak / CONFIG["REQUIRED_CONSECUTIVE"]))
                cv2.rectangle(display_img, (bbox[0], bbox[1]-10), (bbox[0]+bar_len, bbox[1]), (0, 255, 0), -1)
            
            # Tên tạm thời
            if cache["name"]:
                label = f"{cache['name']} ({cache['score']:.2f})"
                cv2.putText(display_img, label, (bbox[0], bbox[1]-15), CONFIG["FONT"], 0.7, color, 2)

        # Layer 2: Vẽ màn hình Thành công (Nếu đang active)
        if success_mode["active"]:
            is_still_active = draw_success_overlay(display_img, success_mode["name"], success_mode["start_time"])
            success_mode["active"] = is_still_active # Cập nhật trạng thái (Hết giờ thì False)

        # Layer 3: FPS
        fps = 1 / (curr_time - prev_fps_time)
        prev_fps_time = curr_time
        cv2.putText(display_img, f"FPS: {int(fps)}", (10, 30), CONFIG["FONT"], 0.7, (0, 255, 0), 2)

        cv2.imshow("Kiosk Face ID", display_img)

        # -----------------------------------------------------------
        # PHASE 4: INPUT HANDLE
        # -----------------------------------------------------------
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Hệ thống tắt.")
            break

    cap.release()
    cv2.destroyAllWindows()