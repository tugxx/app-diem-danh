import cv2
import time
import numpy as np
import warnings

from services.verifier import MultiFlashVerifier, check_image_quality
from services.anti_spoof_lite import AntiSpoofSystem



# # 1. Tắt Future Warning của InsightFace
# warnings.filterwarnings("ignore", category=FutureWarning)

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
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -4.0)
    print("📷 Camera settings applied.")

    # --- KHỞI TẠO ANTI-SPOOFING AI ---
    # Load model 1 lần duy nhất ở đây
    try:
        spoof_checker = AntiSpoofSystem(model_path="weights/2.7_80x80_MiniFASNetV2.pth")
    except Exception as e:
        print(f"❌ Lỗi load Anti-Spoof: {e}")
        return
    
    # State Variables (Biến trạng thái)
    frame_count = 0
    match_streak = 0       # Đếm số lần đúng liên tiếp
    current_candidate = None
    real_counter = 0 # Đếm số lần là người thật liên tiếp
    
    # Cache (Thêm liveness_score vào đây để vẽ UI)
    cache = {
        "bbox": None, 
        "name": None, 
        "score": 0, 
        "liveness_score": 0.0 # <--- MỚI
    }
    
    # Logic thành công & Spam
    success_mode = {"active": False, "name": "", "start_time": 0}
    checkin_history = {}
    prev_fps_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        display_img = frame.copy()
        curr_time = time.time()

        # -----------------------------------------------------------
        # PHASE 1: XỬ LÝ AI - InsightFace (Chỉ chạy khi không hiện Success & đúng nhịp Frame)
        # -----------------------------------------------------------
        should_run_ai = (not success_mode["active"]) and (frame_count % CONFIG["FRAME_SKIP"] == 0)

        if should_run_ai:
            # Resize để tăng tốc
            img_small = cv2.resize(frame, (0,0), fx=CONFIG["PROCESS_SCALE"], fy=CONFIG["PROCESS_SCALE"])
            faces = engine.extract_faces(img_small) 
            
            if len(faces) > 0:
                # Tìm mặt to nhất
                main_face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
                
                # Tính toán lại Bbox trên ảnh gốc (x1, y1, x2, y2)
                bbox_orig = (main_face.bbox / CONFIG["PROCESS_SCALE"]).astype(int)
                cache["bbox"] = bbox_orig
                
                # Nhận diện danh tính
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
                        # Nếu tên người vừa nhận diện (name) GIỐNG người đang theo dõi (current_candidate)
                        if name == current_candidate:
                            match_streak += 1
                        else: # Nếu đổi người khác (hoặc AI nhận nhầm ra người khác)
                            current_candidate = name
                            match_streak = 1
                            real_counter = 0 # Reset bộ đếm thật/giả khi đổi người
                else:
                    match_streak = 0
            else:
                cache["bbox"] = None
                match_streak = 0
                current_candidate = None
                real_counter = 0

        # -----------------------------------------------------------
        # PHASE 2: KIỂM TRA LIVENESS (AI DEEP LEARNING - MiniFasnet)
        # -----------------------------------------------------------
        
        spoof_color = (255, 255, 0) # Màu vàng (đang chờ)

        if current_candidate and match_streak >= CONFIG["REQUIRED_CONSECUTIVE"] and not success_mode["active"]:
            
            try:
                # --- GỌI AI ANTI-SPOOF ---
                # Hàm này trả về ngay lập tức: Score thật, True/False
                real_score, is_real = spoof_checker.predict(frame, cache["bbox"])
                
                # Lưu vào cache để vẽ UI
                cache["liveness_score"] = real_score

                if is_real:
                    real_counter += 1
                    spoof_color = (0, 255, 0)
                    print(f"⌛ Verifying... {real_counter}/3 (Score: {real_score:.2f})")

                    if real_counter >= 3:
                        # --- ✅ NGƯỜI THẬT ---
                        print(f"✅ PASSED: {current_candidate} (Real Score: {real_score:.2f})")
                        
                        # Ghi Log Attendance
                        repository.log_attendance(current_candidate, cache["score"])
                        
                        # Kích hoạt màn hình xanh
                        success_mode.update({"active": True, "name": current_candidate, "start_time": time.time()})
                        checkin_history[current_candidate] = time.time()
                        
                        # Reset
                        match_streak = 0
                        current_candidate = None
                        real_counter = 0
                else:
                    # --- ❌ GIẢ MẠO ---
                    real_counter = 0 # Reset ngay lập tức
                    spoof_color = (0, 0, 255) # Đỏ
                    print(f"⚠️ SPOOF BLOCKED: {real_score:.2f}")
                    
                    # Reset streak để bắt user thử lại
                    match_streak = 0
                    
                    # (Tùy chọn) Thêm 1 dòng ngủ ngắn để giảm tải CPU khi bị spam fake
                    time.sleep(0.5)

            except Exception as e:
                # Đôi khi mặt ở sát mép ảnh quá sẽ gây lỗi Crop -> Bỏ qua frame này
                print(f"⚠️ Liveness Check Error: {e}")

        # -----------------------------------------------------------
        # PHASE 3: VẼ GIAO DIỆN (UI RENDERING)
        # -----------------------------------------------------------
        
        # Layer 1: Vẽ khung theo dõi (Tracking Box)
        if cache["bbox"] is not None and not success_mode["active"]:
            bbox = cache["bbox"]
            color = spoof_color if match_streak >= CONFIG["REQUIRED_CONSECUTIVE"] else (0, 255, 255)

            cv2.rectangle(display_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # Tên & Score nhận diện
            if cache["name"]:
                label = f"{cache['name']} ({cache['score']:.2f})"
                cv2.putText(display_img, label, (bbox[0], bbox[1]-15), CONFIG["FONT"], 0.7, color, 2)

            # Score Liveness (Hiển thị góc dưới)
            if match_streak > 1:
                live_txt = f"Real: {cache['liveness_score']:.2f}"
                cv2.putText(display_img, live_txt, (bbox[0], bbox[3] + 25), CONFIG["FONT"], 0.6, color, 2)
                
        # Layer 2: Vẽ màn hình Thành công (Nếu đang active)
        if success_mode["active"]:
            # is_still_active = draw_success_overlay(display_img, success_mode["name"], success_mode["start_time"])
            # success_mode["active"] = is_still_active # Cập nhật trạng thái (Hết giờ thì False)

            elapsed = time.time() - success_mode["start_time"]
            if elapsed < 2.0: # Hiện trong 2 giây
                overlay = np.full(display_img.shape, (0, 200, 0), dtype=np.uint8)
                display_img = cv2.addWeighted(display_img, 0.8, overlay, 0.2, 0)
                cv2.putText(display_img, f"XIN CHAO: {success_mode['name']}", (50, 200), 
                            CONFIG["FONT"], 1.5, (255, 255, 255), 3)
            else:
                success_mode["active"] = False

        # Layer 3: FPS
        fps = 1 / (curr_time - prev_fps_time) if (curr_time - prev_fps_time) > 0 else 0
        prev_fps_time = curr_time
        cv2.putText(display_img, f"FPS: {int(fps)}", (10, 30), CONFIG["FONT"], 0.7, (0, 255, 0), 2)

        cv2.imshow("Kiosk Face ID", display_img)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Hệ thống tắt.")
            break

    cap.release()
    cv2.destroyAllWindows()