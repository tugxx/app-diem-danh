import cv2
import numpy as np
import time
import random



def check_image_quality(frame, face_bbox):
    """
    Kiểm tra xem ảnh có phải là ảnh chụp lại từ màn hình (Screen Replay) 
    hoặc ảnh in mờ hay không.
    """
    x1, y1, x2, y2 = face_bbox
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return False, "No Face"

    # 1. Kiểm tra độ mờ (Blur Detection) - Chống ảnh in chất lượng thấp
    # Dùng Laplacian Variance. Ảnh thật thường sắc nét ở các chi tiết như lông mày, mắt.
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Ngưỡng này cần tinh chỉnh tùy webcam. 
    # Webcam HD thường > 100. Ảnh chụp lại từ đt khác thường < 50 hoặc rất cao do noise.
    if blur_score < 80: 
        return False, f"Too Blurry ({int(blur_score)})"

    # 2. Kiểm tra nhiễu hạt (Noise/Moire Pattern) - Chống chụp lại màn hình
    # Ảnh chụp lại màn hình thường có nhiễu cao tần (high frequency noise) do lưới pixel.
    # Ta dùng biến đổi Fourier hoặc đơn giản hơn là kiểm tra độ chênh lệch màu cục bộ.
    
    # Chuyển sang không gian màu HSV để tách độ sáng (V)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Tính độ lệch chuẩn của kênh sáng. 
    # Màn hình LCD thường có độ sáng rất đều hoặc nhiễu hạt rất gắt.
    std_dev_v = np.std(v)
    
    # Nếu ánh sáng quá phẳng (như ảnh 2D được chiếu sáng đều) -> Nghi ngờ
    if std_dev_v < 15: 
        return False, "Image too Flat (2D Photo?)"

    return True, "OK"


class MultiFlashVerifier:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = "IDLE" 
        self.sequence = [] # Chứa danh sách màu cần flash
        self.current_step = 0
        self.start_time = 0
        self.base_mean = None
        self.flash_mean = None
        self.passed_steps = 0
        self.total_steps = 3 # Test 3 màu liên tiếp
        self.result = False

    def start_challenge(self):
        self.state = "PREPARING"
        self.start_time = time.time()
        self.current_step = 0
        self.passed_steps = 0
        
        # Tạo chuỗi 3 màu ngẫu nhiên (R, G, B)
        # Định dạng (B, G, R)
        pool = [
            ((0, 0, 255), "RED"),
            ((0, 255, 0), "GREEN"),
            ((255, 0, 0), "BLUE")
        ]
        random.shuffle(pool)
        self.sequence = pool[:self.total_steps] # Lấy chuỗi màu
        print(f"🚦 Bắt đầu chuỗi kiểm tra: {[x[1] for x in self.sequence]}")

    def process(self, frame, face_bbox):
        """
        Sử dụng YCrCb cho Red/Blue để chống nhiễu ánh sáng.
        Giữ RGB cho Green.
        """
        current_time = time.time()
        
        # --- 1. CROP VÙNG TRÁN ---
        x1, y1, x2, y2 = face_bbox
        w, h = x2 - x1, y2 - y1
        roi_y1 = y1 + int(h * 0.15)
        roi_y2 = y1 + int(h * 0.50)
        roi_x1 = x1 + int(w * 0.25)
        roi_x2 = x2 - int(w * 0.25)
        
        if roi_y1 >= roi_y2 or roi_x1 >= roi_x2: return None, "Face error", False
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0: return None, "No Face", False
        
        # --- TÍNH TOÁN GIÁ TRỊ MÀU ---
        # 1. Hệ RGB (Dùng cho Green)
        mean_bgr = np.mean(roi, axis=(0, 1)) # [Blue, Green, Red]
        
        # 2. Hệ YCrCb (Dùng cho Red/Blue) -> Quan trọng nhất!
        roi_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        mean_ycrcb = np.mean(roi_ycrcb, axis=(0, 1)) # [Y, Cr, Cb]
        
        # Giá trị hiện tại gói gọn
        curr_vals = {"bgr": mean_bgr, "ycrcb": mean_ycrcb}

        # --- STATE MACHINE ---
        if self.state == "PREPARING":
            if current_time - self.start_time < 0.5: return None, "Stay still...", False
            
            # Lưu cả 2 hệ màu làm base
            self.base_bgr = mean_bgr
            self.base_ycrcb = mean_ycrcb
            
            self.state = "FLASHING"
            self.start_time = current_time
            self.flash_vals = None 
            return None, "Ready...", False

        elif self.state == "FLASHING":
            target_color, color_name = self.sequence[self.current_step]
            elapsed = current_time - self.start_time

            if elapsed < 0.8:
                if elapsed > 0.1:
                    # Gom dữ liệu hiện tại
                    curr_vals = {"bgr": mean_bgr, "ycrcb": mean_ycrcb}
                    
                    if self.flash_vals is None:
                        self.flash_vals = curr_vals
                    else:
                        # Logic tìm Max (Peak) thông minh hơn
                        # Nếu là màu ĐỎ -> Tìm lúc Cr cao nhất
                        if color_name == "RED":
                            if mean_ycrcb[1] > self.flash_vals["ycrcb"][1]: # Kênh Cr
                                self.flash_vals = curr_vals
                        # Nếu là màu XANH DƯƠNG -> Tìm lúc Cb cao nhất
                        elif color_name == "BLUE":
                            if mean_ycrcb[2] > self.flash_vals["ycrcb"][2]: # Kênh Cb
                                self.flash_vals = curr_vals
                        # Nếu là XANH LÁ -> Dùng kênh Green của RGB
                        else:
                            if mean_bgr[1] > self.flash_vals["bgr"][1]:
                                self.flash_vals = curr_vals
                                
                return target_color, f"Look at screen ({color_name})", False
            
            # [FIX 2]: Hết giờ Flash -> Trước khi đi, kiểm tra lần cuối
            if self.flash_vals is None:
                # Nếu chưa bắt được gì (do FPS thấp), lấy ngay frame cuối cùng này!
                self.flash_vals = curr_vals

            self.state = "EVALUATING"
            return None, "Analyzing...", False

        elif self.state == "EVALUATING":
            if self.flash_vals is None:
                # Fallback an toàn
                self.flash_vals = {"bgr": self.base_bgr, "ycrcb": self.base_ycrcb}

            _, color_name = self.sequence[self.current_step]
            is_pass = False
            debug_info = ""

            # --- LOGIC ĐÁNH GIÁ CHUYÊN SÂU ---
            # [FIX 3]: Hạ Threshold xuống 1.0 (Webcam thường chỉ đạt tầm 1.2 - 2.0)
            THRESHOLD = 1.0

            # CASE 1: MÀU ĐỎ (Dùng Cr)
            if color_name == "RED":
                # Cr (Red-Difference) phải tăng lên
                diff = self.flash_vals["ycrcb"][1] - self.base_ycrcb[1]
                debug_info = f"Delta Cr={diff:.2f}"
                # Ngưỡng thấp hơn RGB vì YCrCb rất nhạy
                if diff > THRESHOLD: is_pass = True 

            # CASE 2: MÀU XANH DƯƠNG (Dùng Cb)
            elif color_name == "BLUE":
                # Cb (Blue-Difference) phải tăng lên
                diff = self.flash_vals["ycrcb"][2] - self.base_ycrcb[2]
                debug_info = f"Delta Cb={diff:.2f}"
                if diff > THRESHOLD: is_pass = True

            # CASE 3: MÀU XANH LÁ (Dùng Green RGB - Fallback)
            elif color_name == "GREEN":
                # Kênh Green phải tăng mạnh hơn các kênh khác
                diff_bgr = self.flash_vals["bgr"] - self.base_bgr
                val_g = diff_bgr[1]
                val_others = (diff_bgr[0] + diff_bgr[2]) / 2
                debug_info = f"Delta G={val_g:.2f} vs Others={val_others:.2f}"
                
                # Logic tương quan (như cũ)
                if val_g > THRESHOLD and val_g > val_others: is_pass = True
                elif val_g > (val_others + 1.0): is_pass = True

            print(f"DEBUG [{color_name}]: {debug_info} -> {'✅ OK' if is_pass else '❌ FAIL'}")

            if is_pass: self.passed_steps += 1
            
            self.current_step += 1
            if self.current_step < self.total_steps:
                self.state = "PREPARING"
                self.start_time = time.time()
                return None, "Next...", False
            else:
                self.state = "FINISHED"
                return None, "Done", False

        elif self.state == "FINISHED":
            print(f"📊 KẾT QUẢ: {self.passed_steps}/{self.total_steps}")
            self.result = self.passed_steps >= 2
            return None, "Success" if self.result else "Failed", True

        return None, "", False

    # def process(self, frame, face_bbox):
    #     """
    #     Phiên bản TỐI ƯU: Forehead Crop + Anti-Crash + Auto-Exposure Logic
    #     """
    #     current_time = time.time()
        
    #     # --- 1. TỐI ƯU ROI: CHỈ LẤY VÙNG TRÁN ---
    #     # Lý do: Trán là vùng da phẳng, phản chiếu ánh sáng màn hình tốt nhất 
    #     # và không bị nhiễu do chớp mắt hay cử động miệng.
    #     x1, y1, x2, y2 = face_bbox
    #     w = x2 - x1
    #     h = y2 - y1
        
    #     # Crop vùng trán (Từ 15% đến 50% chiều cao khuôn mặt)
    #     roi_y1 = y1 + int(h * 0.15)
    #     roi_y2 = y1 + int(h * 0.50)
    #     roi_x1 = x1 + int(w * 0.25) # Bỏ tóc mai 2 bên
    #     roi_x2 = x2 - int(w * 0.25)
        
    #     # Safety check: Nếu mặt quá xa hoặc crop bị lỗi
    #     if roi_y1 >= roi_y2 or roi_x1 >= roi_x2:
    #          return None, "Face too far", False

    #     roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
    #     if roi.size == 0: return None, "No Face", False
        
    #     # Tính trung bình màu (BGR)
    #     current_mean = np.mean(roi, axis=(0, 1)) 

    #     # --- STATE MACHINE ---
        
    #     # GIAI ĐOẠN 1: CHUẨN BỊ (Lấy mẫu nền - Base)
    #     if self.state == "PREPARING":
    #         # Nghỉ 0.5s để camera ổn định lại sau lần flash trước
    #         if current_time - self.start_time < 0.5: 
    #             return None, "Stay still...", False
            
    #         self.base_mean = current_mean
    #         self.state = "FLASHING"
    #         self.start_time = current_time
    #         self.flash_mean = None # Reset giá trị flash
    #         return None, "Ready...", False

    #     # GIAI ĐOẠN 2: CHIẾU SÁNG (Bật màn hình màu)
    #     elif self.state == "FLASHING":
    #         target_color, color_name = self.sequence[self.current_step]
            
    #         # Flash trong 0.8s (Đủ lâu để cam nhận nhận ánh sáng)
    #         if current_time - self.start_time < 0.8:
    #             # Bỏ qua 0.25s đầu tiên (Thời gian màn hình chuyển màu + cam thích ứng)
    #             if current_time - self.start_time > 0.25:
    #                 if self.flash_mean is None:
    #                     self.flash_mean = current_mean
    #                 else:
    #                     # Logic: Giữ lại khoảnh khắc sáng nhất (Peak brightness)
    #                     idx = np.argmax(target_color) 
    #                     if current_mean[idx] > self.flash_mean[idx]:
    #                         self.flash_mean = current_mean
                            
    #             return target_color, f"Look at screen ({color_name})", False
            
    #         # Hết giờ Flash -> Sang bước chấm điểm
    #         self.state = "EVALUATING"
    #         return None, "Analyzing...", False

    #     # GIAI ĐOẠN 3: ĐÁNH GIÁ (Tính điểm)
    #     elif self.state == "EVALUATING":
    #         # [FIX LỖI CRASH]: Nếu máy lag quá không kịp lấy mẫu Flash
    #         if self.flash_mean is None:
    #             print("⚠️ Missed flash window. Using base as backup.")
    #             self.flash_mean = self.base_mean

    #         target_color, color_name = self.sequence[self.current_step]
    #         flash_idx = np.argmax(target_color) # Index kênh màu chính (0=B, 1=G, 2=R)

    #         # Tính độ chênh lệch: Flash - Base
    #         diff = self.flash_mean - self.base_mean
            
    #         val_main = diff[flash_idx] # Giá trị thay đổi của kênh màu Flash
            
    #         # Tính nhiễu (Trung bình thay đổi của 2 kênh còn lại)
    #         others = list(diff)
    #         others.pop(flash_idx)
    #         val_noise = np.mean(others)
            
    #         # DEBUG LOG
    #         print(f"DEBUG [{color_name}]: Main={val_main:.2f} | Noise={val_noise:.2f}", end="")

    #         # --- LOGIC QUYẾT ĐỊNH (CORE LOGIC) ---
    #         is_pass = False
            
    #         # Trường hợp 1: Tăng trưởng Tuyệt đối (Lý tưởng)
    #         # Ánh sáng phản chiếu làm mặt sáng lên > 1.5 đơn vị
    #         if val_main > 1.5 and val_main > val_noise:
    #             is_pass = True
                
    #         # Trường hợp 2: Tương quan (Auto Exposure kick-in)
    #         # Nếu camera tự làm tối đi (Main < 0), nhưng kênh màu Flash bị tối đi ÍT HƠN các kênh khác
    #         # Ví dụ: Main giảm -2, nhưng Noise giảm -5 -> Nghĩa là có ánh sáng màu đó bù vào.
    #         elif val_main > (val_noise + 1.0): 
    #             # (Main vẫn lớn hơn Noise ít nhất 1 đơn vị dù cả 2 đều âm)
    #             is_pass = True
            
    #         # Trường hợp 3: Bonus Pass (Thay đổi quá rõ rệt)
    #         if val_main > 6.0: is_pass = True

    #         if is_pass:
    #             print(" -> ✅ OK")
    #             self.passed_steps += 1
    #         else:
    #             print(" -> ❌ FAIL")
                
    #         # Chuyển bước
    #         self.current_step += 1
    #         if self.current_step < self.total_steps:
    #             self.state = "PREPARING"
    #             self.start_time = time.time()
    #             return None, "Next color...", False
    #         else:
    #             self.state = "FINISHED"
    #             return None, "Done", False

    #     # GIAI ĐOẠN 4: KẾT THÚC
    #     elif self.state == "FINISHED":
    #         print(f"📊 KẾT QUẢ: {self.passed_steps}/{self.total_steps}")
            
    #         # Pass nếu đúng ít nhất 2/3 bước
    #         self.result = self.passed_steps >= 2 
    #         return None, "Success" if self.result else "Failed", True

    #     return None, "", False

    # def process(self, frame, face_bbox):
    #     current_time = time.time()
        
    #     # --- 1. CẢI TIẾN CROP VÙNG TRÁN (Forehead) ---
    #     # Vùng trán phản chiếu ánh sáng tốt hơn và ít bị nhiễu bởi mắt/miệng
    #     x1, y1, x2, y2 = face_bbox
    #     w = x2 - x1
    #     h = y2 - y1
        
    #     # Lấy vùng trán: Từ 15% đến 50% chiều cao khuôn mặt (tính từ trên xuống)
    #     roi_y1 = y1 + int(h * 0.15)
    #     roi_y2 = y1 + int(h * 0.50)
    #     roi_x1 = x1 + int(w * 0.25) # Bỏ bớt tóc 2 bên
    #     roi_x2 = x2 - int(w * 0.25)
        
    #     # Kiểm tra bounds
    #     if roi_y1 >= roi_y2 or roi_x1 >= roi_x2:
    #          return None, "Face too far/small", False

    #     roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
    #     if roi.size == 0: return None, "No Face", False
        
    #     # Tính mean theo chuẩn BGR của OpenCV
    #     current_mean = np.mean(roi, axis=(0, 1)) 

    #     # --- STATE MACHINE ---
    #     if self.state == "PREPARING":
    #         if current_time - self.start_time < 0.5: # Tăng thời gian nghỉ lên 0.5s để cam ổn định
    #             return None, "Stay still...", False
            
    #         self.base_mean = current_mean
    #         # DEBUG: In ra base để xem cam có bị tối quá không
    #         # print(f"DEBUG: Base BGR={self.base_mean.astype(int)}") 
            
    #         self.state = "FLASHING"
    #         self.start_time = current_time
    #         self.flash_mean = None 
    #         return None, "Ready...", False

    #     elif self.state == "FLASHING":
    #         target_color, color_name = self.sequence[self.current_step]
            
    #         # Flash trong 0.8s (Tăng thời gian lên chút)
    #         if current_time - self.start_time < 0.8:
    #             # Bỏ qua 0.2s đầu cho cam thích ứng
    #             if current_time - self.start_time > 0.2:
    #                 if self.flash_mean is None:
    #                     self.flash_mean = current_mean
    #                 else:
    #                     # Logic tìm max: OK
    #                     # Lưu ý: target_color phải match với hệ màu BGR của frame
    #                     # Ví dụ: Màu đỏ phải check kênh 2 (R), Màu xanh dương check kênh 0 (B)
    #                     idx = np.argmax(target_color) 
    #                     if current_mean[idx] > self.flash_mean[idx]:
    #                         self.flash_mean = current_mean
                            
    #             return target_color, f"Look at screen ({color_name})", False
            
    #         self.state = "EVALUATING"
    #         return None, "Analyzing...", False

    #     elif self.state == "EVALUATING":
    #         if self.flash_mean is None:
    #             print("⚠️ Missed flash window (Low FPS/Face lost). Treat as no change.")
    #             self.flash_mean = self.base_mean # Gán bằng base để hiệu số = 0 -> Tự động Fail an toàn

    #         target_color, color_name = self.sequence[self.current_step]
            
    #         # Tính diff
    #         diff = self.flash_mean - self.base_mean
            
    #         # --- QUAN TRỌNG: XỬ LÝ AUTO EXPOSURE ---
    #         # Nếu cam tự điều chỉnh tối đi, diff có thể âm. 
    #         # Ta không clamp về 0 ngay mà xem xét tương quan.
            
    #         flash_idx = np.argmax(target_color) # Giả sử target_color tuân thủ BGR
            
    #         val_main = diff[flash_idx]
            
    #         # Tính noise từ các kênh còn lại
    #         others = list(diff)
    #         others.pop(flash_idx)
    #         val_noise = np.mean(others)
            
    #         # DEBUG: In ra để biết tại sao fail
    #         print(f"DEBUG: Color={color_name} | Base={self.base_mean.astype(int)} | Flash={self.flash_mean.astype(int)}")
    #         print(f"   Step {self.current_step+1}: Main={val_main:.2f}, Noise={val_noise:.2f}")

    #         # --- LOGIC PASS MỚI (LỎNG HƠN) ---
    #         is_pass = False
            
    #         # Điều kiện 1: Có sự thay đổi dương (dù nhỏ)
    #         # Hạ threshold xuống 1.5 (thay vì 3.0)
    #         if val_main > 1.5: 
    #             # Điều kiện 2: Kênh chính phải tăng nhiều hơn trung bình các kênh khác
    #             # (Tránh trường hợp sáng đều do bật đèn phòng)
    #             if val_main > val_noise:
    #                 is_pass = True
            
    #         # Bonus: Nếu chênh lệch rất lớn (>5) thì auto pass
    #         if val_main > 5.0: is_pass = True

    #         if is_pass:
    #             print("   -> ✅ OK")
    #             self.passed_steps += 1
    #         else:
    #             print("   -> ❌ FAIL")
                
    #         self.current_step += 1
    #         if self.current_step < self.total_steps:
    #             self.state = "PREPARING"
    #             self.start_time = time.time()
    #             return None, "Next color...", False
    #         else:
    #             self.state = "FINISHED"
    #             return None, "Done", False

    #     elif self.state == "FINISHED":
    #         # Pass nếu đúng 2/3 (hoặc 1/3 nếu môi trường quá khó)
    #         print(f"📊 KẾT QUẢ: {self.passed_steps}/{self.total_steps}")
    #         self.result = self.passed_steps >= 2 
    #         return None, "Success" if self.result else "Failed", True

    #     return None, "", False

    # def process(self, frame, face_bbox):
    #     """
    #     Trả về: overlay_color, status_text, is_finished
    #     """
    #     current_time = time.time()
        
    #     # Crop khuôn mặt
    #     x1, y1, x2, y2 = face_bbox
    #     h, w = y2 - y1, x2 - x1
    #     roi = frame[y1 + int(h*0.2):y2 - int(h*0.2), 
    #                 x1 + int(w*0.2):x2 - int(w*0.2)]
        
    #     if roi.size == 0: return None, "No Face", False
    #     current_mean = np.mean(roi, axis=(0, 1))

    #     # --- STATE MACHINE ---
        
    #     # 1. PREPARING (Nghỉ giữa các lần flash để lấy base)
    #     if self.state == "PREPARING":
    #         if current_time - self.start_time < 0.4: # Nghỉ 0.4s
    #             return None, "Stay still...", False
            
    #         self.base_mean = current_mean
    #         self.state = "FLASHING"
    #         self.start_time = current_time
    #         self.flash_mean = None # Reset mẫu flash
    #         return None, "Ready...", False

    #     # 2. FLASHING (Bật màu)
    #     elif self.state == "FLASHING":
    #         target_color, color_name = self.sequence[self.current_step]
            
    #         # Flash trong 0.5s
    #         if current_time - self.start_time < 0.5:
    #             # Bỏ qua 0.15s đầu để camera thích ứng
    #             if current_time - self.start_time > 0.15:
    #                 if self.flash_mean is None:
    #                     self.flash_mean = current_mean
    #                 else:
    #                     # Lấy giá trị lớn nhất ghi nhận được (lúc màn hình sáng nhất)
    #                     idx = np.argmax(target_color)
    #                     if current_mean[idx] > self.flash_mean[idx]:
    #                         self.flash_mean = current_mean
                            
    #             return target_color, f"Look at screen ({color_name})", False
            
    #         # Hết giờ Flash -> Chuyển sang tính điểm bước này
    #         self.state = "EVALUATING"
    #         return None, "Analyzing...", False

    #     # 3. EVALUATING (Chấm điểm bước hiện tại)
    #     elif self.state == "EVALUATING":
    #         if self.flash_mean is None:
    #             print("⚠️ Missed flash window (Low FPS/Face lost). Treat as no change.")
    #             self.flash_mean = self.base_mean # Gán bằng base để hiệu số = 0 -> Tự động Fail an toàn

    #         target_color, color_name = self.sequence[self.current_step]
            
    #         diff = self.flash_mean - self.base_mean
    #         diff = np.maximum(diff, 0) # Chỉ lấy tăng dương
            
    #         # Logic đơn giản hóa: Màu nào Flash thì màu đó phải TĂNG MẠNH NHẤT
    #         flash_idx = np.argmax(target_color) # 0=B, 1=G, 2=R
            
    #         # Tìm kênh tăng mạnh nhất trong thực tế
    #         actual_max_idx = np.argmax(diff)
            
    #         # Giá trị tăng của kênh chính
    #         val_main = diff[flash_idx]
            
    #         # Giá trị trung bình các kênh còn lại
    #         others = list(diff)
    #         others.pop(flash_idx)
    #         val_noise = np.mean(others)
            
    #         print(f"   Step {self.current_step+1}/{self.total_steps} ({color_name}): "
    #               f"Main={val_main:.1f}, Noise={val_noise:.1f} -> ", end="")

    #         # ĐIỀU KIỆN PASS BƯỚC NÀY:
    #         # 1. Kênh chính tăng ít nhất 3 đơn vị (tránh nhiễu)
    #         # 2. Kênh chính phải là kênh tăng mạnh nhất (Dominant)
    #         # 3. Tỷ lệ Tín hiệu / Nhiễu > 1.2 (Thấp hơn logic cũ, nhưng dùng 3 lần để bù lại)
            
    #         is_pass = False
    #         if val_main > 3.0 and actual_max_idx == flash_idx:
    #             if val_noise == 0 or (val_main / val_noise > 1.2):
    #                 is_pass = True
            
    #         if is_pass:
    #             print("✅ OK")
    #             self.passed_steps += 1
    #         else:
    #             print("❌ FAIL")
                
    #         # Chuyển sang bước tiếp theo
    #         self.current_step += 1
    #         if self.current_step < self.total_steps:
    #             self.state = "PREPARING" # Quay lại chuẩn bị cho màu sau
    #             self.start_time = time.time()
    #             return None, "Next color...", False
    #         else:
    #             self.state = "FINISHED" # Xong hết chuỗi
    #             return None, "Done", False

    #     # 4. FINISHED (Chốt hạ)
    #     elif self.state == "FINISHED":
    #         # Pass nếu đúng ít nhất 2/3 màu
    #         print(f"📊 KẾT QUẢ: {self.passed_steps}/{self.total_steps}")
    #         self.result = self.passed_steps >= 2 
    #         return None, "Success" if self.result else "Failed", True

    #     return None, "", False