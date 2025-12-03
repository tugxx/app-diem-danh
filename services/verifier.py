import cv2
import numpy as np
import time
import random



# class FlashVerifier:
#     def __init__(self):
#         self.state = "IDLE" # IDLE, PREPARING, FLASHING, ANALYZING, FINISHED
#         self.flash_color = (0, 0, 0) # BGR
#         self.base_color_mean = None
#         self.flash_color_mean = None
#         self.start_time = 0
#         self.frames_captured = 0
#         self.result = False
        
#     def start_challenge(self):
#         self.state = "PREPARING"
#         self.start_time = time.time()
#         # Chọn màu ngẫu nhiên (Bỏ qua màu đen/tối)
#         # Random ra Đỏ, Xanh lá, hoặc Xanh dương đậm để dễ nhận biết
#         colors = [
#             (0, 0, 255),   # Đỏ
#             (0, 255, 0),   # Xanh lá
#             (255, 0, 0),   # Xanh dương
#             (0, 255, 255), # Vàng
#             (255, 0, 255)  # Tím
#         ]
#         self.flash_color = random.choice(colors)
#         print(f"⚡ [FLASH] Challenge Color: {self.flash_color}")

#     def process(self, frame, face_bbox):
#         """
#         Phiên bản Smart-Check: So sánh tương đối giữa các kênh màu
#         """
#         current_time = time.time()
        
#         # Cắt khuôn mặt (Lấy vùng trung tâm 50% để tránh nhiễu nền)
#         x1, y1, x2, y2 = face_bbox
#         w = x2 - x1
#         h = y2 - y1
        
#         # Crop chặt hơn (vùng má/trán)
#         roi = frame[y1 + int(h*0.3):y2 - int(h*0.3), 
#                     x1 + int(w*0.3):x2 - int(w*0.3)]
        
#         if roi.size == 0: return None, "No Face", False

#         # Tính màu trung bình
#         current_mean = np.mean(roi, axis=(0, 1)) 

#         # --- STATE MACHINE ---
        
#         # 1. PREPARING (Lấy mẫu nền)
#         if self.state == "PREPARING":
#             if current_time - self.start_time < 0.5: # Giảm thời gian chờ xuống 0.5s cho nhanh
#                 return None, "Stay still...", False
            
#             self.base_color_mean = current_mean
#             self.state = "FLASHING"
#             self.start_time = current_time
#             return None, "Ready!", False

#         # 2. FLASHING (Bật đèn)
#         elif self.state == "FLASHING":
#             # Flash trong 0.6s
#             if current_time - self.start_time < 0.6:
#                 # Bỏ qua 0.2s đầu tiên (đợi màn hình sáng hẳn và camera thích ứng)
#                 if current_time - self.start_time > 0.2:
#                     # Lấy mẫu liên tục và update (để lấy được lúc sáng nhất)
#                     if self.flash_color_mean is None:
#                         self.flash_color_mean = current_mean
#                     else:
#                         # Lấy max value của kênh màu chủ đạo
#                         idx = np.argmax(self.flash_color)
#                         if current_mean[idx] > self.flash_color_mean[idx]:
#                             self.flash_color_mean = current_mean

#                 return self.flash_color, "Analysing...", False
            
#             self.state = "ANALYZING"
#             return None, "Checking...", False

#         # 3. ANALYZING (Phân tích quang phổ)
#         elif self.state == "ANALYZING":
#             # Tính độ lệch: Lúc Flash - Lúc Thường
#             diff = self.flash_color_mean - self.base_color_mean
            
#             # Làm tròn về 0 nếu âm (chỉ quan tâm tăng sáng)
#             diff = np.maximum(diff, 0)
            
#             print(f"📊 Diff Raw: B={diff[0]:.1f}, G={diff[1]:.1f}, R={diff[2]:.1f}")

#             # Xác định các kênh màu
#             # Ví dụ: Flash Tím (255, 0, 255) -> Flash Channels là [0, 2] (Blue, Red)
#             # Non-Flash Channel là [1] (Green)
            
#             flash_channels = []
#             non_flash_channels = []
            
#             for i in range(3):
#                 if self.flash_color[i] > 100: # Kênh nào > 100 là kênh Flash
#                     flash_channels.append(i)
#                 else:
#                     non_flash_channels.append(i)
            
#             # --- LOGIC CHỐNG GIẢ MẠO ---
            
#             # 1. Tính mức tăng trung bình của kênh Flash
#             if len(flash_channels) > 0:
#                 avg_flash_increase = np.mean(diff[flash_channels])
#             else:
#                 avg_flash_increase = 0
                
#             # 2. Tính mức tăng trung bình của kênh KHÔNG Flash (Nhiễu)
#             if len(non_flash_channels) > 0:
#                 avg_noise_increase = np.mean(diff[non_flash_channels])
#             else:
#                 avg_noise_increase = 0
            
#             print(f"🔍 Analysis: Signal={avg_flash_increase:.1f} vs Noise={avg_noise_increase:.1f}")

#             # --- CÁC ĐIỀU KIỆN PASS (Cực gắt) ---
            
#             # Điều kiện 1: Phải có phản xạ dương (Mặt phải sáng lên)
#             has_reflection = avg_flash_increase > 1.5 
            
#             # Điều kiện 2: Tín hiệu phải mạnh hơn Nhiễu ít nhất 2 lần (QUAN TRỌNG)
#             # Ảnh giả thường có Signal ~ Noise (tăng đều) -> Tỷ lệ ~ 1.0 -> FAIL
#             # Mặt thật hấp thụ màu lạ tốt hơn -> Tỷ lệ > 2.0 -> PASS
#             ratio_check = False
#             if avg_noise_increase == 0:
#                 ratio_check = True # Không có nhiễu thì quá tốt
#             else:
#                 ratio = avg_flash_increase / avg_noise_increase
#                 print(f"📉 Signal-to-Noise Ratio: {ratio:.2f} (Yêu cầu > 1.8)")
#                 ratio_check = ratio > 1.8 

#             # Điều kiện 3: Chống cháy sáng (Screen-on-Screen Attack)
#             # Nếu cầm điện thoại soi vào cam, độ sáng thường tăng cực mạnh (> 30)
#             # Da người thật độ nhám cao, ít khi tăng quá 25 đơn vị trừ khi đèn cực mạnh
#             not_too_bright = avg_flash_increase < 25.0

#             # TỔNG HỢP
#             if not has_reflection:
#                 print("❌ FAIL: Không thấy phản xạ ánh sáng (Màn hình tối/Xa quá?)")
#                 self.result = False
#             elif not not_too_bright:
#                 print("❌ FAIL: Phản xạ quá mạnh (Nghi vấn màn hình điện thoại)")
#                 self.result = False
#             elif not ratio_check:
#                 print("❌ FAIL: Tăng sáng đồng đều (Nghi vấn ảnh 2D)")
#                 self.result = False
#             else:
#                 print("✅ PASS: Phản xạ quang phổ chuẩn da người.")
#                 self.result = True

#             self.state = "FINISHED"
#             return None, "Done", True
            
#         return None, "", False

    # def process(self, frame, face_bbox):
    #     """
    #     Hàm này trả về:
    #     - overlay_color: Màu cần phủ lên màn hình (None nếu không flash)
    #     - status_text: Chữ hiển thị
    #     - is_finished: True nếu đã kiểm tra xong
    #     """
    #     current_time = time.time()
        
    #     # Cắt khuôn mặt (ROI) để tính toán màu
    #     x1, y1, x2, y2 = face_bbox
    #     # Lấy vùng trung tâm khuôn mặt (bỏ tóc, bỏ nền) để chính xác hơn
    #     h_face = y2 - y1
    #     w_face = x2 - x1
    #     roi = frame[y1 + int(h_face*0.2):y2 - int(h_face*0.2), 
    #                 x1 + int(w_face*0.2):x2 - int(w_face*0.2)]
        
    #     if roi.size == 0: return None, "No Face", False

    #     # Tính màu trung bình của khuôn mặt hiện tại
    #     current_mean = np.mean(roi, axis=(0, 1)) # Trả về (B, G, R) trung bình

    #     # --- STATE MACHINE ---
        
    #     # 1. Giai đoạn lấy mẫu nền (Lúc màn hình bình thường)
    #     if self.state == "PREPARING":
    #         if current_time - self.start_time < 1.0: # Chờ 1s để ổn định
    #             return None, "Stay still...", False
            
    #         self.base_color_mean = current_mean
    #         self.state = "FLASHING"
    #         self.start_time = current_time # Reset time cho phase sau
    #         return None, "Ready!", False

    #     # 2. Giai đoạn FLASH (Bật màu màn hình)
    #     elif self.state == "FLASHING":
    #         # Giữ màu trong 0.8 giây
    #         if current_time - self.start_time < 0.8:
    #             # Chờ khoảng 0.3s cho camera kịp thích ứng exposure rồi mới lấy mẫu
    #             if current_time - self.start_time > 0.3:
    #                 self.flash_color_mean = current_mean
                
    #             return self.flash_color, "Analysing Light...", False
            
    #         # Hết giờ flash -> Chuyển sang tính toán
    #         self.state = "ANALYZING"
    #         return None, "Checking...", False

    #     # 3. Giai đoạn Tính toán
    #     elif self.state == "ANALYZING":
    #         # Logic: So sánh sự thay đổi màu sắc
    #         # Ví dụ: Flash màu Đỏ (0, 0, 255) -> Kênh R của mặt phải tăng mạnh hơn B và G
            
    #         diff = self.flash_color_mean - self.base_color_mean
    #         print(f"📊 Color Diff (B,G,R): {diff}")
            
    #         # Lấy kênh màu chủ đạo của Flash (ví dụ Flash Đỏ thì index=2)
    #         main_channel_idx = np.argmax(self.flash_color) 
            
    #         # Kiểm tra: Kênh màu chủ đạo có tăng lên đáng kể không?
    #         # Và phải tăng nhiều hơn các kênh còn lại
    #         has_reflection = (diff[main_channel_idx] > 10) and \
    #                          (diff[main_channel_idx] > diff[(main_channel_idx+1)%3]) and \
    #                          (diff[main_channel_idx] > diff[(main_channel_idx+2)%3])
            
    #         self.result = has_reflection
    #         self.state = "FINISHED"
    #         return None, "Done", True
            
    #     return None, "", False

    # def reset(self):
    #     self.state = "IDLE"
    #     self.base_color_mean = None


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
        Trả về: overlay_color, status_text, is_finished
        """
        current_time = time.time()
        
        # Crop khuôn mặt
        x1, y1, x2, y2 = face_bbox
        h, w = y2 - y1, x2 - x1
        roi = frame[y1 + int(h*0.2):y2 - int(h*0.2), 
                    x1 + int(w*0.2):x2 - int(w*0.2)]
        
        if roi.size == 0: return None, "No Face", False
        current_mean = np.mean(roi, axis=(0, 1))

        # --- STATE MACHINE ---
        
        # 1. PREPARING (Nghỉ giữa các lần flash để lấy base)
        if self.state == "PREPARING":
            if current_time - self.start_time < 0.4: # Nghỉ 0.4s
                return None, "Stay still...", False
            
            self.base_mean = current_mean
            self.state = "FLASHING"
            self.start_time = current_time
            self.flash_mean = None # Reset mẫu flash
            return None, "Ready...", False

        # 2. FLASHING (Bật màu)
        elif self.state == "FLASHING":
            target_color, color_name = self.sequence[self.current_step]
            
            # Flash trong 0.5s
            if current_time - self.start_time < 0.5:
                # Bỏ qua 0.15s đầu để camera thích ứng
                if current_time - self.start_time > 0.15:
                    if self.flash_mean is None:
                        self.flash_mean = current_mean
                    else:
                        # Lấy giá trị lớn nhất ghi nhận được (lúc màn hình sáng nhất)
                        idx = np.argmax(target_color)
                        if current_mean[idx] > self.flash_mean[idx]:
                            self.flash_mean = current_mean
                            
                return target_color, f"Look at screen ({color_name})", False
            
            # Hết giờ Flash -> Chuyển sang tính điểm bước này
            self.state = "EVALUATING"
            return None, "Analyzing...", False

        # 3. EVALUATING (Chấm điểm bước hiện tại)
        elif self.state == "EVALUATING":
            target_color, color_name = self.sequence[self.current_step]
            
            diff = self.flash_mean - self.base_mean
            diff = np.maximum(diff, 0) # Chỉ lấy tăng dương
            
            # Logic đơn giản hóa: Màu nào Flash thì màu đó phải TĂNG MẠNH NHẤT
            flash_idx = np.argmax(target_color) # 0=B, 1=G, 2=R
            
            # Tìm kênh tăng mạnh nhất trong thực tế
            actual_max_idx = np.argmax(diff)
            
            # Giá trị tăng của kênh chính
            val_main = diff[flash_idx]
            
            # Giá trị trung bình các kênh còn lại
            others = list(diff)
            others.pop(flash_idx)
            val_noise = np.mean(others)
            
            print(f"   Step {self.current_step+1}/{self.total_steps} ({color_name}): "
                  f"Main={val_main:.1f}, Noise={val_noise:.1f} -> ", end="")

            # ĐIỀU KIỆN PASS BƯỚC NÀY:
            # 1. Kênh chính tăng ít nhất 3 đơn vị (tránh nhiễu)
            # 2. Kênh chính phải là kênh tăng mạnh nhất (Dominant)
            # 3. Tỷ lệ Tín hiệu / Nhiễu > 1.2 (Thấp hơn logic cũ, nhưng dùng 3 lần để bù lại)
            
            is_pass = False
            if val_main > 3.0 and actual_max_idx == flash_idx:
                if val_noise == 0 or (val_main / val_noise > 1.2):
                    is_pass = True
            
            if is_pass:
                print("✅ OK")
                self.passed_steps += 1
            else:
                print("❌ FAIL")
                
            # Chuyển sang bước tiếp theo
            self.current_step += 1
            if self.current_step < self.total_steps:
                self.state = "PREPARING" # Quay lại chuẩn bị cho màu sau
                self.start_time = time.time()
                return None, "Next color...", False
            else:
                self.state = "FINISHED" # Xong hết chuỗi
                return None, "Done", False

        # 4. FINISHED (Chốt hạ)
        elif self.state == "FINISHED":
            # Pass nếu đúng ít nhất 2/3 màu
            print(f"📊 KẾT QUẢ: {self.passed_steps}/{self.total_steps}")
            self.result = self.passed_steps >= 2 
            return None, "Success" if self.result else "Failed", True

        return None, "", False