import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Conv_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Linear_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Depth_Wise(nn.Module):
    def __init__(self, c1, c2, c3, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        self.conv = Conv_block(c1_in, out_c=c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        # Conv_dw: Input = in_c, Output = groups (thường là in_c), Groups = groups
        self.conv_dw = Conv_block(c2_in, c2_out, groups=c2_in, kernel=kernel, padding=padding, stride=stride)
        # Project: Input = groups, Output = out_c
        self.project = Linear_block(c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(nn.Module):
    def __init__(self, c1, c2, c3, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for i in range(num_block):
            c1_tuple = c1[i]
            c2_tuple = c2[i]
            c3_tuple = c3[i]
            modules.append(Depth_Wise(c1_tuple, c2_tuple, c3_tuple, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.model(x)


class MiniFASNet(nn.Module):
    def __init__(self, keep, embedding_size, conv6_kernel=(7, 7), drop_p=0.0, num_classes=3, img_channel=3):
        super(MiniFASNet, self).__init__()
        self.embedding_size = embedding_size
        self.conv1 = Conv_block(img_channel, keep[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(keep[0], keep[1], kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=keep[1])
        c1 = [(keep[1], keep[2])]
        c2 = [(keep[2], keep[3])]
        c3 = [(keep[3], keep[4])]
        self.conv_23 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[3])
        c1 = [(keep[4], keep[5]), (keep[7], keep[8]), (keep[10], keep[11]), (keep[13], keep[14])]
        c2 = [(keep[5], keep[6]), (keep[8], keep[9]), (keep[11], keep[12]), (keep[14], keep[15])]
        c3 = [(keep[6], keep[7]), (keep[9], keep[10]), (keep[12], keep[13]), (keep[15], keep[16])]
        self.conv_3 = Residual(c1, c2, c3, num_block=4, groups=keep[4], kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        c1 = [(keep[16], keep[17])]
        c2 = [(keep[17], keep[18])]
        c3 = [(keep[18], keep[19])]
        self.conv_34 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[19])
        c1 = [(keep[19], keep[20]), (keep[22], keep[23]), (keep[25], keep[26]), (keep[28], keep[29]),
              (keep[31], keep[32]), (keep[34], keep[35])]
        c2 = [(keep[20], keep[21]), (keep[23], keep[24]), (keep[26], keep[27]), (keep[29], keep[30]),
              (keep[32], keep[33]), (keep[35], keep[36])]
        c3 = [(keep[21], keep[22]), (keep[24], keep[25]), (keep[27], keep[28]), (keep[30], keep[31]),
              (keep[33], keep[34]), (keep[36], keep[37])]
        self.conv_4 = Residual(c1, c2, c3, num_block=6, groups=keep[19], kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        c1 = [(keep[37], keep[38])]
        c2 = [(keep[38], keep[39])]
        c3 = [(keep[39], keep[40])]
        self.conv_45 = Depth_Wise(c1[0], c2[0], c3[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=keep[40])
        c1 = [(keep[40], keep[41]), (keep[43], keep[44])]
        c2 = [(keep[41], keep[42]), (keep[44], keep[45])]
        c3 = [(keep[42], keep[43]), (keep[45], keep[46])]
        self.conv_5 = Residual(c1, c2, c3, num_block=2, groups=keep[40], kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(keep[46], keep[47], kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(keep[47], keep[48], groups=keep[48], kernel=conv6_kernel, stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.drop = nn.Dropout(p=drop_p)
        self.prob = nn.Linear(embedding_size, num_classes, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        if self.embedding_size != 512:
            out = self.linear(out)
        out = self.bn(out)
        out = self.drop(out)
        out = self.prob(out)
        return out


# Từ điển chứa kích thước Pruned Model (QUAN TRỌNG NHẤT)
keep_dict = {
    '1.8M_': [32, 32, 103, 103, 64, 13, 13, 64, 13, 13, 64, 13,
              13, 64, 13, 13, 64, 231, 231, 128, 231, 231, 128, 52,
              52, 128, 26, 26, 128, 77, 77, 128, 26, 26, 128, 26, 26,
              128, 308, 308, 128, 26, 26, 128, 26, 26, 128, 512, 512]
}


def MiniFASNetV2(embedding_size=128, conv6_kernel=(5, 5), drop_p=0.2, num_classes=3, img_channel=3):
    return MiniFASNet(keep_dict['1.8M_'], embedding_size, conv6_kernel, drop_p, num_classes, img_channel)


# ==========================================
# 2. CLASS CROP ẢNH 
# ==========================================
class CropImage:
    @staticmethod
    def _get_new_box(src_w, src_h, bbox, scale):
        x = bbox[0]
        y = bbox[1]
        box_w = bbox[2]
        box_h = bbox[3]
        
        # Tính toán scale logic
        scale = min((src_h-1)/box_h, min((src_w-1)/box_w, scale))
        
        new_width = box_w * scale
        new_height = box_h * scale
        center_x, center_y = box_w/2+x, box_h/2+y
        
        left_top_x = center_x - new_width/2
        left_top_y = center_y - new_height/2
        right_bottom_x = center_x + new_width/2
        right_bottom_y = center_y + new_height/2
        
        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0
        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0
        if right_bottom_x > src_w-1:
            left_top_x -= right_bottom_x-src_w+1
            right_bottom_x = src_w-1
        if right_bottom_y > src_h-1:
            left_top_y -= right_bottom_y-src_h+1
            right_bottom_y = src_h-1
            
        return int(left_top_x), int(left_top_y), \
               int(right_bottom_x), int(right_bottom_y)

    def crop(self, org_img, bbox, scale, out_w, out_h, crop=True):
        if not crop:
            dst_img = cv2.resize(org_img, (out_w, out_h))
        else:
            src_h, src_w, _ = np.shape(org_img)
            left_top_x, left_top_y, \
                right_bottom_x, right_bottom_y = self._get_new_box(src_w, src_h, bbox, scale)
            
            img = org_img[left_top_y: right_bottom_y+1,
                          left_top_x: right_bottom_x+1]
            dst_img = cv2.resize(img, (out_w, out_h))
        return dst_img


# ==========================================
# 3. CLASS HỆ THỐNG
# ==========================================

class AntiSpoofSystem:
    def __init__(self, model_path, device_id=0):
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        print(f"AntiSpoof running on: {self.device}")
        
        if not os.path.exists(model_path):
             print(f"❌ CRITICAL: Không tìm thấy file tại {model_path}")
             return

        # Khởi tạo kiến trúc MiniFASNetV2 với kernel 5x5 (cho ảnh 80x80)
        # Sử dụng đúng cấu hình kênh từ keep_dict['1.8M_']
        self.model = MiniFASNetV2(conv6_kernel=(5, 5), num_classes=3)

        # 1. Load file weights lên trước để kiểm tra cấu trúc
        try:
            state_dict = torch.load(model_path, map_location=self.device) 
            
            # Xử lý key thừa "module."
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            self.model.load_state_dict(new_state_dict, strict=True) # hoặc chuyển về strict=False để nó tự khớp những gì khớp được
            print(">> Model loaded (Mode: V1 Compatibility).")

        except Exception as e:
            print(f"CRITICAL ERROR: Cannot load weights file. {e}")
            raise e
        
        self.model.to(self.device)
        self.model.eval()

        # Khởi tạo bộ crop
        self.cropper = CropImage()


    # def preprocess(self, img_crop):
    #     """Chuyển ảnh OpenCV sang Tensor 80x80 chuẩn của Model"""
    #     img = cv2.resize(img_crop, (80, 80))
    #     img = img.astype(np.float32)
    #     img = img.transpose((2, 0, 1)) 
    #     img = torch.from_numpy(img).unsqueeze(0).to(self.device)
    #     return img


    def predict(self, frame, face_bbox):
        """
        Input: Frame (BGR), face_bbox [x, y, w, h] (Lưu ý: bbox dạng w,h chứ không phải x2, y2)
        Nếu bbox là [x1, y1, x2, y2] thì phải đổi lại bên dưới
        """
        # CHUYỂN ĐỔI BBOX
        # Giả sử đầu vào face_bbox là [x1, y1, x2, y2] (từ thư viện nhận diện mặt phổ biến)
        x1, y1, x2, y2 = face_bbox
        w_box = x2 - x1
        h_box = y2 - y1
        bbox_standard = [x1, y1, w_box, h_box] # Định dạng cho hàm crop: [x, y, w, h]

        # 1. CROP CHUẨN (Scale 2.7 cho file model 2.7_80x80)
        img_crop = self.cropper.crop(frame, bbox_standard, scale=2.7, out_w=80, out_h=80)
        
        # 2. CHUYỂN MÀU BGR -> RGB 
        img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)

        # 3. CHUYỂN SANG TENSOR
        img_crop = img_crop.astype(np.float32)
        img_crop = img_crop.transpose((2, 0, 1)) # HWC -> CHW
        img_crop = torch.from_numpy(img_crop).unsqueeze(0).to(self.device)
        
        # 4. PREDICT
        with torch.no_grad():
            output = self.model(img_crop)
            output = F.softmax(output, dim=1).cpu().numpy()[0]
            
        # Class 1 = Real, Class 0/2 = Fake
        score_real = output[1]
        is_real = score_real > 0.90 # Ngưỡng tin cậy
        
        return score_real, is_real