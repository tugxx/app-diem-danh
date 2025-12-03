import os
import cv2
import numpy as np
from tqdm import tqdm

def run_import(engine, repository, dataset_dir="dataset", overwrite=False):
    """
    overwrite=False: Nếu user đã có trong DB thì bỏ qua (Mặc định).
    overwrite=True: Tính toán lại từ đầu cho tất cả user trong folder.
    """
    print("\n--- 📂 BẮT ĐẦU IMPORT DỮ LIỆU (INCREMENTAL) ---")
    
    if not os.path.exists(dataset_dir):
        print(f"❌ Không tìm thấy thư mục '{dataset_dir}'.")
        return

    # Lấy danh sách tất cả folder trong dataset
    all_users = os.listdir(dataset_dir)
    
    # Lọc danh sách cần xử lý
    users_to_process = []
    skipped_count = 0

    if overwrite:
        users_to_process = all_users
        print("⚠️ Chế độ GHI ĐÈ: Sẽ xử lý lại toàn bộ dữ liệu.")
    else:
        # Chỉ lấy những người CHƯA CÓ trong database
        existing_users = repository.cache.keys()
        for user in all_users:
            if user in existing_users:
                skipped_count += 1
            else:
                users_to_process.append(user)
        
        if skipped_count > 0:
            print(f"⏩ Đã bỏ qua {skipped_count} người (đã tồn tại trong DB).")

    if len(users_to_process) == 0:
        print("✅ Hệ thống đã cập nhật đầy đủ. Không có dữ liệu mới.")
        return

    print(f"🚀 Đang xử lý {len(users_to_process)} người dùng mới...")
    count_success = 0

    # Chỉ chạy vòng lặp với những người cần xử lý
    for user_name in tqdm(users_to_process):
        user_folder = os.path.join(dataset_dir, user_name)
        if not os.path.isdir(user_folder): continue

        embeddings = []
        valid_images = 0
        
        # Duyệt file ảnh
        for file_name in os.listdir(user_folder):
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(user_folder, file_name)
            img = cv2.imread(img_path)
            if img is None: continue

            faces = engine.extract_faces(img)
            if len(faces) == 0: continue
            
            # Logic: Lấy mặt to nhất
            main_face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
            embeddings.append(main_face.embedding)
            valid_images += 1
        
        # Tính toán & Lưu
        if len(embeddings) > 0:
            mean_emb = np.mean(embeddings, axis=0)
            mean_emb = mean_emb / np.linalg.norm(mean_emb)
            
            repository.save_user(user_name, mean_emb)
            count_success += 1
        else:
            print(f"⚠️ {user_name}: Không tìm thấy khuôn mặt hợp lệ.")

    print(f"\n✅ HOÀN TẤT! Đã thêm mới/cập nhật {count_success} người dùng.")