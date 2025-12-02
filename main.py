from core.engine import FaceEngine
from database.storage import FaceRepository
from services import importer, camera

def main():
    # 1. Khởi tạo các thành phần cốt lõi (Core & DB)
    core_engine = FaceEngine()
    db_repo = FaceRepository()

    while True:
        print("\n=== HỆ THỐNG ĐIỂM DANH FACE ID ===")
        print("1. Import NGƯỜI MỚI (Chỉ quét user chưa có)")
        print("2. Re-train TOÀN BỘ (Quét lại tất cả - Chậm)")
        print("3. Chạy Camera (Real-time check)")
        print("4. Điểm danh NGAY (Chụp ảnh 1 lần)") 
        print("5. Thoát")
        
        choice = input("👉 Chọn chức năng: ")
        
        if choice == '1':
            importer.run_import(core_engine, db_repo, overwrite=False)
        elif choice == '2':
            confirm = input("⚠️ Bạn có chắc muốn chạy lại toàn bộ? (y/n): ")
            if confirm.lower() == 'y':
                importer.run_import(core_engine, db_repo, overwrite=True)
        elif choice == '3':
            camera.run_camera(core_engine, db_repo)
        elif choice == '4':
            # Gọi hàm mới viết
            camera.run_auto_checkin(core_engine, db_repo) 
        elif choice == '5':
            print("Tạm biệt!")
            break
        else:
            print("❌ Lựa chọn không hợp lệ.")


if __name__ == "__main__":
    main()