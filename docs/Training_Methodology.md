# Báo cáo các Kỹ thuật Huấn luyện Model Chẩn đoán Viêm khớp gối

Tài liệu này tổng hợp tất cả các phương pháp, kỹ thuật đã được áp dụng trong quy trình huấn luyện các mô hình Deep Learning để phân loại mức độ viêm khớp gối (Kellgren-Lawrence).

## 1. Quản lý Dữ liệu (Data Management)

### 1.1. Tiền xử lý & Làm sạch
- **Loại bỏ Tệp trùng lặp**: Trước khi bắt đầu mỗi phiên training, một bước kiểm tra và loại bỏ các ảnh bị trùng lặp được thực hiện. Thuật toán sử dụng mã băm MD5 để xác định và xóa các bản sao, đảm bảo bộ dữ liệu không bị sai lệch (bias) bởi các mẫu xuất hiện nhiều lần.

### 1.2. Tăng cường Dữ liệu (Data Augmentation)
Một chuỗi các phép biến đổi được áp dụng để tăng sự đa dạng của dữ liệu, giúp model khái quát hóa tốt hơn và chống lại overfitting.

- **Tăng cường Độ tương phản (CLAHE)**:
  - **Kỹ thuật**: `OpenCVCLAHE` (Contrast Limited Adaptive Histogram Equalization).
  - **Mô tả**: Được áp dụng ngay trên ảnh gốc (sau khi được pad vuông). Kỹ thuật này rất hiệu quả với ảnh X-quang, giúp làm nổi bật các chi tiết ở rìa xương và các vùng có độ tương phản thấp. Nó được áp dụng trên kênh độ sáng (Luminance) của không gian màu LAB để không làm thay đổi màu sắc ảnh.

- **Các phép biến đổi Hình học và Màu sắc**:
  - `SquarePadOpenCV`: Đệm các ảnh chữ nhật thành ảnh vuông để giữ nguyên tỷ lệ khi resize.
  - `RandomAffine`: Áp dụng các phép biến đổi hình học ngẫu nhiên ở mức độ nhẹ để giả lập các góc chụp khác nhau:
    - Xoay (degrees): ±3°
    - Dịch chuyển (translate): ±2% theo chiều ngang và dọc.
    - Co giãn (scale): từ 95% đến 105%.
    - Biến dạng cắt (shear): ±2°
  - `ColorJitter`: Thay đổi nhẹ độ sáng và độ tương phản (±3%) để model không bị phụ thuộc vào điều kiện phơi sáng của ảnh.
  - `RandomHorizontalFlip`: Lật ảnh ngẫu nhiên theo chiều ngang với xác suất 50%.

- **Chuẩn hóa (Normalization)**: Sau tất cả các phép biến đổi, ảnh được chuẩn hóa bằng giá trị trung bình và độ lệch chuẩn của bộ dữ liệu ImageNet.

## 2. Cấu trúc Model & Transfer Learning

### 2.1. Kiến trúc
Dự án hỗ trợ nhiều kiến trúc model thông qua một "Model Registry":
- `EfficientNet-B0`
- `MobileNet-V2`
- `DenseNet-121`

### 2.2. Kỹ thuật Transfer Learning
- **Trọng số được huấn luyện trước (Pretrained Weights)**: Tất cả các model đều được khởi tạo với trọng số đã được huấn luyện trước trên bộ dữ liệu ImageNet, sử dụng API `weights` mới của `torchvision`.
- **Thay thế Lớp Classifier**: Lớp classifier cuối cùng của mỗi model được thay thế bằng một lớp mới phù hợp với bài toán 5 lớp (KL-Grade 0-4).
- **Tăng cường Dropout**: Tỷ lệ Dropout trong lớp classifier đã được tăng lên **0.5** (từ 0.2 mặc định) để tăng cường khả năng chống overfitting.

### 2.3. Chiến lược Đóng băng & Fine-tuning
- **Đóng băng một phần (Partial Freezing) cho EfficientNet-B0**:
  - **Mô tả**: Khi training `EfficientNet-B0` (không có cờ `--fine-tune`), một chiến lược tùy chỉnh được áp dụng: các khối đặc trưng sớm (0-3) bị đóng băng, trong khi các khối sâu hơn (4-7) và lớp classifier được "mở băng" để huấn luyện.
  - **Lý do**: Giữ lại các đặc trưng bậc thấp (cạnh, góc) đã học được từ ImageNet và chỉ tinh chỉnh các đặc trưng bậc cao phức tạp hơn cho phù hợp với ảnh X-quang.

- **Fine-tuning Toàn bộ mạng (`--fine-tune`)**:
  - **Mô tả**: Khi sử dụng cờ `--fine-tune`, toàn bộ mạng sẽ được huấn luyện.
  - **Learning Rate vi sai (Differential LR)**: Một learning rate nhỏ hơn (`lr * 0.1`) được áp dụng cho các lớp backbone, và learning rate tiêu chuẩn (`lr`) được áp dụng cho lớp classifier. Điều này giúp tinh chỉnh các lớp backbone một cách cẩn thận mà không phá vỡ các trọng số đã học.

## 3. Quy trình Huấn luyện (Training Loop)

### 3.1. Optimizer
- **Loại**: `AdamW` được sử dụng. Đây là một phiên bản cải tiến của Adam, thường cho kết quả tốt hơn trong các bài toán thị giác bằng cách xử lý `weight_decay` một cách hiệu quả hơn.

### 3.2. Hàm Loss
- **Mặc định**: `CrossEntropyLoss`.
- **Tùy chọn (`--focal-loss`)**: Hỗ trợ sử dụng `SigmoidFocalLoss` (`gamma=2.0`, `alpha=0.25`). Kỹ thuật này tập trung vào việc học các mẫu khó và hữu ích cho các bộ dữ liệu mất cân bằng.

### 3.3. Learning Rate & Scheduler
- **Learning Rate khởi điểm**: Mặc định là **`1e-4` (0.0001)**. Đây là một mức LR an toàn và hiệu quả cho việc fine-tuning.
- **Scheduler**: `ReduceLROnPlateau` được sử dụng để tự động điều chỉnh LR.
  - **Cơ chế**: Theo dõi `validation loss`.
  - `patience=3`: Nếu `validation loss` không cải thiện trong 3 epochs liên tiếp, LR sẽ được giảm.
  - `factor=0.5`: Khi giảm, LR sẽ được nhân với 0.5 (giảm một nửa).
  - `min_lr=1e-6`: LR sẽ không bao giờ giảm xuống dưới mức này.

## 4. Quản lý & Tái lập
- **Giao diện Dòng lệnh**: Toàn bộ quy trình training có thể được điều khiển thông qua các tham số dòng lệnh (ví dụ: `--model`, `--dataset-name`, `--lr`, `--fine-tune`, v.v.).
- **Registry Pattern**: Sử dụng `ModelRegistry` và `DatasetRegistry` để dễ dàng thêm mới và lựa chọn các thành phần mà không cần sửa đổi code logic chính.
- **Lưu Checkpoint**: Sau mỗi epoch, trạng thái của model, optimizer và scheduler được lưu vào file `last_model.pth`. Nếu model đạt được validation accuracy cao hơn, nó sẽ được lưu vào `best_model.pth`.
- **Khôi phục Training**: Chương trình có khả năng tự động tìm và tải file `last_model.pth` để tiếp tục quá trình training từ đúng epoch đã dừng.
