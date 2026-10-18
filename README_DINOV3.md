TF-C + DINOv3 – Thay thế hai Encoder của mô hình TF-C

1. Giới thiệu
Dự án được phát triển từ mã nguồn TF-C Pretraining, một mô hình học biểu diễn chuỗi thời gian từ hai nhánh thời gian và tần số.
Yêu cầu thực hiện: Thay Encoder (Backbone) của TF-C bằng DINOv3 và đưa mã nguồn đã cập nhật lên GitHub, kèm README chỉ rõ những chỗ đã sửa.
Trong phiên bản này, em thay cả hai Transformer Encoder của bản code TF-C bằng hai backbone DINOv3 ViT-S/16 riêng biệt. Do DINOv3 xử lý ảnh, em bổ sung bước chuyển tín hiệu EEG một chiều thành ảnh để đưa vào mô hình.

2. Những thay đổi trong mã nguồn
File	Nội dung đã cập nhật
[`code/TFC/model.py`](code/TFC/model.py)	Thay cả Time Encoder và Frequency Encoder từ Transformer sang DINOv3; lấy đặc trưng và đưa qua hai projector.
[`code/TFC/dinov3_adapter.py`](code/TFC/dinov3_adapter.py)	File mới. Chuyển EEG thành ảnh miền thời gian (tín hiệu, độ biến thiên, đường bao) và ảnh phổ miền tần số (log-STFT), kích thước 224 × 224.
[`code/TFC/dataloader.py`](code/TFC/dataloader.py)	Điều chỉnh dữ liệu và cách tạo dữ liệu tăng cường để phù hợp với hai nhánh DINOv3; bổ sung cách chọn tập con để chạy thử.
[`code/TFC/main.py`](code/TFC/main.py)	Bổ sung tham số chọn backbone, kích thước ảnh và chế độ --subset; đặt tên thư mục checkpoint riêng cho DINOv3.
[`code/TFC/trainer.py`](code/TFC/trainer.py)	Điều chỉnh một số phần xử lý khi huấn luyện, đánh giá và tải checkpoint.
[`code/TFC/requirements-dinov3.txt`](code/TFC/requirements-dinov3.txt)	File mới. Danh sách thư viện bổ sung.
[`code/TFC/smoke_test_dinov3.py`](code/TFC/smoke_test_dinov3.py)	File mới. Kiểm tra đầu vào, đầu ra và gradient bằng backbone giả lập.


Vị trí thay backbone chính: code/TFC/model.py, tại hai thành phần encoder_t và encoder_f. Mỗi nhánh dùng một DINOv3 độc lập; đặc trưng 384 chiều được chiếu về 128 chiều mỗi nhánh rồi ghép thành 256 chiều cho bộ phân loại.
Các mô hình khác trong code/baselines không sửa.
3. Kết quả đã thực hiện
Kiểm tra hai backbone DINOv3 thật:
- Tải được trọng số pretrained DINOv3 ViT-S/16 từ Hugging Face.
- Chạy thành công cả hai Encoder độc lập với dữ liệu EEG thật.
- Đầu ra hai nhánh đều có kích thước 384 chiều; qua projector còn 128 chiều mỗi nhánh.
- Vector ghép có kích thước 256 chiều và không chứa giá trị NaN/Inf.
Pre-training trên SleepEEG:
Nội dung	Kết quả
Dữ liệu chạy thử	1.280 mẫu SleepEEG (--subset)
Số epoch	20
Training Loss epoch 1	5.2391
Training Loss epoch 20	2.0547
Thời gian	23 phút 56 giây
Checkpoint	Lưu thành công tại code/experiments_logs/.../pre_train_seed_42_dinov3_both/saved_models/ckp_last.pt


Fine-tuning và Testing trên Epilepsy:
- Hoàn thành 20 epoch Fine-tuning và bước Testing.
- Dữ liệu Fine-tuning: 60 mẫu; tập Testing ở chế độ chạy thử: 600 mẫu (--subset).
- Thời gian Fine-tuning và Testing: 1 giờ 28 phút 46 giây.
Bộ phân loại	Accuracy	F1-score	AUROC	AUPRC
MLP	18,50%	15,61%	42,54%	50,38%
KNN	85,50%	64,16%	74,68%	73,29%


Các số liệu được lấy từ log chạy thử: MLP dùng dòng “Best Testing Performance”, KNN dùng kết quả xuất hiện ở epoch 2–20. Log KNN hiển thị chỉ số dưới dạng từ 0 đến 1 nên đã đổi sang phần trăm cho thống nhất. Đây là kết quả trên tập test con 600 mẫu, không phải đánh giá chính thức trên toàn bộ Epilepsy.
Nhận xét: Mô hình đã chạy được đủ quy trình từ Pre-training đến Fine-tuning và Testing. Loss ở giai đoạn Pre-training giảm, nhưng kết quả của MLP còn thấp và log có cảnh báo về Precision do có lớp chưa được dự đoán. Kết quả này chưa đủ để kết luận DINOv3 cải thiện so với TF-C gốc.

4. Cách chạy
Tại thư mục code/TFC, cài thư viện:
pip install -r requirements-dinov3.txt
Trọng số DINOv3 yêu cầu được cấp quyền trên Hugging Face. Sau khi được chấp thuận, đăng nhập:
hf auth login
Kiểm tra mô hình:
python smoke_test_dinov3.py
Pre-training:
python main.py --training_mode pre_train --pretrain_dataset SleepEEG --target_dataset Epilepsy --device cpu --subset
Fine-tuning và Testing (sử dụng checkpoint Pre-training đã tạo):
python main.py --training_mode fine_tune_test --pretrain_dataset SleepEEG --target_dataset Epilepsy --device cpu --subset
Cần chuẩn bị dữ liệu SleepEEG và Epilepsy theo cấu trúc thư mục mà mã nguồn TF-C yêu cầu. Tùy chọn --subset dùng để chạy thử trên một phần dữ liệu, giúp giảm thời gian thực hiện. Trọng số DINOv3 và checkpoint huấn luyện không được đưa kèm trong repository.

5. Kết luận
Em đã hoàn thành việc thay cả hai Encoder của TF-C bằng DINOv3, bổ sung bước chuyển đổi tín hiệu EEG sang ảnh và chạy thành công cả Pre-training 20 epochs lẫn Fine-tuning/Testing 20 epochs trên tập dữ liệu chạy thử. Các kết quả hiện tại chủ yếu chứng minh mã nguồn sau khi thay backbone có thể chạy được; việc cải thiện độ chính xác cần được đánh giá thêm trên tập dữ liệu đầy đủ và so sánh công bằng với mô hình gốc.
Tài liệu tham khảo
- TF-C – mã nguồn gốc của Harvard
- DINOv3 – mã nguồn Meta
- DINOv3 ViT-S/16 – Hugging Face
