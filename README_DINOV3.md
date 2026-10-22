# THAY ĐỔI DINOv3 + LOSS — HƯỚNG DẪN KIỂM TRA CÂU 1

**Đường dẫn code để thầy kiểm tra trực tiếp:**

- [Hai DINOv3 pretrained độc lập: `model.py`](code/TFC/model.py)
- [Chuyển chuỗi tín hiệu thành ảnh: `dinov3_adapter.py`](code/TFC/dinov3_adapter.py)
- [Loss mới: `dinov3_loss.py`](code/TFC/dinov3_loss.py)
- [Hàm `loss.py` trỏ tới loss mới](code/TFC/loss.py)
- [Pre-training/Fine-tuning gọi loss mới: `trainer.py`](code/TFC/trainer.py)
- [Các trọng số cấu hình loss: `main.py`](code/TFC/main.py)
- [Bài kiểm thử loss/gradient: `test_dinov3_loss.py`](code/TFC/test_dinov3_loss.py)

## Cập nhật hàm loss cho hai DINOv3

Mã nguồn gốc có `NTXentLoss_poly` tính `L_t`, `L_f` trên đặc trưng backbone
`h_t`, `h_f`, trong khi hai backbone DINOv3 có thể bị đóng băng. Phiên bản
này chuyển loss sang **đầu ra projector** `z_t`, `z_f` để các thành phần loss
đều có gradient đối với tham số được huấn luyện. Hàm loss mới là
`DINOv3TFCLoss` được viết tại `code/TFC/dinov3_loss.py` và dùng thực tế
trong `model_pretrain()` và `model_finetune()`.

Ký hiệu `t`, `f` cho miền thời gian và tần số; dấu `'` chỉ mẫu tăng cường.
`NTXent(a,b)` là InfoNCE đối xứng có positive cùng chỉ số mẫu:

```
L_time  = NTXent(z_t, z_t')
L_freq  = NTXent(z_f, z_f')
L_cross = 1/4 * ( NTXent(z_t, z_f) + NTXent(z_t, z_f')
                + NTXent(z_t', z_f) + NTXent(z_t', z_f') )
L_pretrain = 0.2*L_time + 0.2*L_freq + 1.0*L_cross
L_finetune = CrossEntropy(classifier([z_t;z_f]), y) + 0.1*L_pretrain
```

Các hệ số và nhiệt độ có thể chỉnh bằng các tham số `--dinov3_loss_*`
trong `code/TFC/main.py`. Đây là **biến thể loss cho DINOv3-TF-C**,
không được trình bày như nguyên bản của bài báo TF-C. Mặc định backbone
DINOv3 được đóng băng; projector và classifier được huấn luyện.

**Kiểm thử trên CPU, không cần tải DINOv3:**

```bash
cd code/TFC
python test_dinov3_loss.py
python smoke_test_dinov3.py
```

Bài kiểm thử xác minh loss hữu hạn, positive alignment, và gradient của
bốn vector projector. Kiểm thử đơn vị không thay thế việc huấn luyện thực tế.
Các kết quả accuracy/epoch được báo cáo trước đây **không phải** kết quả
đo lại sau khi cập nhật loss; muốn báo cáo số mới phải huấn luyện lại.

---

F-C + DINOv3 – Thay thế hai Encoder của mô hình TF-C

1. Giới thiệu
Dự án được thực hiện dựa trên mã nguồn TF-C Pretraining. TF-C là mô hình học đặc trưng từ dữ liệu chuỗi thời gian thông qua hai nhánh: miền thời gian và miền tần số.
Trong bài tập này, em thay thế cả hai Transformer Encoder trong phiên bản mã nguồn TF-C bằng hai backbone DINOv3 ViT-S/16 pretrained độc lập. Vì DINOv3 nhận đầu vào là ảnh, em bổ sung bước chuyển tín hiệu EEG một chiều thành ảnh trước khi đưa vào mô hình.
Mục tiêu chính là thay backbone, điều chỉnh code để mô hình chạy được và kiểm tra kết quả, không phải chứng minh DINOv3 có độ chính xác cao hơn TF-C gốc.

2. Những thay đổi trong mã nguồn
2.1. Thay cả hai Encoder bằng DINOv3
File đã sửa: [`code/TFC/model.py`](code/TFC/model.py)
Ban đầu: Mã nguồn sử dụng transformer_encoder_t để xử lý nhánh thời gian và transformer_encoder_f để xử lý nhánh tần số.
Sau khi sửa:
- Thay transformer_encoder_t bằng encoder_t sử dụng mô hình DINOv3 ViT-S/16.
- Thay transformer_encoder_f bằng encoder_f sử dụng một mô hình DINOv3 ViT-S/16 riêng biệt.
- Cập nhật hàm forward() để chuyển tín hiệu thành hai ảnh tương ứng, đưa mỗi ảnh qua một backbone rồi lấy đặc trưng.
- Mỗi DINOv3 trích xuất vector đặc trưng 384 chiều. Hai projector chuyển các vector này về 128 chiều cho từng nhánh.
- Khi phân loại, hai vector 128 chiều được ghép thành vector 256 chiều để đưa vào bộ phân loại.
Như vậy, phần thay đổi chính là bộ trích xuất đặc trưng; mô hình vẫn giữ hướng tiếp cận thời gian – tần số và cơ chế học tương phản của TF-C trong phiên bản code sử dụng.
2.2. Chuyển tín hiệu EEG thành ảnh cho DINOv3
File tạo mới: [`code/TFC/dinov3_adapter.py`](code/TFC/dinov3_adapter.py)
EEG là chuỗi số một chiều, còn DINOv3 được huấn luyện trước trên ảnh. Vì vậy, em tạo bộ chuyển đổi đầu vào gồm hai cách biểu diễn:
- Ảnh miền thời gian: Kết hợp tín hiệu gốc, độ biến thiên và đường bao biên độ thành ba kênh ảnh.
- Ảnh miền tần số: Sử dụng STFT để tạo ảnh phổ thời gian – tần số, lấy giá trị theo thang log rồi chuyển thành ảnh ba kênh.
Cả hai ảnh được đưa về kích thước 224 × 224 và chuẩn hóa để phù hợp với DINOv3.
Đây là cách biểu diễn được xây dựng để thử nghiệm khả năng áp dụng DINOv3 cho EEG, không phải cách biểu diễn chính thức của TF-C gốc.
2.3. Điều chỉnh phần chuẩn bị dữ liệu
File đã sửa: [`code/TFC/dataloader.py`](code/TFC/dataloader.py)
- Điều chỉnh nhánh tần số để đưa tín hiệu dạng sóng vào bộ chuyển đổi ảnh phổ, thay vì trực tiếp dùng độ lớn FFT một chiều làm đầu vào Encoder.
- Với tăng cường dữ liệu miền tần số, thực hiện biến đổi FFT, thay đổi một số thành phần tần số rồi dùng iFFT để đưa về tín hiệu một chiều.
- Điều chỉnh tùy chọn lấy một phần dữ liệu (--subset) để kiểm tra chương trình với ít mẫu hơn khi chạy trên CPU, bao gồm tập test ở chế độ chạy thử.
2.4. Bổ sung các tham số chạy chương trình
File đã sửa: [`code/TFC/main.py`](code/TFC/main.py)
Tham số	Mục đích
--backbone_name	Chọn checkpoint DINOv3 pretrained.
--unfreeze_backbone	Cho phép cập nhật trọng số của backbone nếu cần; mặc định backbone được đóng băng.
--image_size	Chọn kích thước ảnh đưa vào mô hình.
--n_fft	Thiết lập tham số FFT cho ảnh phổ STFT.
--subset	Chỉ lấy một phần dữ liệu để chạy thử.

Em cũng đặt tên thư mục lưu checkpoint có hậu tố dinov3_both để phân biệt với mô hình Transformer cũ.
2.5. Điều chỉnh quá trình huấn luyện
File đã sửa: [`code/TFC/trainer.py`](code/TFC/trainer.py)
- Sửa bước xóa gradient trong quá trình Pre-training trước khi tính gradient mới.
- Sử dụng kích thước batch thực tế khi tính NT-Xent loss.
- Thay cách dùng kiểu dữ liệu NumPy đã cũ để tương thích với thư viện hiện tại.
- Điều chỉnh việc tải checkpoint với map_location để có thể chạy trên CPU.
Những thay đổi này giúp chương trình hoạt động ổn định hơn với backbone mới; không thay thế mục tiêu học tương phản đang được cài đặt trong mã nguồn TF-C.
2.6. Bổ sung thư viện và các file kiểm tra
- [`code/TFC/requirements-dinov3.txt`](code/TFC/requirements-dinov3.txt) (file mới): Khai báo những thư viện bổ sung cần thiết cho phiên bản DINOv3.
- [`code/TFC/smoke_test_dinov3.py`](code/TFC/smoke_test_dinov3.py) (file mới): Kiểm tra kích thước ảnh, đầu ra của hai nhánh và gradient của projector bằng backbone giả lập.
- [`code/TFC/test_real_dinov3.py`](code/TFC/test_real_dinov3.py) (file kiểm tra): Thử chạy ảnh EEG qua một backbone DINOv3 pretrained thật.
- [`code/TFC/test_two_real_dinov3.py`](code/TFC/test_two_real_dinov3.py) (file kiểm tra): Kiểm tra hai backbone DINOv3 thật hoạt động độc lập trong cùng mô hình.
Phạm vi thay đổi: Chủ yếu ở code/TFC/. Các mô hình so sánh trong code/baselines/ không được thay backbone.

3. Kết quả thực hiện
3.1. Kiểm tra hai backbone DINOv3
Em đã tải được trọng số DINOv3 ViT-S/16 từ Hugging Face và chạy forward thành công với EEG thật:
Thành phần	Kích thước đầu ra
Time Encoder	384 chiều
Time Projector	128 chiều
Frequency Encoder	384 chiều
Frequency Projector	128 chiều
Ghép hai vector projector	256 chiều


Kết quả kiểm tra xác nhận hai Encoder là hai mô hình độc lập và các vector đầu ra không chứa giá trị NaN/Inf.
3.2. Pre-training trên SleepEEG
Chạy thử trên 1.280 mẫu SleepEEG với tùy chọn --subset:
Nội dung	Kết quả
Số epoch	20
Training Loss – epoch 1	5.2391
Training Loss – epoch 20	2.0547
Thời gian chạy	23 phút 56 giây
Lưu checkpoint	Thành công


Checkpoint được lưu trong thư mục code/experiments_logs/SleepEEG_2_Epilepsy/run1/pre_train_seed_42_dinov3_both/saved_models/ 
Nhận xét: Loss giảm qua 20 epoch, cho thấy quá trình Pre-training đã chạy được với hai DINOv3. Loss giảm chưa đủ để kết luận mô hình phân loại tốt hơn.
3.3. Fine-tuning và Testing trên Epilepsy
Em tiếp tục sử dụng checkpoint Pre-training để chạy 20 epoch Fine-tuning và Testing. Trong chế độ --subset, tập Fine-tuning gồm 60 mẫu, tập Testing gồm 600 mẫu. Thời gian chạy khoảng 1 giờ 28 phút 46 giây.
Bộ phân loại	Accuracy	F1-score	AUROC	AUPRC
MLP	18,50%	15,61%	42,54%	50,38%
KNN	85,50%	64,16%	74,68%	73,29%

4. Cách chạy
Mở Terminal tại thư mục code/TFC, sau đó cài thư viện:
pip install -r requirements-dinov3.txt
Checkpoint DINOv3 cần được cấp quyền truy cập trên Hugging Face. Sau khi được chấp thuận, đăng nhập:
hf auth login
Kiểm tra cấu trúc mô hình:
python smoke_test_dinov3.py
Chạy Pre-training với dữ liệu SleepEEG:
python main.py --training_mode pre_train --pretrain_dataset SleepEEG --target_dataset Epilepsy --device cpu --subset
Chạy Fine-tuning và Testing trên Epilepsy bằng checkpoint vừa tạo:
python main.py --training_mode fine_tune_test --pretrain_dataset SleepEEG --target_dataset Epilepsy --device cpu --subset
Dữ liệu: Cần chuẩn bị các file .pt của SleepEEG và Epilepsy theo cấu trúc mà bộ nạp dữ liệu của dự án yêu cầu. Tùy chọn --subset chỉ dùng để chạy thử nhanh. Mã nguồn trên GitHub không kèm dữ liệu lớn, checkpoint huấn luyện hoặc trọng số DINOv3.
5. Kết luận
Em đã thay thành công cả hai Encoder của TF-C bằng DINOv3, bổ sung cách chuyển EEG thành ảnh và chạy được toàn bộ quá trình Pre-training → Fine-tuning → Testing với dữ liệu chạy thử.
Kết quả hiện tại xác nhận mã nguồn đã thay backbone có thể hoạt động. Để đánh giá chất lượng mô hình một cách đầy đủ, cần chạy trên tập test hoàn chỉnh và so sánh với TF-C gốc trong cùng điều kiện.
Tài liệu tham khảo
- TF-C Pretraining – mã nguồn gốc
- DINOv3 – mã nguồn Meta
- DINOv3 ViT-S/16 – Hugging Face