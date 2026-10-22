TF-C + DINOv3 — Học biểu diễn chuỗi thời gian bằng tương phản thời gian–tần số
Trạng thái: Đã chạy thử hai encoder DINOv3 pretrained thực tế; hoàn thành 20 epoch Pre-training, 20 epoch Fine-tuning và Testing trên tập con SleepEEG → Epilepsy.
Phạm vi: Dự án nghiên cứu AI chạy cục bộ bằng Python. Repository này chưa triển khai website, ứng dụng người dùng hoặc REST API.

1. Tổng quan dự án
Dự án thử nghiệm thay thế hai Transformer Encoder trong mô hình TF-C (Time-Frequency Consistency) bằng hai backbone DINOv3 ViT-S/16 pretrained độc lập, một backbone xử lý miền thời gian và một backbone xử lý miền tần số.

Vì DINOv3 được thiết kế để nhận ảnh, tín hiệu EEG một chiều được biến đổi thành ảnh đặc trưng trước khi đưa vào encoder. Các đặc trưng được kết hợp để thực hiện bài toán phân loại Epilepsy.

Mục tiêu:
- Thay hai encoder TF-C bằng hai DINOv3 pretrained độc lập.
- Chuyển tín hiệu EEG thành ảnh thời gian và ảnh phổ tần số.
- Cài đặt và sử dụng Loss Function mới trong Pre-training và Fine-tuning.
- Kiểm thử cấu trúc mô hình, gradient, huấn luyện và đánh giá trên tập dữ liệu thật.
Các file chính để kiểm tra thay đổi:
Thành phần	File mã nguồn
Hai DINOv3 encoder, projector và classifier	[`code/TFC/model.py`](code/TFC/model.py)
Chuyển tín hiệu thành ảnh	[`code/TFC/dinov3_adapter.py`](code/TFC/dinov3_adapter.py)
Loss Function mới	[`code/TFC/dinov3_loss.py`](code/TFC/dinov3_loss.py)
Module liên kết Loss	[`code/TFC/loss.py`](code/TFC/loss.py)
Sử dụng Loss khi Pre-training/Fine-tuning	[`code/TFC/trainer.py`](code/TFC/trainer.py)
Tham số cấu hình và chương trình chính	[`code/TFC/main.py`](code/TFC/main.py)
Kiểm thử Loss và gradient	[`code/TFC/test_dinov3_loss.py`](code/TFC/test_dinov3_loss.py)
Kiểm thử encoder với backbone giả lập	[`code/TFC/smoke_test_dinov3.py`](code/TFC/smoke_test_dinov3.py)
Kiểm thử hai encoder pretrained thật	[`code/TFC/test_two_real_dinov3.py`](code/TFC/test_two_real_dinov3.py)


2. Kiến trúc hệ thống (System Overview)
                 EEG: [B, 1, 178]
                        |
               SignalImageAdapter
                 /            \
                v              v
         Ảnh thời gian     Ảnh phổ STFT
           [B,3,224,224]   [B,3,224,224]
                |              |
         DINOv3 Encoder T  DINOv3 Encoder F
                |              |
          384 đặc trưng   384 đặc trưng
                |              |
           Projector T     Projector F
                |              |
          z_t: 128 chiều  z_f: 128 chiều
                 \            /
                  \          /
          DINOv3TFCLoss (Pre-training)
                        |
               Ghép z_t và z_f
                    256 chiều
                        |
                  MLP hoặc KNN
                        |
                Nhãn Epilepsy
- Input: tensor EEG kích thước [B, 1, 178]; B là số mẫu trong batch.
- Ảnh đầu vào encoder: mỗi ảnh có 3 kênh và kích thước 224 × 224.
- Encoder: mỗi DINOv3 ViT-S/16 xuất vector 384 chiều; hai mô hình được khởi tạo độc lập.
- Projector: mỗi nhánh ánh xạ từ 384 xuống 128 chiều.
- Bộ phân loại: nhận vector ghép 256 chiều, sử dụng MLP hoặc KNN trong thí nghiệm đánh giá.
- Môi trường: Python chạy local, không có web frontend, backend hoặc cơ sở dữ liệu trong repository thử nghiệm này.
2.1. Hàm mất mát mới (DINOv3TFCLoss)
Hàm loss sử dụng đầu ra projector có thể huấn luyện (z_t, z_f) thay vì chỉ tối ưu trên đặc trưng backbone đóng băng. Ký hiệu _aug là mẫu đã tăng cường dữ liệu.
L_time  = NT-Xent(z_t, z_t_aug)
L_freq  = NT-Xent(z_f, z_f_aug)
L_cross = mean(
    NT-Xent(z_t,     z_f),
    NT-Xent(z_t,     z_f_aug),
    NT-Xent(z_t_aug, z_f),
    NT-Xent(z_t_aug, z_f_aug)
)

L_pretrain = 0.2 * L_time + 0.2 * L_freq + 1.0 * L_cross
L_finetune = CrossEntropy + 0.1 * L_pretrain
Temperature = 0.2
Đây là biến thể Loss dành cho thử nghiệm TF-C + DINOv3, không phải công thức nguyên bản được công bố trong bài báo TF-C. Mặc định hai backbone DINOv3 được đóng băng; các projector và classifier được huấn luyện.
3. Môi trường (Environment)
Thành phần	Yêu cầu / thông tin
Hệ điều hành đã chạy	Windows, PowerShell trong VS Code
Python	Python 3, sử dụng .venv; kiểm tra bản cụ thể bằng python --version
PyTorch	torch>=2.2
Transformers	transformers>=4.56.0
Hugging Face Hub	huggingface_hub>=0.34
NumPy	numpy>=1.24
scikit-learn	scikit-learn>=1.3
Backbone	facebook/dinov3-vits16-pretrain-lvd1689m
Thiết bị đã thử nghiệm	CPU; tốc độ tùy cấu hình máy


Dữ liệu cần có trước khi chạy:
datasets/
├── SleepEEG/
│   └── train.pt
└── Epilepsy/
    ├── train.pt
    └── test.pt
Các tệp dữ liệu .pt phải có cấu trúc tương thích với code/TFC/dataloader.py. Nếu clone repo mà thiếu dữ liệu, cần chuẩn bị bộ SleepEEG và Epilepsy theo hướng dẫn mã nguồn TF-C gốc trước khi chạy huấn luyện.
4. Cài đặt trên Windows (Setup)
Bước 1 — Clone repository và mở thư mục dự án:
git clone https://github.com/vuvandat109/TF-C-DINOv3.git
cd TF-C-DINOv3
Bước 2 — Tạo và kích hoạt môi trường ảo:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
Nếu PowerShell không cho chạy script kích hoạt, thực hiện:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
Kết quả mong đợi: Terminal có tiền tố (.venv).

Bước 3 — Cài đặt thư viện:
python -m pip install -r .\code\TFC\requirements-dinov3.txt
python --version
python -c "import torch, transformers; print('PyTorch:', torch.__version__); print('Transformers:', transformers.__version__)"
Bước 4 — Kiểm tra dữ liệu ở thư mục gốc:
Test-Path .\datasets\SleepEEG\train.pt
Test-Path .\datasets\Epilepsy\train.pt
Test-Path .\datasets\Epilepsy\test.pt
Kết quả mong đợi: cả ba lệnh đều trả về True.

Bước 5 — Cấp quyền truy cập trọng số DINOv3:

Mở DINOv3 ViT-S/16 trên Hugging Face, hoàn tất điều kiện truy cập của nhà cung cấp (nếu được yêu cầu), sau đó đăng nhập:
hf auth login
Không đưa token Hugging Face lên GitHub.
5. Chạy và kiểm tra (Run & Verify)
Các lệnh dưới đây cần chạy từ thư mục code/TFC:
cd .\code\TFC
5.1. Kiểm thử Loss Function
python test_dinov3_loss.py
Kết quả kỳ vọng:
PASS: DINOv3-TF-C loss, positive pairing and gradients in all four projected views
Bài kiểm thử dùng tensor giả lập để kiểm tra tính hữu hạn của loss, các cặp positive và gradient của projector; không chứng minh hiệu năng mô hình pretrained.
5.2. Kiểm thử cấu trúc hai encoder (giả lập)
python smoke_test_dinov3.py
Kết quả kỳ vọng:
PASS: both DINOv3 encoder slots replaced, adapters, output shapes and projector gradients
5.3. Kiểm thử hai encoder DINOv3 pretrained thật
python test_two_real_dinov3.py
Kết quả đã ghi nhận:
EEG input: torch.Size([2, 1, 178])
Independent encoders: True
Time features: torch.Size([2, 384])
Time projector: torch.Size([2, 128])
Frequency features: torch.Size([2, 384])
Frequency projector: torch.Size([2, 128])
Combined: torch.Size([2, 256])
Finite: True
SUCCESS: Both real DINOv3 encoders passed!
5.4. Pre-training với Loss Function mới
python main.py --training_mode pre_train --pretrain_dataset SleepEEG --target_dataset Epilepsy --device cpu --subset
Dấu hiệu chạy đúng:
Mode: pre_train
Pretraining on source dataset
DINOv3-TF-C pretrain | total=... | L_time=... | L_freq=... | L_cross=...
Pre-training Epoch : 1 | Train Loss : ...
Trong cấu hình đã kiểm thử: 20 epoch, learning rate 3e-4, batch size 8, seed 42; cờ --subset lấy 80 mẫu SleepEEG để thử nhanh. Lưu ý: main.py hiện tự động ưu tiên CUDA khi khả dụng nên --device cpu không đảm bảo cưỡng chế CPU nếu máy có GPU CUDA.

Vị trí checkpoint Pre-training (tính từ thư mục gốc repository):
code/experiments_logs/SleepEEG_2_Epilepsy/run1/
└── pre_train_seed_42_dinov3_both/
    └── saved_models/
        └── ckp_last.pt
5.5. Fine-tuning và Testing
Chỉ chạy sau khi Pre-training đã lưu checkpoint mới.
python main.py --training_mode fine_tune_test --pretrain_dataset SleepEEG --target_dataset Epilepsy --device cpu --subset
Cấu hình --subset đã kiểm thử dùng 60 mẫu Fine-tuning và 600 mẫu Testing.

Dấu hiệu chạy đúng:
Mode: fine_tune_test
Fine-tune on Fine-tuning set
Epoch : 1
Finetune: loss = ... | Acc=... | F1=...
MLP Testing: Acc=... | F1=...
KNN Testing: Acc=... | F1=...
...
Best Testing Performance: ...
Best KNN F1 ...
Training is Done!
Vị trí kết quả (theo đường dẫn trong code):
File / thư mục	Chức năng
code/experiments_logs/SleepEEG_2_Epilepsy/...	Log từng lần chạy
code/experiments_logs/.../saved_models/ckp_last.pt	Checkpoint Pre-training
code/TFC/experiments_logs/finetunemodel/sleepedf2eplipsy_model.pt	Checkpoint Fine-tuning
code/TFC/experiments_logs/finetunemodel/sleepedf2eplipsy_classifier.pt	Checkpoint classifier MLP


Các giá trị MLP/KNN được in ra Terminal; phiên bản hiện tại chưa tự động lưu bảng metric đầy đủ thành CSV.
6. Kết quả thực nghiệm (Experimental Results)
6.1. Kết quả kiểm thử chức năng
Phép kiểm thử	Kết quả
Loss Function và gradient (test_dinov3_loss.py)	PASS
Kiến trúc encoder giả lập (smoke_test_dinov3.py)	PASS
Hai encoder DINOv3 pretrained thật (test_two_real_dinov3.py)	SUCCESS


6.2. Pre-training — 20 epoch
Chỉ số	Giá trị
Bộ dữ liệu	SleepEEG (--subset: 80 mẫu)
Epoch hoàn thành	20/20
Loss epoch 1	3.2115
Loss epoch 20	0.2784
L_time epoch 20	0.1390
L_freq epoch 20	0.1775
L_cross epoch 20	0.2151
Thời gian chạy	00:23:57.778917
Checkpoint	ckp_last.pt — đã kiểm tra tồn tại


Loss giảm từ 3.2115 xuống 0.2784, cho thấy hàm mất mát mới đã được sử dụng trong quá trình tối ưu. Loss giảm không đồng nghĩa với Accuracy cao trên tập kiểm thử.
6.3. Fine-tuning và Testing — 20 epoch
Thí nghiệm dùng Epilepsy, 60 mẫu Fine-tuning và 600 mẫu Testing với --subset.
Metric	MLP Testing (Best Testing Performance)	KNN Testing (epoch có F1 tốt nhất)
Accuracy	18.50%	90.33%
Precision	9.25%	94.70%
Recall	50.00%	73.87%
F1-score	15.61%	79.52%
AUROC	67.83%	91.21%
AUPRC	72.78%	89.79%


Nhận xét kết quả:
- KNN đạt Accuracy cao trong lần chạy thử, nhưng kết quả trên tập con không đủ để kết luận mô hình tổng quát hóa tốt.
- MLP đạt Accuracy thấp và gần như không cải thiện từ epoch 2 đến epoch 20. Cần kiểm tra lại quá trình Fine-tuning, phân bố nhãn và logic tải checkpoint trước khi đánh giá cuối cùng.
- Log xuất hiện UndefinedMetricWarning vì có lớp không được dự đoán. Đây là giới hạn thực tế cần ghi nhận, không nên bỏ qua.
- Không thể khẳng định biến thể DINOv3 vượt TF-C gốc nếu chưa đánh giá đối chứng cùng tập dữ liệu và cùng điều kiện.
7. Mô hình AI và giới hạn (AI Model)
Thuộc tính	Nội dung
Mô hình	TF-C + hai DINOv3 ViT-S/16
Backbone pretrained	facebook/dinov3-vits16-pretrain-lvd1689m
Loss Function	DINOv3TFCLoss (Time, Frequency, Cross-domain NT-Xent)
Input thử nghiệm	EEG [B, 1, 178]
Đầu ra encoder	Hai vector 384 chiều
Đầu ra projector	Hai vector 128 chiều
Đầu ra phân loại	Logits hai lớp Epilepsy
Cấu hình thực nghiệm	seed 42, --subset, 20 epoch
Triển khai	Chạy local bằng Python; chưa có REST API hoặc website


Các giới hạn cần lưu ý:
- DINOv3 được pretrained trên ảnh. Biến đổi EEG thành ảnh là phương pháp thử nghiệm, chưa chứng minh là tối ưu.
- dataloader.py hiện lấy kênh tín hiệu đầu tiên trong thí nghiệm này.
- Hai backbone DINOv3 được đóng băng theo mặc định; projector và classifier được cập nhật.
- Chưa thực hiện đánh giá đầy đủ trên toàn bộ dữ liệu, nhiều seed hoặc so sánh đối chứng có kiểm soát với TF-C nguyên bản.
- Không có chức năng API dự đoán, confidence threshold và fallback dành cho người dùng cuối trong repository này.
- Kết quả trong README không phải kết quả huấn luyện trên bộ dữ liệu cử chỉ iPhone 12.
8. Cách sử dụng (Usage)
Dự án hiện hướng đến người nghiên cứu và lập trình viên. Sau khi cài đặt, có thể:
1. Chạy các bài kiểm thử tại mục 5.1–5.3.
2. Chạy đúng quy trình Pre-training → Fine-tuning → Testing tại mục 5.4–5.5.
3. Đọc log, theo dõi Loss, Accuracy, Precision, Recall, F1, AUROC và AUPRC trên Terminal.
4. Sử dụng checkpoint cho thí nghiệm tiếp theo với cùng kiến trúc và cấu hình tương thích.
Kiểm tra checkpoint Pre-training từ code/TFC bằng PowerShell:
Test-Path "..\experiments_logs\SleepEEG_2_Epilepsy\run1\pre_train_seed_42_dinov3_both\saved_models\ckp_last.pt"
Kết quả kỳ vọng sau khi Pre-training: True.
9. Xử lý lỗi thường gặp (Troubleshooting)
Hiện tượng	Nguyên nhân thường gặp	Cách xử lý
ModuleNotFoundError: transformers	Chưa cài thư viện trong .venv	Kích hoạt .venv, cài lại requirements-dinov3.txt
Lỗi 401 / 403 khi tải DINOv3	Chưa được cấp quyền truy cập hoặc chưa đăng nhập	Mở trang DINOv3 trên Hugging Face, chấp thuận điều kiện, chạy hf auth login
FileNotFoundError: train.pt	Thiếu dữ liệu hoặc chạy sai thư mục	Kiểm tra ba tệp .pt; chạy main.py từ code/TFC
Không tìm thấy ckp_last.pt	Chưa Pre-training hoặc tên thư mục thí nghiệm không khớp	Chạy Pre-training trước, giữ nguyên dataset/seed/run description
Epoch chạy lâu	DINOv3 và STFT sử dụng CPU/RAM đáng kể	Theo dõi Task Manager; dùng --subset để thử; cân nhắc GPU nếu có
UndefinedMetricWarning	Có nhãn không được classifier dự đoán	Kiểm tra dự đoán, phân bố nhãn và confusion matrix
MLP Accuracy thấp và không cải thiện	Mô hình chưa hội tụ hoặc luồng huấn luyện/đánh giá có vấn đề	Kiểm tra trainer.py, checkpoint và optimizer; chưa kết luận MLP tốt
can't open file ...main.py	Terminal đang ở sai thư mục	Chạy cd code\TFC từ thư mục gốc trước khi gọi python main.py


10. Tái lập thí nghiệm và tài liệu tham khảo
Để người khác tái lập, cần giữ đúng mã nguồn, phiên bản dữ liệu và chia tập, cấu hình, trọng số pretrained, seed và môi trường. Cấu hình đã chạy: seed 42, learning rate 3e-4, batch size Pre-training 8, số epoch 20, Loss weights 0.2 / 0.2 / 1.0, hệ số SSL khi Fine-tuning 0.1.

Để ghi lại phiên bản thư viện của máy:
python -m pip freeze > environment_versions.txt
Nguồn tham khảo:
- TF-C — Self-Supervised Contrastive Pre-Training for Time Series via Time-Frequency Consistency (NeurIPS 2022).
- Website dự án TF-C.
- DINOv3 ViT-S/16 pretrained trên Hugging Face.
- Repository TF-C + DINOv3 của bài thực nghiệm.
