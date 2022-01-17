# These-Images-Do-Not-Exist

## Clone code và cài đặt các thư viện cần thiết
```bash
git clone https://github.com/louisdo/These-Image-Do-Not-Exist.git
pip install -r This-Galaxy-Does-Not-Exist/requirements.txt
```

## Thu thập và xử lý dữ liệu
Mã nguồn cho việc thu thập và xử lý dữ liệu nằm trong thư mục **data_retrieving_and_processing**. Chi tiết về các file trong thư mục này có thể tìm thấy trong báo cáo dự án hoặc người dùng cũng có thể đọc từng file để hiểu rõ.

## Huấn luyện mô hình
Trước tiên cần thay đổi các cấu hình trong file **config.py** sao cho phù hợp. Sau khi đã thay đổi cấu hình, chúng ta có thể huấn luyện mô hình bằng lệnh sau
```bash
python train.py
```

Ngoài cách trên, người dùng có thể sử dụng notebook **Train.ipynb** để huấn luyện mô hình.


## Đánh giá mô hình đã huấn luyện
Người dùng sử dụng notebook **Evaluation.ipynb** để đánh giá mô hình đã huấn luyện. Trong dự án này, mô hình được đánh giá bằng chỉ số FID. Một lần nữa, chi tiết về cách đánh giá có thể được tìm thấy trong báo cáo môn học
