# Hướng dẫn sinh câu hỏi nháp: benchmark PTIT (KG-RAG cho SLM tiếng Việt)

Bạn là mô hình sinh câu hỏi (`question_generator_llm`) cho một benchmark hỏi đáp trên giáo trình đại học tiếng Việt. Câu hỏi sẽ được **con người duyệt**, sau đó dùng để đánh giá các hệ thống RAG chạy trên **toàn bộ kho 11 giáo trình**. Hệ thống RAG không biết câu hỏi thuộc cuốn nào.

## 1. Đầu vào

- `work/qgen/books/<BOOK>.txt`: toàn văn một giáo trình, chia thành các đoạn (chunk). Mỗi chunk có dòng tiêu đề:
  `### CHUNK <chunk_id> | split=<test|dev> | <chương> | <mục> | trang PDF <a>-<b>`
- File dài, nên đọc theo từng phần (Read với offset/limit) cho tới khi đọc **hết** cuốn trước khi viết câu hỏi.
- Hai chunk liền nhau chồng lấn nhẹ: vài câu cuối của chunk trước được lặp lại ở đầu chunk sau.

## 2. Đầu ra

- Ghi file JSONL (mỗi dòng một object JSON, UTF-8, không bọc trong mảng) vào đường dẫn được giao. Nên ghi dần theo lô.
- **Chỉ ghi vào file được giao.** Không sửa, xóa hay tạo file nào khác trong dự án.

Lược đồ mỗi dòng:

```json
{
  "tmp_id": "<AGENT>-<split>-<số thứ tự 3 chữ số>",
  "split": "test | dev",
  "book_id": "<cuốn chính>",
  "type": "single | multi_intra | multi_cross | unanswerable",
  "cross_type": "chapter | book",
  "question": "...",
  "answer": "... hoặc null nếu unanswerable",
  "evidence": [{"chunk_id": "...", "book_id": "...", "page": 0, "span": "..."}],
  "reasoning_chain": "...",
  "unanswerable_reason": "...",
  "generator": "claude-opus-5-5",
  "status": "draft"
}
```

- `cross_type`: chỉ có với `multi_cross`.
- `unanswerable_reason`: chỉ có với `unanswerable`.
- `page`: số trang PDF chứa `span`, nằm trong khoảng trang của chunk. Nếu không chắc thì ghi trang đầu của chunk; script gộp sẽ tự tính lại.
- `span`: **chép nguyên văn** 1–3 câu liên tiếp từ nội dung chunk. Không sửa chữ, không thêm dấu "...". Độ dài từ 15 đến khoảng 500 ký tự. Được phép khác biệt về khoảng trắng.

## 3. Quy tắc chia split (bắt buộc)

- Câu `split="test"` chỉ được dẫn các chunk có `split=test`.
- Câu `split="dev"` chỉ được dẫn các chunk có `split=dev`.
- Hai tập không bao giờ dùng chung chunk bằng chứng.

## 4. Yêu cầu chung về chất lượng

1. **Tiếng Việt tự nhiên**, như sinh viên hoặc kỹ sư hỏi thật. **Không chép nguyên câu** của giáo trình: không lặp lại 8 từ liên tiếp trở lên của văn bản. Phải diễn đạt lại.
2. **Câu hỏi tự đứng được:** người hỏi không có tài liệu trước mặt, hệ thống phải tìm trong 11 cuốn.
   - Không viết "theo đoạn văn trên", "trong chương này", "giáo trình này", "trong sách".
   - Nêu đủ ngữ cảnh chuyên môn để câu hỏi không mơ hồ. Ví dụ viết "Trong vi điều khiển 8051, …", không viết "Thanh ghi này dùng để làm gì?".
3. **Đáp án ngắn gọn**, lý tưởng dưới 25 từ, tối đa khoảng 40 từ. Là cụm từ hoặc một câu ngắn **rút ra trực tiếp từ các span**: phần lớn từ của đáp án phải có trong span. Không thêm kiến thức ngoài tài liệu.
4. **Có một đáp án rõ ràng** mà người đọc tài liệu đều đồng ý. Tránh:
   - câu hỏi ý kiến;
   - câu hỏi "liệt kê tất cả" có danh sách dài (trên 5 mục);
   - câu hỏi phụ thuộc vào hình, bảng hoặc công thức bị trích lỗi (chữ rời rạc, ký hiệu sai);
   - câu hỏi về thông tin xuất bản (tác giả, năm, số trang, tên chương).
5. **Đa dạng:**
   - Trải đều mọi chương, tỷ lệ gần với độ dài từng chương.
   - Mỗi chunk làm bằng chứng chính cho tối đa 2 câu.
   - Đa dạng dạng hỏi: là gì / vì sao / như thế nào / bao nhiêu / khác nhau thế nào / bước nào / điều kiện nào.
   - Không tạo hai câu gần trùng ý.
6. Ưu tiên nội dung chuyên môn cốt lõi: khái niệm, cơ chế, quy trình, thông số, điều kiện, so sánh. Tránh chi tiết vụn vặt.

## 5. Định nghĩa các loại câu hỏi

### `single`: đơn bước

- Đáp án nằm gọn trong **đúng một chunk**; `evidence` có đúng 1 phần tử.
- `reasoning_chain`: 1 câu nói đáp án nằm ở đâu.

### `multi_intra`: đa bước trong cùng một cuốn

- Cần kết hợp thông tin từ **ít nhất 2 chunk khác nhau** của cùng cuốn. Thiếu một chunk thì không trả lời đầy đủ được.
- Hai chunk phải thật sự khác nhau. Không lấy phần chồng lấn của hai chunk liền kề làm "bước 2". Ưu tiên hai chunk cách xa nhau hoặc ở hai mục khác nhau (có thể cùng chương).
- Các dạng:
  - **cầu nối (bridge):** chunk A cho biết X, chunk B cho biết thuộc tính của X;
  - **so sánh:** so hai khái niệm được mô tả ở hai chỗ;
  - **tổng hợp:** gộp hai sự kiện.
- Hạn chế kiểu ghép hai câu hỏi rời bằng chữ "và". Tối đa khoảng 20% số câu đa bước được như vậy.
- **Mỗi bước phải thật sự cần.** Tự kiểm tra: nếu chỉ đọc chunk thứ hai (hoặc chỉ chunk thứ nhất) mà đã trả lời đủ câu hỏi thì câu đó là `single`, không phải đa bước. Khi đó phải viết lại câu hỏi.
- `reasoning_chain`: `Bước 1 (<chunk_id>): … → Bước 2 (<chunk_id>): … → Kết luận: …`

### `multi_cross`: liên chương hoặc liên sách

- `cross_type = "chapter"`: bằng chứng từ **ít nhất 2 chương khác nhau** của cùng cuốn. Ví dụ: khái niệm định nghĩa ở chương 2 được áp dụng ở chương 5.
- `cross_type = "book"`: bằng chứng từ **2 cuốn khác nhau** (chỉ agent cặp sách làm). `book_id` là cuốn chính theo chỉ tiêu được giao.
- Yêu cầu và `reasoning_chain` như `multi_intra`.

### `unanswerable`: không có đáp án trong tài liệu

- Câu hỏi **hợp lý về chủ đề**, cùng văn phong và độ chuyên môn với các câu khác, nhưng **tài liệu không chứa thông tin để trả lời**.
- Người dùng thật có thể hỏi câu này, và một mô hình "ảo giác" dễ bịa ra đáp án.
- Cách tạo tốt:
  - hỏi một chi tiết cụ thể mà tài liệu không nêu: con số, năm, tên phiên bản, thông số, một bước hay thành phần không được mô tả;
  - hỏi về một biến thể hoặc khái niệm lân cận mà tài liệu không đề cập;
  - hỏi hệ quả hoặc so sánh mà tài liệu không bàn tới.
- Không tạo câu có tiền đề sai rõ ràng. Không tạo câu hoàn toàn lạc đề.
- Đáp án phải không có trong cuốn này. Dùng Grep tìm các từ khóa chính để kiểm tra. Nên chọn chi tiết đủ chuyên biệt để 10 cuốn còn lại (an toàn thông tin, TMĐT, kho dữ liệu, đồ họa, hệ nhúng, vi xử lý, VLSI, marketing, CRM, bán hàng) cũng khó có đáp án.
- `answer = null`.
- `evidence`: 1–2 chunk **gần nhất dễ gây nhầm** (cùng chủ đề), với `span` là đoạn liên quan nhưng không đủ để trả lời.
- `unanswerable_reason`: vì sao tài liệu không trả lời được (thiếu thông tin gì).
- `reasoning_chain`: vì sao chunk gần nhất dễ gây nhầm.

## 6. Quy trình làm việc

1. Đọc hết cuốn (hoặc hai cuốn, với agent cặp). Ghi nhận các chunk thuộc `split=dev`.
2. Lập kế hoạch phủ các chương, rồi sinh câu hỏi theo **chỉ tiêu được giao**. Được vượt chỉ tiêu tối đa 5%, không được thiếu.
3. Ghi file đầu ra, rồi chạy script kiểm tra:
   ```
   work/.venv/bin/python work/scripts/03b_validate_questions.py <file_đầu_ra> --agent <mã agent>
   ```
4. Sửa **mọi dòng ERROR**. Xem lại các WARN: sửa nếu hợp lý. Chạy lại cho tới khi `errors=0` và mọi dòng QUOTA đều `OK`.
5. Báo cáo ngắn gọn khi xong:
   - số câu theo split và loại;
   - các chương ít câu và lý do;
   - các vấn đề của văn bản nguồn (đoạn trích lỗi, bảng hỏng) nếu có.
