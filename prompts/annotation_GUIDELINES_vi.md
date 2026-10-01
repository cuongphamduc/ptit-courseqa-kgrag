# Hướng dẫn gán nhãn: Benchmark hỏi đáp PTIT (vòng 1)

## 0. Mục đích và cách làm việc

Bộ câu hỏi nháp do một LLM sinh tự động (`claude-opus-5-5`) từ 11 giáo trình PTIT. Nhiệm vụ của bạn là quyết định mỗi câu **giữ nguyên**, **giữ sau khi sửa**, hay **loại**, để có một benchmark sạch dùng đánh giá hệ thống RAG.

- **File**
  - `annotation/round1/annotation_A.xlsx`: người gán nhãn A, gồm toàn bộ câu.
  - `annotation/round1/annotation_B.xlsx`: người gán nhãn B, gồm 20% câu test chọn ngẫu nhiên, dùng để đo độ đồng thuận.
  - **Hai người làm độc lập, không trao đổi** cho tới khi xong.
- **Sheet `questions`:** chỉ điền các cột nền vàng. Không sửa cột nền xám, không xóa hay đổi thứ tự dòng.
- **Sheet `chunks`:** toàn văn các đoạn bằng chứng. Cột `pdf_files` chỉ tới giáo trình gốc trong `dataset/PTIT/`.
- **Số trang** là **số trang PDF** (số trang trong trình xem PDF), không phải số in ở chân trang.
- **Chỉ dựa vào giáo trình.** Không dùng kiến thức bên ngoài để coi một đáp án là đúng. Nếu giáo trình viết sai về chuyên môn, vẫn đánh giá theo giáo trình và ghi vào `note`.

## 1. Các cột cần điền

| Cột | Giá trị | Ý nghĩa |
|---|---|---|
| `label` | `accept` / `edit` / `reject` | Quyết định cuối cho câu hỏi (bắt buộc với mọi dòng) |
| `edited_question` | văn bản | Chỉ điền khi `label=edit` và cần sửa câu hỏi |
| `edited_answer` | văn bản | Chỉ điền khi `label=edit` và cần sửa đáp án. Với `unanswerable` thì để trống |
| `type_correct` | `yes` / `no` | Loại câu hỏi (`type`, `cross_type`) có đúng không |
| `evidence_correct` | `yes` / `no` | Bằng chứng (chunk, span, trang) có đúng và đủ để trả lời không |
| `note` | văn bản | Lý do reject, loại đúng nếu `type_correct=no`, hoặc nhận xét khác |

## 2. Định nghĩa nhãn `label`

### `accept`: giữ nguyên

Chọn `accept` khi thỏa **tất cả** các tiêu chí sau:

1. **Câu hỏi**
   - Viết bằng tiếng Việt tự nhiên, rõ nghĩa, không mơ hồ.
   - Tự đứng được: người đọc hiểu được mà không cần biết đang nói về cuốn sách hay đoạn văn nào.
   - Không chép nguyên câu của giáo trình.
2. **Đáp án**
   - **Đúng theo giáo trình**, **đầy đủ** với những gì câu hỏi yêu cầu, và **ngắn gọn**.
   - Không chứa thông tin ngoài tài liệu.
3. **Bằng chứng**
   - Span đúng là nơi chứa đáp án (hoặc chứa thông tin gây nhầm, với `unanswerable`).
   - Số trang đúng, sai lệch cho phép là ±1 trang.
4. **Loại câu hỏi đúng** theo định nghĩa ở mục 3.
5. Câu hỏi có ích để đánh giá, không quá tầm thường. Ví dụ tầm thường: "Chương 2 tên là gì?".

### `edit`: giữ lại sau khi sửa

Chọn `edit` khi câu hỏi có giá trị nhưng cần **sửa nhỏ** mới đạt chuẩn `accept`:

- Diễn đạt vụng, lỗi chính tả, hoặc thiếu ngữ cảnh làm câu hỏi mơ hồ → ghi lại vào `edited_question`.
- Đáp án thiếu, thừa hoặc chưa chính xác nhưng sửa được từ chính bằng chứng → ghi vào `edited_answer`.
- Câu hỏi hỏi nhiều hơn một ý mà đáp án chỉ trả lời một ý → thu hẹp câu hỏi hoặc bổ sung đáp án.
- Loại câu sai nhưng câu hỏi vẫn tốt, ví dụ ghi `multi_intra` nhưng thực chất là `single` → giữ `edit`, ghi `type_correct=no` và loại đúng vào `note`.

**Không** dùng `edit` để viết một câu hỏi hoàn toàn mới. Nếu phải thay đổi quá nửa nội dung thì chọn `reject`.

### `reject`: loại

Chọn `reject` khi có **ít nhất một** lỗi sau:

- Đáp án sai theo giáo trình và không sửa được từ bằng chứng.
- Câu hỏi mơ hồ, có nhiều đáp án đều hợp lý, hoặc đáp án chỉ là ý kiến chủ quan.
- Câu hỏi dựa vào nội dung bị trích lỗi (công thức, hình, bảng hỏng) nên không kiểm chứng được.
- Câu hỏi quá tầm thường, hoặc hỏi về thông tin xuất bản.
- Câu gần trùng ý với một câu đã chấp nhận trước đó. Ghi ID của câu kia vào `note`.
- Câu `unanswerable` nhưng thực ra giáo trình (bất kỳ cuốn nào trong 11 cuốn) **có** đáp án, hoặc câu hỏi lạc đề, hoặc có tiền đề sai rõ ràng.
- Câu đa bước mà hai bước ghép gượng ép, không tự nhiên.

Ghi ngắn gọn lý do reject vào `note`.

## 3. Định nghĩa loại câu hỏi (dùng cho `type_correct`)

| Loại | Điều kiện đúng |
|---|---|
| `single` | Đáp án nằm gọn trong **một** đoạn (chunk). |
| `multi_intra` | Phải kết hợp **từ 2 đoạn trở lên trong cùng một cuốn**. Thiếu một đoạn thì không trả lời đủ. |
| `multi_cross` + `cross_type=chapter` | Như `multi_intra`, nhưng các đoạn thuộc **ít nhất hai chương khác nhau** của cùng cuốn. |
| `multi_cross` + `cross_type=book` | Các đoạn thuộc **hai cuốn khác nhau**, và cần cả hai cuốn mới trả lời đủ. |
| `unanswerable` | Câu hỏi hợp lý về chủ đề, nhưng **không cuốn nào trong 11 giáo trình** có đủ thông tin để trả lời. |

Mẹo kiểm tra câu đa bước: che một đoạn bằng chứng đi. Nếu vẫn trả lời đủ câu hỏi chỉ với đoạn còn lại thì `type_correct=no` (thực chất là `single`).

## 4. Đánh giá bằng chứng (`evidence_correct`)

- **`yes`** khi các span được dẫn **đủ để suy ra đáp án**, mỗi span thuộc đúng chunk, và trang PDF đúng (±1 trang).
- **`no`** khi thiếu một bước, span không chứa thông tin cần thiết, hoặc sai trang quá 1 trang. Ghi chỗ đúng vào `note` nếu tìm được.
- **Với `unanswerable`:** `yes` khi (a) giáo trình thật sự không có đáp án **và** (b) chunk được dẫn đúng là đoạn gần nhất dễ gây nhầm. Nếu tìm thấy đáp án ở bất kỳ đâu, chọn `evidence_correct=no`, `label=reject`, và ghi vị trí đáp án (cuốn, trang) vào `note`.
- **Cột `auto_check`** (chỉ với `unanswerable`) là kết quả kiểm tra tự động trên cả 11 giáo trình. `answerable` nghĩa là máy nghi có đáp án ở chunk được nêu. **Hãy xem kỹ các dòng này**, nhưng quyết định cuối cùng là của bạn.

## 5. Xử lý trường hợp khó

- **Đáp án đúng nhưng diễn đạt khác span:** chấp nhận nếu cùng nghĩa. Đáp án không cần chép nguyên văn.
- **Đáp án dài hơn cần thiết:** chọn `edit` và rút gọn trong `edited_answer`. Giữ phần cốt lõi, lý tưởng dưới 25 từ.
- **Câu hỏi đúng nhưng có thể có thêm đáp án ở chỗ khác trong sách:** vẫn `accept` nếu đáp án đề xuất đúng và đủ. Ghi chỗ bổ sung vào `note` nếu nó làm đáp án thành "chưa đủ".
- **Lỗi chính tả trong span:** span được chép nguyên văn từ PDF, kể cả lỗi chính tả, nên không cần sửa. Nếu lỗi nằm trong câu hỏi hoặc đáp án thì sửa bằng `edit`.
- **Không chắc chắn sau khoảng 3 phút tìm kiếm:** chọn nhãn theo đánh giá tốt nhất của bạn và ghi `CHƯA CHẮC` vào `note`. Các câu này sẽ được xem lại khi giải quyết bất đồng.
- **Nội dung giáo trình sai về chuyên môn:** đánh giá theo giáo trình. Ghi `GIÁO TRÌNH SAI: …` vào `note`.
- **Câu `unanswerable` mà giáo trình có một phần thông tin:** nếu phần có sẵn đủ để trả lời đúng câu hỏi thì `reject`. Nếu chỉ liên quan mà không trả lời được thì `accept`.
- **Câu thuộc split `dev`:** gán nhãn như câu `test`.

## 6. Ví dụ mẫu

Các ví dụ dưới đây lấy từ chính bộ nháp. Nhãn gợi ý chỉ để minh họa cách áp dụng tiêu chí.

### single

**PTIT-TEST-0479** (CNTT-PTTK)

- Câu hỏi: Loại hệ thống tiền TMĐT dựa trên cơ sở tri thức ra đời vào thập niên nào và được tạo ra với mục đích gì?
- Đáp án đề xuất: Xuất hiện vào những năm 80 của thế kỷ XX, nhằm tạo ra sự linh hoạt đối với người sử dụng.
- Bằng chứng: CNTT-PTTK-0002 (trang PDF 10)
- **Gợi ý:** `label=accept`, `type_correct=yes`, `evidence_correct=yes`. Đáp án ngắn, đúng với span, câu hỏi diễn đạt lại chứ không chép nguyên văn.

**PTIT-TEST-0731** (DT-VLSI)

- Câu hỏi: Vì sao trở kháng dây dẫn tính từ layout được biểu diễn theo ôm trên ô vuông chứ không theo ôm trên micrômét vuông?
- Đáp án đề xuất: Vì trở kháng của một đơn vị vuông vật liệu là giống nhau cho hình vuông với mọi kích thước, nên chỉ cần đo tỷ lệ các cạnh của dây.
- Bằng chứng: DT-VLSI-0036 (trang PDF 43)
- **Gợi ý:** `label=accept`, `type_correct=yes`, `evidence_correct=yes`. Đáp án diễn giải lại span nhưng giữ đúng ý. Không cần trùng chữ.

**PTIT-TEST-0157** (ATTT-TMDT)

- Câu hỏi: Khi quyết định có hiệu lực cho phép người khiếu kiện đăng ký tên miền .vn bị thu hồi, họ được quyền ưu tiên đăng ký trong bao lâu?
- Đáp án đề xuất: Được ưu tiên đăng ký trong vòng 10 ngày liên tục kể từ khi có hiệu lực pháp luật; hết thời hạn này tên miền được cho đăng ký tự do.
- Bằng chứng: ATTT-TMDT-0192 (trang PDF 140)
- **Gợi ý:** `label=edit`, `type_correct=yes`, `evidence_correct=yes`. Câu hỏi chỉ hỏi *bao lâu*, còn vế sau của đáp án ("hết thời hạn này…") là thừa. Ghi `edited_answer`: "Trong vòng 10 ngày liên tục kể từ khi quyết định có hiệu lực pháp luật."

### multi_intra

**PTIT-TEST-0066** (ATTT-CS)

- Câu hỏi: Mô hình được dùng rộng rãi nhất để hiện thực điều khiển truy nhập bắt buộc là mô hình nào, và theo nguyên tắc "ghi lên" của mô hình đó người dùng mức k được ghi vào đối tượng nào?
- Đáp án đề xuất: Mô hình Bell-LaPadula; theo nguyên tắc ghi lên, người dùng mức k chỉ ghi được các đối tượng ở cùng mức bảo mật hoặc cao hơn.
- Bằng chứng: ATTT-CS-0091 (trang PDF 107); ATTT-CS-0092 (trang PDF 109)
- **Gợi ý:** `label=accept`, `type_correct=yes`, `evidence_correct=yes`. Kiểu câu cầu nối: chunk 0091 cho biết *mô hình nào* (Bell-LaPadula), chunk 0092 cho biết *quy tắc ghi lên* của mô hình đó. Cần cả hai chunk.

**PTIT-TEST-0293** (CNTT-DH)

- Câu hỏi: Thuật toán Cohen–Sutherland và thuật toán Sutherland–Hodgman lần lượt dùng để xén loại đối tượng nào và xác định vị trí tương đối của điểm bằng cách nào?
- Đáp án đề xuất: Cohen–Sutherland xén đoạn thẳng, gán mã vùng 4-bit cho mỗi điểm cuối; Sutherland–Hodgman xén đa giác, xét các đỉnh nằm bên trái hay bên phải từng cạnh của cửa sổ cắt tỉa.
- Bằng chứng: CNTT-DH-0047 (trang PDF 61); CNTT-DH-0054 (trang PDF 69)
- **Gợi ý:** `label=accept`, `type_correct=yes`, `evidence_correct=yes`. Kiểu câu so sánh hai thuật toán nằm ở hai mục khác nhau. Có dùng chữ "và" nhưng hai vế cùng một trục so sánh, nên chấp nhận.

**PTIT-TEST-1125** (MKT-PT)

- Câu hỏi: Trong thuật toán K-means, sau khi tạo k điểm trung tâm ngẫu nhiên thì từng điểm dữ liệu được xử lý thế nào, và độ 'gần' giữa các cá nhân thường được đo bằng thước đo nào?
- Đáp án đề xuất: Mỗi điểm được chỉ định vào điểm trung tâm gần nhất, sau đó tính lại tâm cụm và lặp lại đến khi hội tụ; khoảng cách giữa các cá nhân đo bằng khoảng cách Euclidean.
- Bằng chứng: MKT-PT-0020 (trang PDF 28); MKT-PT-0016 (trang PDF 22)
- **Gợi ý:** `label=accept`, `type_correct=yes`, `evidence_correct=yes`. Kiểu câu tổng hợp hai sự kiện (các bước K-means và thước đo khoảng cách). Chấp nhận được, nhưng nếu thấy hai vế rời rạc thì có thể chọn `edit` để thu hẹp câu hỏi.

### multi_cross (cross_type=chapter)

**PTIT-TEST-0323** (CNTT-DH)

- Câu hỏi: Hệ màu định hướng phần cứng nào được dùng cho máy tính, và OpenGL mở rộng hệ màu đó thành chế độ màu nào với thành phần bổ sung có ý nghĩa gì?
- Đáp án đề xuất: RGB dùng cho máy tính; OpenGL có chế độ RGBA, trong đó thành phần A (alpha) không ảnh hưởng trực tiếp lên màu pixel mà có thể dùng để xác định độ trong suốt.
- Bằng chứng: CNTT-DH-0085 (trang PDF 105); CNTT-DH-0139 (trang PDF 165)
- **Gợi ý:** `label=accept`, `type_correct=yes`, `evidence_correct=yes`. Chương 6 (hệ màu RGB) nối với chương 9 (chế độ RGBA của OpenGL). Là câu cầu nối tự nhiên.

**PTIT-TEST-0446** (CNTT-KDL)

- Câu hỏi: Trong khối dữ liệu OLAP xây từ bảng thời tiết theo các chiều time (tuần), outlook và play, muốn tổng hợp số liệu từ mức tuần lên mức cao hơn trong phân cấp thời gian thì dùng thao tác nào?
- Đáp án đề xuất: Thao tác cuộn lên (drill-up/roll-up), dùng để tổng hợp dữ liệu từ mức thấp lên mức cao trong phân cấp của một chiều.
- Bằng chứng: CNTT-KDL-0011 (trang PDF 21); CNTT-KDL-0061 (trang PDF 97)
- **Gợi ý:** `label=accept`, `type_correct=yes`, `evidence_correct=yes`. Chương 1 cho ví dụ khối dữ liệu thời tiết, chương 3 định nghĩa thao tác cuộn lên. Hai chương khác nhau, cả hai bước đều cần.

**PTIT-TEST-1281** (MKT-QTBH)

- Câu hỏi: Kế hoạch bán hàng lý tưởng kỳ vọng bao nhiêu phần trăm doanh số đến từ khách hàng mới, trong khi chi phí chinh phục một khách hàng mới gấp bao nhiêu lần chi phí duy trì một khách hàng hiện có?
- Đáp án đề xuất: Khoảng 75% doanh số từ khách hàng mới (25% từ khách hiện tại); chi phí chinh phục khách mới gấp khoảng 5 lần chi phí duy trì khách hiện có.
- Bằng chứng: MKT-QTBH-0012 (trang PDF 21); MKT-QTBH-0115 (trang PDF 145)
- **Gợi ý:** `label=accept`, `type_correct=yes`, `evidence_correct=yes`. Ghép hai con số từ chương 2 và chương 6 cùng chủ đề khách hàng mới/cũ. Chấp nhận được, nhưng đây là kiểu câu "ghép" nên không nên có quá nhiều.

### multi_cross (cross_type=book)

**PTIT-TEST-0933** (DT-VXL)

- Câu hỏi: Giao diện nối tiếp của 8051 phải đấu thêm bộ đệm thích hợp mới thành cổng RS-232; vì sao không thể nối thẳng tín hiệu logic của chip vào đường RS-232?
- Đáp án đề xuất: Vì mức điện áp RS-232 không tương thích với TTL: logic 0 là từ +3V đến +25V, logic 1 là từ -25V đến -3V.
- Bằng chứng: DT-VXL-0049 (trang PDF 61); DT-NHUNG-0140 (trang PDF 192)
- **Gợi ý:** `label=accept`, `type_correct=yes`, `evidence_correct=yes`. Câu cầu nối liên sách tốt: DT-VXL cho biết 8051 cần bộ đệm để thành cổng RS-232, DT-NHUNG giải thích mức điện áp RS-232 không tương thích TTL.

**PTIT-TEST-0100** (ATTT-CS)

- Câu hỏi: Điều 289 Bộ luật Hình sự 2015 coi việc cố ý vượt qua tường lửa để xâm nhập mạng máy tính của người khác là tội phạm; về mặt kỹ thuật, tường lửa thường được đặt ở đâu trong mạng và xác định gói tin hợp pháp bằng cách nào?
- Đáp án đề xuất: Tường lửa thường đặt ở cổng vào mạng nội bộ để mọi gói tin vào, ra đều đi qua; gói tin hợp pháp được xác định bằng thao tác lọc dựa trên các luật.
- Bằng chứng: ATTT-TMDT-0160 (trang PDF 116); ATTT-CS-0097 (trang PDF 116)
- **Gợi ý:** `label=edit`, `type_correct=no`, `evidence_correct=yes`. Che chunk luật (ATTT-TMDT-0160) đi thì vẫn trả lời đủ chỉ bằng ATTT-CS-0097, vì Điều 289 chỉ đóng vai trò bối cảnh trong câu hỏi. Chọn `type_correct=no`, ghi vào `note`: "thực chất là single (ATTT-CS-0097)". Có thể `edit` để câu hỏi cần cả hai cuốn, ví dụ hỏi thêm hình phạt.

**PTIT-TEST-0812** (DT-VLSI)

- Câu hỏi: Trong đường dữ liệu CMOS, bộ tạo bit chẵn lẻ được hiện thực bằng cấu trúc cổng nào, và khi bật kiểm tra chẵn lẻ cho USART của STM32F1 thì bit chẵn lẻ tính được đặt ở vị trí nào của dữ liệu truyền?
- Đáp án đề xuất: Bộ tạo chẵn lẻ là một cột tuyến tính với kênh định tuyến dạng cây nối với các cổng XOR; trên USART, parity được chèn vào vị trí MSB của dữ liệu truyền (bit 9 với word length 9 bit, bit 8 với 8 bit).
- Bằng chứng: DT-VLSI-0083 (trang PDF 118); DT-NHUNG-0142 (trang PDF 195)
- **Gợi ý:** `label=accept`, `type_correct=yes`, `evidence_correct=yes`. Hai vế từ hai cuốn (bộ tạo chẵn lẻ trong VLSI và vị trí bit chẵn lẻ của USART STM32). Là câu ghép, nhưng mỗi vế cần một cuốn.

### unanswerable

**PTIT-TEST-0823** (DT-VLSI)

- Câu hỏi: Liều cấy Bo (số ion trên mỗi cm²) điển hình để tạo các đảo p trong quy trình CMOS SOI là bao nhiêu?
- Đáp án đề xuất: (không có đáp án)
- Bằng chứng: DT-VLSI-0026 (trang PDF 32)
- **Gợi ý:** `label=accept`, `type_correct=yes`, `evidence_correct=yes`. Tài liệu chỉ nói Bo được cấy để tạo vùng p, không có liều cấy. Câu hỏi hợp lý, dễ gây ảo giác (mô hình có thể bịa ra một con số).

**PTIT-TEST-0119** (ATTT-CS)

- Câu hỏi: Theo Luật An toàn thông tin mạng 2015 của Việt Nam, hệ thống thông tin được phân loại thành bao nhiêu cấp độ an toàn?
- Đáp án đề xuất: (không có đáp án)
- Bằng chứng: ATTT-CS-0120 (trang PDF 139)
- **Gợi ý:** `label=accept`, `type_correct=yes`, `evidence_correct=yes`. Tài liệu chỉ nói Luật ATTT mạng "có quy định phân loại cấp độ", không nêu số cấp. Lưu ý cột `similar_questions`: PTIT-TEST-0228 hỏi gần như cùng ý, nên câu nào gán nhãn sau thì `reject` và ghi `trùng PTIT-TEST-0119`.

**PTIT-TEST-1180** (MKT-PT)

- Câu hỏi: Khi các biến phân khúc có thang đo khác nhau, có cần chuẩn hóa dữ liệu (ví dụ z-score) trước khi tính khoảng cách Euclidean không, và chuẩn hóa như thế nào?
- Đáp án đề xuất: (không có đáp án)
- Bằng chứng: MKT-PT-0016 (trang PDF 22)
- **Gợi ý:** `label=reject`, `type_correct=no`, `evidence_correct=no`. `auto_check=answerable`: câu sinh từ MKT-PT, nhưng **cuốn khác** (CNTT-KDL-0115/0116) có trình bày chuẩn hóa z-score trước khi phân cụm. Câu `unanswerable` phải không có đáp án trong **toàn bộ 11 cuốn**. Ghi vào `note`: "có đáp án ở CNTT-KDL-0116, trang …".

**PTIT-TEST-0476** (CNTT-KDL)

- Câu hỏi: Làm thế nào xác định số cụm K tối ưu cho K-means bằng phương pháp khuỷu tay (elbow) trên đồ thị SSE?
- Đáp án đề xuất: (không có đáp án)
- Bằng chứng: CNTT-KDL-0122 (trang PDF 183)
- **Gợi ý:** `label=reject`, `type_correct=no`, `evidence_correct=no`. `auto_check=answerable`: MKT-PT-0020 mô tả tiêu chí elbow để chọn số cụm K, đủ trả lời câu hỏi.

**PTIT-TEST-0460** (CNTT-KDL)

- Câu hỏi: Trong quy trình KDD, riêng bước khai phá (tìm mẫu) chiếm khoảng bao nhiêu phần trăm công sức?
- Đáp án đề xuất: (không có đáp án)
- Bằng chứng: CNTT-KDL-0007 (trang PDF 14)
- **Gợi ý:** `label=accept`, `type_correct=yes`, `evidence_correct=yes`. `auto_check=partial`: tài liệu có tỷ lệ 60% cho *tiền xử lý*, nhưng không có tỷ lệ cho *bước khai phá* mà câu hỏi hỏi. Vẫn là câu không trả lời được; nên đọc kỹ để chắc câu hỏi không bị hiểu nhầm.


## 7. Nộp kết quả

1. Lưu trực tiếp vào file được giao: `annotation/round1/annotation_A.xlsx` hoặc `annotation_B.xlsx`. **Không đổi tên file, không đổi tên hay thứ tự cột, không xóa dòng.**
2. Mọi dòng phải có `label`. Dòng `label=edit` phải có ít nhất một trong hai cột `edited_question` hoặc `edited_answer`. Dòng `label=reject` nên có lý do ở `note`.
3. Sau khi nộp, hệ thống sẽ tự động:
   - kiểm tra còn dòng nào thiếu nhãn không;
   - tính Cohen's kappa trên phần chồng lấn A–B (cho `label`, `type_correct`, `evidence_correct`). Nếu kappa của `label` < 0,7, các câu bất đồng được xuất ra `annotation/disagreements.xlsx` để hai người thống nhất;
   - áp dụng các chỉnh sửa, bỏ câu `reject`, rồi kiểm tra lại số lượng và phân bố. Nếu thiếu câu, sẽ sinh bổ sung cho vòng 2.

**Ước lượng khối lượng:** người A làm 1.579 câu, khoảng 1–1,5 phút/câu (≈ 26–40 giờ). Người B làm 263 câu (≈ 4–7 giờ). Có thể chia file A cho nhiều người, miễn giữ nguyên ID và ghi rõ ai gán nhãn phần nào.


---

# Gán nhãn ảo giác (Giai đoạn 7)

## 8. Mục đích và file

- Tập câu hỏi: 150 câu PTIT test, lấy mẫu phân tầng theo loại câu hỏi (seed 42).
- Mỗi câu có câu trả lời của **4 hệ thống ẩn danh**, cùng dùng một mô hình ngôn ngữ nhỏ. Tổng cộng 600 dòng, thứ tự đã xáo trộn.
- File:
  - `annotation/halluc/halluc_A.xlsx`: toàn bộ 600 dòng;
  - `annotation/halluc/halluc_B.xlsx`: 120 dòng (20%), dùng để tính độ đồng thuận.
- Người gán B làm **mù**: không xem file A, không trao đổi với người gán A trước khi nộp. Hãy bắt đầu từ file trống được giao.
- Cột xám (chỉ đọc):
  - `question`;
  - `gold_answer` và `gold_evidence`: đáp án và bằng chứng chuẩn đã duyệt ở vòng 1;
  - `context`: tài liệu mà hệ thống đã thấy khi trả lời;
  - `answer`: câu trả lời cần chấm.
- Cột vàng: `label`, `note`.
- Một số dòng có `context` = "(hệ thống này không dùng tài liệu)". Hệ thống đó trả lời bằng kiến thức riêng của mô hình. Vẫn chấm theo cùng định nghĩa: so với đáp án chuẩn và nội dung giáo trình.
- **Không suy đoán hệ thống nào tạo câu trả lời.**

## 9. Định nghĩa nhãn `label`

Chuẩn đối chiếu là **nội dung giáo trình**, thể hiện qua `gold_answer`, `gold_evidence` và `context`. Không dùng hiểu biết riêng của bạn để "chấm đúng" một thông tin mà giáo trình không nêu.

| Nhãn | Khi nào dùng |
|---|---|
| `faithful` | Câu trả lời đưa ra thông tin, và **mọi** thông tin đó đúng với giáo trình: có trong `gold_evidence` hoặc `context`, hoặc suy ra trực tiếp từ đó. Câu trả lời thiếu ý hoặc diễn đạt khác vẫn là `faithful`, miễn không có ý sai hay ý bịa. |
| `partially_hallucinated` | Có phần đúng với giáo trình, nhưng **có ít nhất một** thông tin sai, mâu thuẫn, hoặc không có căn cứ trong giáo trình. Ví dụ: con số, tên, bước hoặc thành phần được thêm vào. |
| `hallucinated` | Thông tin **chính** của câu trả lời sai, mâu thuẫn với giáo trình, hoặc bịa ra. Với câu hỏi không có đáp án (`gold_answer` = "(câu hỏi không có đáp án trong tài liệu)"), mọi câu trả lời đưa ra một đáp án cụ thể thay vì từ chối đều là `hallucinated`. |
| `correct_refusal` | Hệ thống từ chối (ví dụ "Không tìm thấy thông tin trong tài liệu.") và câu hỏi **thật sự không có đáp án** trong giáo trình. |
| `wrong_refusal` | Hệ thống từ chối, nhưng câu hỏi **có đáp án** trong giáo trình (có `gold_answer`). Nhãn này không phải ảo giác, nhưng là lỗi bỏ sót. |

**Quy tắc phân biệt**
1. Câu trả lời có dạng từ chối thì chọn `correct_refusal` hoặc `wrong_refusal`, tùy câu hỏi có đáp án hay không. Nếu vừa từ chối vừa đưa thêm thông tin, chấm theo phần thông tin đó (một trong ba nhãn đầu) và ghi `note`.
2. Có ý **sai hoặc bịa** thì không bao giờ là `faithful`. Nếu sai ở ý chính, chọn `hallucinated`. Nếu sai ở ý phụ, chọn `partially_hallucinated`.
3. Thông tin **đúng với giáo trình nhưng không có trong `context`** (ví dụ hệ thống lấy từ kiến thức riêng mà trùng với `gold_evidence`) vẫn là `faithful`. Nhãn này đo ảo giác so với giáo trình, không đo việc bám `context`.
4. Thông tin **không có trong giáo trình**, dù có thể đúng ngoài đời, thì chọn `partially_hallucinated` nếu là ý phụ, `hallucinated` nếu là ý chính. Ghi vào `note`: "ngoài giáo trình".
5. Câu trả lời lạc đề (không trả lời điều được hỏi) nhưng không sai sự thật: chọn `hallucinated` và ghi `note`: "lạc đề".

## 10. Ví dụ minh họa

Các ví dụ dưới đây được tự dựng để minh họa; chúng không nằm trong file cần gán nhãn.

- Câu hỏi: "Trong vi điều khiển 8051, cổng nào có chức năng thứ hai là các tín hiệu ngắt ngoài?" `gold_answer`: "Cổng P3 (INT0 = P3.2, INT1 = P3.3)".
  - "Cổng P3." → `faithful`: thiếu chi tiết nhưng không sai.
  - "Cổng P3, gồm INT0 ở P3.2 và INT1 ở P3.4." → `partially_hallucinated`: INT1 sai chân.
  - "Cổng P1." → `hallucinated`.
  - "Không tìm thấy thông tin trong tài liệu." → `wrong_refusal`.
- Câu hỏi không có đáp án: "Chuẩn ISO/IEC 27001 phiên bản năm 2022 bổ sung bao nhiêu biện pháp kiểm soát mới?"
  - "Không tìm thấy thông tin trong tài liệu." → `correct_refusal`.
  - "11 biện pháp mới." → `hallucinated`, kể cả khi con số đó đúng ngoài đời; ghi `note`: "ngoài giáo trình".

## 11. Nộp kết quả

- Lưu trực tiếp vào file được giao. Không đổi tên file, không đổi thứ tự hay xóa dòng, không sửa cột xám.
- Mọi dòng phải có `label`. Dòng `partially_hallucinated` và `hallucinated` nên ghi ngắn ý sai vào `note`.
- Sau khi nộp, hệ thống sẽ:
  - kiểm tra đủ nhãn;
  - tính Cohen's kappa trên 120 dòng chồng lấn A–B;
  - giải mã tên hệ thống;
  - tính tỷ lệ ảo giác thủ công;
  - đối chiếu với bộ chấm tự động (tương quan Spearman).
- **Ước lượng khối lượng:** khoảng 1–1,5 phút/dòng. Người A làm 600 dòng (khoảng 10–15 giờ), người B làm 120 dòng (khoảng 2–3 giờ).
