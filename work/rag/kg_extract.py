"""S5 offline step: entity / relation extraction per chunk with the primary SLM (structured JSON output).

The structural frame (book > chapter > section) is given in the prompt to disambiguate terms; the frame itself is
added to the graph from metadata, not by the SLM (see kg.py). Results are cached per chunk in
work/index/<corpus>/kg_extract.jsonl so the build can resume.
"""
import json
import os
import time

from . import data, llm

SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "type": {"type": "string"}, "description": {"type": "string"}},
            "required": ["name", "type", "description"]}},
        "relations": {"type": "array", "items": {"type": "object", "properties": {
            "source": {"type": "string"}, "relation": {"type": "string"}, "target": {"type": "string"}},
            "required": ["source", "relation", "target"]}},
    },
    "required": ["entities", "relations"],
}

PROMPT = """Bạn trích xuất tri thức từ tài liệu để xây dựng đồ thị tri thức.
Vị trí đoạn văn: {frame}

ĐOẠN VĂN:
{text}

Yêu cầu:
1. "entities": tối đa 12 thực thể quan trọng nhất (khái niệm, thành phần, thiết bị, thuật toán, phương pháp, quy trình, giao thức, tiêu chuẩn, văn bản pháp luật, tổ chức, người, địa danh, sự kiện, đại lượng). Mỗi thực thể gồm:
   - "name": tên ngắn gọn, đúng như trong đoạn văn; nếu có tên tiếng Anh hoặc viết tắt thì ghi trong ngoặc, ví dụ "Kho dữ liệu (Data Warehouse)";
   - "type": một trong KHÁI_NIỆM, THÀNH_PHẦN, PHƯƠNG_PHÁP, QUY_TRÌNH, TIÊU_CHUẨN_PHÁP_LUẬT, TỔ_CHỨC_NGƯỜI, ĐỊA_DANH_SỰ_KIỆN, ĐẠI_LƯỢNG, KHÁC;
   - "description": một câu ngắn (tối đa 15 từ) mô tả thực thể, chỉ dựa trên đoạn văn.
2. "relations": tối đa 12 quan hệ giữa các thực thể ở trên. Mỗi quan hệ gồm "source", "relation" (cụm động từ ngắn, ví dụ "là một loại", "gồm", "dùng để", "khác với", "quy định"), "target". "source" và "target" phải trùng tên với thực thể đã liệt kê.
Chỉ dùng thông tin có trong đoạn văn. Trả về JSON."""


def frame_of(chunk):
    m = chunk["meta"]
    if m.get("chapter") is not None and "section" in m:
        parts = [f"Giáo trình «{chunk['title']}»", m.get("chapter") or "", m.get("section") or ""]
    else:
        parts = [f"Bài viết «{chunk['title']}»"]
    return " > ".join(p for p in parts if p)


def extract(chunk, slm=llm.PRIMARY):
    r = llm.chat(slm, PROMPT.format(frame=frame_of(chunk), text=chunk["text"]), num_predict=2500, fmt=SCHEMA)
    try:
        out = json.loads(r["text"])
        ok = True
    except json.JSONDecodeError:
        out, ok = {"entities": [], "relations": []}, False
    return {"chunk_id": chunk["chunk_id"], "ok": ok, "entities": out.get("entities", [])[:20], "relations": out.get("relations", [])[:20],
            "latency_s": r["latency_s"], "prompt_tokens": r["prompt_tokens"], "output_tokens": r["output_tokens"]}


def build(corpus_name, workers=1, limit=None):
    from concurrent.futures import ThreadPoolExecutor
    path = os.path.join(data.WORK, "index", corpus_name, "kg_extract.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["chunk_id"] for l in open(path, encoding="utf-8")}
    todo = [c for c in data.corpus(corpus_name) if c["chunk_id"] not in done][:limit]
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as ex:
        for i, rec in enumerate(ex.map(extract, todo)):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"{corpus_name}: {i + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    return path
