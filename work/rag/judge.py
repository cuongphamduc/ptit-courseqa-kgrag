"""LLM-as-judge for faithfulness / groundedness of RAG answers (judge model differs from every evaluated SLM).

For each non-refusal answer of a RAG system, the judge sees the exact context given to the SLM (passages + facts),
the question and the answer, and labels the answer:
  supported      every factual claim is stated in or directly entailed by the context       -> 1.0
  partial        some claims are supported, at least one is not                              -> 0.5
  unsupported    the main claim is not in the context or contradicts it                      -> 0.0
The judge is validated against the manual hallucination labels of Stage 7 (Spearman).
"""
import json

import ollama

JUDGE = {"tag": "qwen3.6:latest", "think": False}
HOST = "http://127.0.0.1:11435"
SCHEMA = {"type": "object", "properties": {"label": {"type": "string", "enum": ["supported", "partial", "unsupported"]},
                                           "reason": {"type": "string"}}, "required": ["label", "reason"]}
SCORE = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}

PROMPT = """Bạn là người chấm điểm độ trung thực (faithfulness) của câu trả lời so với TÀI LIỆU được cung cấp.
Không dùng kiến thức bên ngoài; chỉ so câu trả lời với TÀI LIỆU.

TÀI LIỆU:
{context}

CÂU HỎI: {question}
CÂU TRẢ LỜI CẦN CHẤM: {answer}

Gán nhãn:
- "supported": mọi thông tin trong câu trả lời đều có trong TÀI LIỆU hoặc suy ra trực tiếp từ TÀI LIỆU;
- "partial": một phần có trong TÀI LIỆU, nhưng có ít nhất một thông tin không có trong TÀI LIỆU;
- "unsupported": thông tin chính của câu trả lời không có trong TÀI LIỆU hoặc mâu thuẫn với TÀI LIỆU.
Trả về JSON gồm "label" và "reason" (một câu ngắn)."""

_client = None


def judge(context, question, answer, num_ctx=12288):
    global _client
    if _client is None:
        _client = ollama.Client(host=HOST, timeout=900)
    r = _client.chat(model=JUDGE["tag"], messages=[{"role": "user", "content": PROMPT.format(context=context, question=question, answer=answer)}],
                     think=JUDGE["think"], format=SCHEMA, keep_alive="30m",
                     options={"temperature": 0.0, "seed": 42, "num_ctx": num_ctx, "num_predict": 200})
    try:
        out = json.loads(r.message.content)
        label = out["label"] if out.get("label") in SCORE else "unsupported"
        return {"label": label, "score": SCORE[label], "reason": out.get("reason", ""), "ok": True}
    except (json.JSONDecodeError, KeyError, TypeError):
        return {"label": None, "score": None, "reason": r.message.content[:200], "ok": False}
