"""Answer prompts shared by every system (S0-S5) and every SLM. The refusal sentence is identical everywhere."""
from .text import REFUSAL

RULES = (
    "Trả lời câu hỏi bằng tiếng Việt, ngắn gọn và trực tiếp (một cụm từ hoặc tối đa hai câu), không giải thích thêm.\n"
    f"Nếu không đủ thông tin để trả lời, chỉ viết đúng câu: {REFUSAL}"
)


def rag_prompt(question, passages, facts=None):
    """passages: list of (header, text); facts: optional list of knowledge-graph statements (S4/S5)"""
    docs = "\n\n".join(f"[{i + 1}] {h}\n{t}".replace("] \n", "]\n") for i, (h, t) in enumerate(passages))
    fact_block = ""
    if facts:
        fact_block = "\n\nCÁC QUAN HỆ LIÊN QUAN (trích từ tài liệu):\n" + "\n".join(f"- {f}" for f in facts)
    return (f"Chỉ sử dụng thông tin trong TÀI LIỆU dưới đây. {RULES}\n\n"
            f"TÀI LIỆU:\n{docs}{fact_block}\n\nCÂU HỎI: {question}\nTRẢ LỜI:")


def closed_book_prompt(question):
    return f"{RULES}\nChỉ trả lời khi bạn chắc chắn.\n\nCÂU HỎI: {question}\nTRẢ LỜI:"
