"""
Module hỗ trợ làm sạch, trích xuất và tự động sửa lỗi JSON (Auto-repair JSON)
khi nhận phản hồi từ các mô hình ngôn ngữ lớn (LLM, đặc biệt là các mô hình Local Qwen, DeepSeek, Llama).

Các lỗi phổ biến được xử lý:
1. Thẻ suy luận nội bộ (<think>...</think>, <thought>...</thought>).
2. Bao bọc markdown codeblock (```json ... ```) kèm văn bản giải thích phía trước hoặc phía sau.
3. Ký tự điều khiển (control characters) trong chuỗi.
4. Chú thích kiểu JavaScript (// comment, /* comment */).
5. Dấu phẩy thừa (Trailing commas: `,}` hoặc `,]`) - nguyên nhân gây lỗi "Expecting property name enclosed in double quotes".
6. Cú pháp Python dict (dùng nháy đơn 'key': 'val', True/False/None).
7. Dấu ngoặc kép lồng nhau chưa escape trong giá trị chuỗi.
8. JSON bị cắt cụt giữa chừng do chạm giới hạn token (Truncated JSON - thiếu ngoặc đóng).
"""

import re
import json
import ast
import logging
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


def robust_json_loads(text: Any) -> Any:
    """
    Trích xuất và phân tích cú pháp JSON từ chuỗi sinh ra bởi LLM một cách an toàn và chống chịu lỗi cao.

    Args:
        text: Chuỗi văn bản hoặc đối tượng chứa JSON.

    Returns:
        dict hoặc list đã được phân tích cú pháp.

    Raises:
        json.JSONDecodeError hoặc ValueError nếu chuỗi hoàn toàn không thể khôi phục thành JSON.
    """
    if text is None:
        raise ValueError("Chuỗi JSON đầu vào là None")

    if isinstance(text, (dict, list)):
        return text

    if not isinstance(text, str):
        text = str(text)

    # 1. Bỏ thẻ suy luận nội bộ (Qwen 3 / DeepSeek reasoning tokens)
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<thought>[\s\S]*?</thought>", "", text, flags=re.IGNORECASE)
    text = text.strip()

    # 2. Bóc tách nội dung từ codeblock markdown nếu có
    md_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if md_match:
        candidate_text = md_match.group(1).strip()
    else:
        candidate_text = text

    # Trích xuất đoạn từ { hoặc [ đầu tiên đến } hoặc ] cuối cùng để loại bỏ lời dẫn/kết luận
    first_brace = candidate_text.find("{")
    first_bracket = candidate_text.find("[")

    start_idx = -1
    if first_brace != -1 and first_bracket != -1:
        start_idx = min(first_brace, first_bracket)
    elif first_brace != -1:
        start_idx = first_brace
    elif first_bracket != -1:
        start_idx = first_bracket

    if start_idx != -1:
        last_brace = candidate_text.rfind("}")
        last_bracket = candidate_text.rfind("]")
        end_idx = max(last_brace, last_bracket)
        if end_idx > start_idx:
            candidate_text = candidate_text[start_idx : end_idx + 1]

    # Bước 3: Thử trực tiếp với json.loads chuẩn
    try:
        return json.loads(candidate_text, strict=False)
    except Exception:
        pass

    # Bước 4: Xóa ký tự điều khiển lạ nhưng giữ \n, \r, \t
    cleaned = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]",
        lambda m: " " if m.group(0) not in ("\n", "\r", "\t") else m.group(0),
        candidate_text,
    )
    try:
        return json.loads(cleaned, strict=False)
    except Exception:
        pass

    # Bước 5: Xóa comment dạng // hoặc /* */
    cleaned_no_comments = re.sub(r"(?<!https:)(?<!http:)//.*$", "", cleaned, flags=re.MULTILINE)
    cleaned_no_comments = re.sub(r"/\*[\s\S]*?\*/", "", cleaned_no_comments)
    try:
        return json.loads(cleaned_no_comments, strict=False)
    except Exception:
        pass

    # Bước 6: Xóa trailing commas (dấu phẩy thừa trước } hoặc ])
    # Đây là nguyên nhân hàng đầu gây ra: Expecting property name enclosed in double quotes
    cleaned_trailing = re.sub(r",+\s*([}\]])", r"\1", cleaned_no_comments)
    try:
        return json.loads(cleaned_trailing, strict=False)
    except Exception:
        pass

    # Bước 7: Thử qua ast.literal_eval (hỗ trợ nháy đơn 'key': 'value' và trailing commas kiểu Python)
    try:
        py_text = re.sub(r"\btrue\b", "True", cleaned_trailing, flags=re.IGNORECASE)
        py_text = re.sub(r"\bfalse\b", "False", py_text, flags=re.IGNORECASE)
        py_text = re.sub(r"\bnull\b", "None", py_text, flags=re.IGNORECASE)
        res = ast.literal_eval(py_text)
        if isinstance(res, (dict, list)):
            return res
    except Exception:
        pass

    # Bước 8: Tự sửa dấu ngoặc kép lồng nhau chưa escape (Unescaped quotes inside string values)
    def escape_inner_quotes(s: str) -> str:
        chars = list(s)
        n = len(chars)
        out = []
        i = 0
        in_string = False
        escape = False

        while i < n:
            c = chars[i]
            if escape:
                out.append(c)
                escape = False
                i += 1
                continue
            if c == "\\":
                escape = True
                out.append(c)
                i += 1
                continue
            if c == '"':
                if not in_string:
                    in_string = True
                    out.append(c)
                else:
                    # Đang ở trong chuỗi, kiểm tra ký tự tiếp theo bỏ qua khoảng trắng
                    j = i + 1
                    while j < n and chars[j] in " \t\r\n":
                        j += 1
                    next_c = chars[j] if j < n else ""
                    if next_c in (":", ",", "}", "]"):
                        # Dấu đóng chuỗi hợp lệ
                        in_string = False
                        out.append(c)
                    else:
                        # Dấu " nằm bên trong chuỗi chưa được escape -> escape thành \"
                        out.append('\\"')
                i += 1
                continue
            out.append(c)
            i += 1
        return "".join(out)

    escaped_quotes = None
    try:
        escaped_quotes = escape_inner_quotes(cleaned_trailing)
        return json.loads(escaped_quotes, strict=False)
    except Exception:
        pass

    # Bước 9: Tự động đóng các ngoặc bị thiếu do LLM bị cắt cụt (Truncated JSON Repair)
    def repair_truncated_json(s: str) -> Optional[Union[dict, list]]:
        chars = list(s)
        stack = []
        in_string = False
        escape = False

        for ch in chars:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if not in_string:
                if ch in ("{", "["):
                    stack.append("}" if ch == "{" else "]")
                elif ch in ("}", "]"):
                    if stack and stack[-1] == ch:
                        stack.pop()

        repaired = s
        if in_string:
            repaired += '"'

        # Bỏ dấu phẩy thừa hoặc hai chấm lơ lửng ở cuối
        repaired = re.sub(r"[,:]+\s*$", "", repaired.strip())

        # Đóng các ngoặc còn thiếu
        while stack:
            closing = stack.pop()
            repaired = re.sub(r"[,:]+\s*$", "", repaired.strip())
            repaired += closing

        # Xóa trailing commas lần nữa
        repaired = re.sub(r",+\s*([}\]])", r"\1", repaired)
        return json.loads(repaired, strict=False)

    for candidate in [escaped_quotes, cleaned_trailing]:
        if candidate:
            try:
                return repair_truncated_json(candidate)
            except Exception:
                pass

    # Nếu tất cả các bước khôi phục nâng cao đều không giải quyết được, ném ngoại lệ gốc
    return json.loads(candidate_text, strict=False)
