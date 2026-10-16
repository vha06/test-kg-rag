import json
import os
import urllib.request
import urllib.error
from typing import Dict, Any, Optional
from generation.models import LLMConfig


class MultiProviderLLMClient:
    """
    Client kết nối đa nguồn mô hình LLM (Gemini, OpenAI-compatible, Ollama, Mock),
    sử dụng chuẩn HTTP REST không phụ thuộc thư viện bên ngoài.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        provider = (self.config.provider or "gemini").lower()

        # Check API key from env if missing
        api_key = self.config.api_key
        if not api_key:
            if provider == "gemini":
                api_key = os.environ.get("GEMINI_API_KEY", "")
            elif provider in ("openai_compatible", "openai"):
                api_key = os.environ.get("OPENAI_API_KEY", "")

        # Fallback to mock mode if testing or missing required credentials
        ans, _ = self.generate_with_thought(system_prompt, user_prompt)
        return ans

    def generate_with_thought(self, system_prompt: str, user_prompt: str) -> tuple[str, Optional[str]]:
        provider = (self.config.provider or "gemini").lower()

        # Check API key from env if missing
        api_key = self.config.api_key
        if not api_key:
            if provider == "gemini":
                api_key = os.environ.get("GEMINI_API_KEY", "")
            elif provider in ("openai_compatible", "openai"):
                api_key = os.environ.get("OPENAI_API_KEY", "")

        # Fallback to mock mode if testing or missing required credentials
        if provider == "mock" or (provider in ("gemini", "openai_compatible", "openai") and not api_key):
            if provider != "mock":
                return self._mock_generate(system_prompt, user_prompt, note="[LƯU Ý: Chưa cấu hình API Key. Đây là câu trả lời mô phỏng cấu trúc trích dẫn]."), None
            return self._mock_generate(system_prompt, user_prompt), None

        if provider == "gemini":
            return self._call_gemini(api_key, system_prompt, user_prompt)
        elif provider in ("openai_compatible", "openai"):
            return self._call_openai_compatible(api_key, system_prompt, user_prompt)
        elif provider == "ollama":
            return self._call_ollama(system_prompt, user_prompt), None
        else:
            raise ValueError(f"Provider không được hỗ trợ: {provider}")

    def _call_gemini(self, api_key: str, system_prompt: str, user_prompt: str) -> tuple[str, Optional[str]]:
        model = self.config.model_name or "gemini-2.5-flash"
        # Strip models/ prefix if provided
        if model.startswith("models/"):
            model = model[len("models/"):]

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        
        # Đảm bảo maxOutputTokens đủ lớn (tối thiểu 8192) để mô hình có đủ token cho cả Thinking và Answer hoàn chỉnh
        max_tokens = max(self.config.max_tokens or 8192, 8192)

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                candidates = res_data.get("candidates", [])
                if not candidates:
                    raise RuntimeError("Gemini không trả về kết quả.")

                parts = candidates[0].get("content", {}).get("parts", [])
                
                # Tách biệt phần suy luận nội bộ (thought: True) và phần câu trả lời chính thức
                answer_parts = []
                thought_parts = []
                for p in parts:
                    if p.get("thought", False):
                        thought_parts.append(p.get("text", ""))
                    else:
                        answer_parts.append(p.get("text", ""))

                thought_text = "".join(thought_parts).strip() or None
                if answer_parts:
                    answer_text = "".join(answer_parts).strip()
                else:
                    # Fallback nếu mô hình không phân tách bằng cờ thought
                    answer_text = "".join(p.get("text", "") for p in parts).strip()

                finish_reason = candidates[0].get("finishReason", "")
                if finish_reason == "MAX_TOKENS":
                    answer_text += "\n\n⚠️ *[Lưu ý: Câu trả lời đã đạt giới hạn độ dài token tối đa (MAX_TOKENS) nên bị dừng lại. Hãy tăng max_tokens để nhận câu trả lời trọn vẹn hơn].*"

                return answer_text, thought_text
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Gemini API Error {e.code}: {err_body}")

    def _call_openai_compatible(self, api_key: str, system_prompt: str, user_prompt: str) -> str:
        base_url = (self.config.base_url or "https://api.openai.com/v1").rstrip("/")
        url = f"{base_url}/chat/completions"
        model = self.config.model_name or "gpt-4o-mini"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                choices = res_data.get("choices", [])
                if not choices:
                    raise RuntimeError("OpenAI-compatible API không trả về choices.")
                return choices[0].get("message", {}).get("content", "")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenAI-compatible API Error {e.code}: {err_body}")

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        base_url = (self.config.base_url or "http://localhost:11434").rstrip("/")
        url = f"{base_url}/api/chat"
        model = self.config.model_name or "qwen2.5:7b"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
            },
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data.get("message", {}).get("content", "")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Không thể kết nối tới Ollama tại {base_url}. Hãy đảm bảo Ollama đã được khởi động. Lỗi: {e}")

    def _mock_generate(self, system_prompt: str, user_prompt: str, note: str = "") -> str:
        """Sinh câu trả lời mẫu có trích dẫn nội tuyến chuẩn phục vụ kiểm thử và preview."""
        header = f"{note}\n\n" if note else ""
        return (
            f"{header}### Tóm tắt câu trả lời:\n"
            f"Theo quy định của pháp luật hiện hành [#1], các cá nhân và cơ quan có trách nhiệm "
            f"thực hiện nghiêm chỉnh các điều kiện đã ban hành [#1]. Đồng thời, các quy định này được bổ sung, hướng dẫn "
            f"chi tiết theo các văn bản chuyên ngành có liên quan [#2].\n\n"
            f"### Căn cứ pháp lý chi tiết:\n"
            f"1. **Quy định căn bản:** Được xác lập tại [#1]. Mọi quy trình thực hiện phải tuân thủ điều kiện quy định.\n"
            f"2. **Quy định sửa đổi / bổ sung hoặc hướng dẫn liên quan:** Căn cứ theo [#2], một số nội dung và tiêu chuẩn đã được cập nhật áp dụng.\n\n"
            f"### Lưu ý về hiệu lực pháp lý:\n"
            f"Người dân và cơ quan cần đối chiếu với ngày hiệu lực thi hành để áp dụng chính xác phiên bản quy phạm pháp luật đang có hiệu lực."
        )
