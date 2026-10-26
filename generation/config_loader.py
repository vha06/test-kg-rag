import os
from pathlib import Path
from typing import Dict, Any


def load_keys_config(base_dir: str = ".") -> Dict[str, str]:
    """
    Tự động đọc file key.txt hoặc .env trong thư mục gốc dự án
    và thiết lập các biến môi trường tương ứng cho hệ thống.
    """
    config: Dict[str, str] = {}
    base_path = Path(base_dir)

    # 1. Đọc key.txt nếu có
    key_file = base_path / "key.txt"
    if key_file.exists():
        try:
            with open(key_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            for line in lines:
                if "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()
                    config[k] = v
                    os.environ[k] = v
                elif not config.get("GEMINI_API_KEY") and len(line) > 20:
                    # Dòng đơn lẻ không có '=' thường là Gemini/Google API Key
                    config["GEMINI_API_KEY"] = line
                    os.environ["GEMINI_API_KEY"] = line
        except Exception as e:
            print(f"[config_loader] Lỗi khi đọc key.txt: {e}")

    # 2. Đọc .env nếu có
    env_file = base_path / ".env"
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip()
                        if k not in config:
                            config[k] = v
                            os.environ[k] = v
        except Exception as e:
            print(f"[config_loader] Lỗi khi đọc .env: {e}")

    # 3. Đồng bộ hóa mapping cho QWEN3 sang chuẩn OPENAI-compatible
    qwen_key = config.get("QWEN3_API_KEY") or os.environ.get("QWEN3_API_KEY")
    qwen_model = config.get("QWEN3_MODEL") or os.environ.get("QWEN3_MODEL")
    qwen_url = config.get("QWEN3_BASE_URL") or os.environ.get("QWEN3_BASE_URL")

    if qwen_key:
        os.environ.setdefault("OPENAI_API_KEY", qwen_key)
        config["OPENAI_API_KEY"] = qwen_key
    if qwen_model:
        os.environ.setdefault("OPENAI_MODEL", qwen_model)
        config["OPENAI_MODEL"] = qwen_model
    if qwen_url:
        os.environ.setdefault("OPENAI_BASE_URL", qwen_url)
        config["OPENAI_BASE_URL"] = qwen_url

    # 4. Tự động đồng bộ ra file .env nếu chưa có để đảm bảo tính sẵn sàng
    if not env_file.exists() and config:
        try:
            with open(env_file, "w", encoding="utf-8") as f:
                f.write("# Tự động sinh từ key.txt cho mô hình Local Qwen & API\n")
                if qwen_key:
                    f.write(f"QWEN3_API_KEY={qwen_key}\n")
                    f.write(f"OPENAI_API_KEY={qwen_key}\n")
                if qwen_model:
                    f.write(f"QWEN3_MODEL={qwen_model}\n")
                    f.write(f"OPENAI_MODEL={qwen_model}\n")
                if qwen_url:
                    f.write(f"QWEN3_BASE_URL={qwen_url}\n")
                    f.write(f"OPENAI_BASE_URL={qwen_url}\n")
                gemini_key = config.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
                if gemini_key:
                    f.write(f"GEMINI_API_KEY={gemini_key}\n")
        except Exception as e:
            print(f"[config_loader] Không thể tạo file .env: {e}")

    return config
