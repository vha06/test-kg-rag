import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.json_repair import robust_json_loads


def test_markdown_and_commentary():
    text = 'Dưới đây là JSON trích xuất:\n```json\n{"entities": [{"name": "A"}]}\n```\nHy vọng hữu ích.'
    res = robust_json_loads(text)
    assert res == {"entities": [{"name": "A"}]}


def test_trailing_commas():
    # Lỗi điển hình: Expecting property name enclosed in double quotes
    text = '{"entities": [{"name": "A", "type": "ORG",}], "relationships": [],}'
    res = robust_json_loads(text)
    assert res["entities"][0]["name"] == "A"
    assert res["relationships"] == []


def test_think_tags_cleaning():
    text = '<think>\nĐây là suy luận nội bộ của Qwen/DeepSeek\n</think>\n{"entities": [{"name": "B"}]}'
    res = robust_json_loads(text)
    assert res["entities"][0]["name"] == "B"


def test_single_quotes_python_dict():
    text = "{'entities': [{'name': 'C', 'aliases': ['c1']}], 'valid': true, 'none_val': null}"
    res = robust_json_loads(text)
    assert res["entities"][0]["name"] == "C"
    assert res["entities"][0]["aliases"] == ["c1"]
    assert res["valid"] is True
    assert res["none_val"] is None


def test_inner_unescaped_quotes():
    text = '{"entities": [\n  {"name": "D", "description": "Tổ chức "Thành Công" tại VN", "type": "ORG"}\n]}'
    res = robust_json_loads(text)
    assert res["entities"][0]["name"] == "D"
    assert "Thành Công" in res["entities"][0]["description"]


def test_truncated_json():
    text = '{"entities": [{"name": "E", "description": "Dở dang'
    res = robust_json_loads(text)
    assert res["entities"][0]["name"] == "E"


def test_combination_repairs():
    text = '{"entities": [\n  {"name": "F", "description": "Hệ thống "An ninh" tại VN", }, ]'
    res = robust_json_loads(text)
    assert res["entities"][0]["name"] == "F"
    assert "An ninh" in res["entities"][0]["description"]
