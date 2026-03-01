"""Common IO utilities with lazy export loading."""

from __future__ import annotations

from importlib import import_module

_LAZY_MODULE_ATTRS: dict[str, tuple[str, str]] = {
    "write_csv_dicts": ("ztb.io.csv_io", "write_csv_dicts"),
    "read_json": ("ztb.io.json_io", "read_json"),
    "read_json_object": ("ztb.io.json_io", "read_json_object"),
    "read_json_array": ("ztb.io.json_io", "read_json_array"),
    "write_json": ("ztb.io.json_io", "write_json"),
    "append_jsonl": ("ztb.io.jsonl", "append_jsonl"),
    "iter_jsonl_objects": ("ztb.io.jsonl", "iter_jsonl_objects"),
    "read_jsonl_objects": ("ztb.io.jsonl", "read_jsonl_objects"),
    "append_jsonl_gz": ("ztb.io.jsonl_gz", "append_jsonl_gz"),
    "read_jsonl_gz": ("ztb.io.jsonl_gz", "read_jsonl_gz"),
    "write_state_payload": ("ztb.io.state_persistence", "write_state_payload"),
    "read_state_payload": ("ztb.io.state_persistence", "read_state_payload"),
    "read_yaml": ("ztb.io.yaml_io", "read_yaml"),
    "write_yaml": ("ztb.io.yaml_io", "write_yaml"),
    "read_text": ("ztb.io.text_io", "read_text"),
    "write_text": ("ztb.io.text_io", "write_text"),
    "read_last_lines": ("ztb.io.text_io", "read_last_lines"),
}

__all__ = list(_LAZY_MODULE_ATTRS)


def __getattr__(name: str) -> object:
    target = _LAZY_MODULE_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
