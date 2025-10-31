"""Utility helpers for loading YAML configuration without external dependencies."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import ast

try:  # pragma: no cover - executed only when PyYAML is installed
    import yaml as _pyyaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - default path on the sandbox
    _pyyaml = None  # type: ignore


def _strip_comments(line: str) -> str:
    """Remove comments from a YAML line while respecting quoted strings."""
    result_chars: List[str] = []
    in_single = False
    in_double = False
    prev_char = ""

    for char in line:
        if char == "'" and not in_double:
            # toggle single quote status unless escaped
            if prev_char != "\\":
                in_single = not in_single
        elif char == '"' and not in_single:
            if prev_char != "\\":
                in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            break
        result_chars.append(char)
        prev_char = char

    return "".join(result_chars).rstrip()


def _literal_eval(value: str) -> Any:
    """Safely coerce a scalar value to the appropriate Python object."""
    text = value.strip()
    if not text:
        return None

    lowered = text.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


class _SimpleYAMLParser:
    """Very small subset YAML parser that understands mappings and inline lists."""

    def __init__(self, text: str) -> None:
        self._lines = text.splitlines()
        self._index = 0

    def parse(self) -> Dict[str, Any]:
        root: Dict[str, Any] = {}
        stack: List[Tuple[int, Union[Dict[str, Any], List[Any]]]] = [(-1, root)]

        while self._index < len(self._lines):
            raw_line = self._lines[self._index]
            self._index += 1

            cleaned = _strip_comments(raw_line)
            if not cleaned.strip():
                continue

            indent = len(cleaned) - len(cleaned.lstrip(" "))
            if indent % 2 != 0:
                raise ValueError(f"Indentación inválida en YAML en la línea: {raw_line!r}")

            stripped = cleaned.strip()

            while stack and indent <= stack[-1][0]:
                stack.pop()

            if not stack:
                raise ValueError(f"Estructura YAML inválida cerca de: {raw_line!r}")

            container = stack[-1][1]

            if stripped.startswith("- "):
                if not isinstance(container, list):
                    raise ValueError(f"Se esperaba una lista YAML cerca de: {raw_line!r}")
                value = _literal_eval(stripped[2:])
                container.append(value)
                continue

            if ":" not in stripped:
                raise ValueError(f"Formato clave:valor inválido en línea: {raw_line!r}")

            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()

            target_container = stack[-1][1]
            if not isinstance(target_container, dict):
                raise ValueError(f"Se esperaba un diccionario YAML cerca de: {raw_line!r}")

            if value:
                target_container[key] = _literal_eval(value)
            else:
                new_container = self._decide_container(indent)
                target_container[key] = new_container
                stack.append((indent, new_container))

        return root

    def _decide_container(self, current_indent: int) -> Union[Dict[str, Any], List[Any]]:
        for idx in range(self._index, len(self._lines)):
            cleaned = _strip_comments(self._lines[idx])
            if not cleaned.strip():
                continue
            indent = len(cleaned) - len(cleaned.lstrip(" "))
            stripped = cleaned.strip()
            if indent <= current_indent:
                break
            if stripped.startswith("- "):
                return []
            break
        return {}


def safe_load(stream: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML content from a path or text using PyYAML when available."""
    if _pyyaml is not None:  # pragma: no cover - exercised when dependency exists
        if isinstance(stream, (str, Path)):
            with open(stream, "r", encoding="utf-8") as fh:
                return _pyyaml.safe_load(fh)
        return _pyyaml.safe_load(stream)

    if hasattr(stream, "read"):
        text = stream.read()
    else:
        path = Path(stream)
        text = path.read_text(encoding="utf-8")

    parser = _SimpleYAMLParser(text)
    data = parser.parse()
    if not isinstance(data, dict):
        raise ValueError("El archivo YAML debe contener un mapeo en el nivel superior.")
    return data


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Convenience helper used across the project to load the main config file."""
    config = safe_load(Path(path))
    if not isinstance(config, dict):
        raise ValueError("La configuración debe ser un diccionario en la raíz del YAML.")
    return config
