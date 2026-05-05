"""Tree-sitter based code analyser for robust, multi-language symbol extraction.

This is an internal module — not part of the public API surface.  It provides
deterministic extraction of function/method/class definitions and import
statements, used by the generator to populate ``content_description`` and
scope fields on :class:`~swe_trace_sdk.models.State`.

Supports 100+ languages through ``tree-sitter-languages``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

try:
    from tree_sitter_languages import get_parser, get_language  # noqa: F401

    _TREE_SITTER_AVAILABLE = True
except ImportError:
    _TREE_SITTER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Language mapping
# ---------------------------------------------------------------------------

EXTENSION_TO_LANGUAGE: Dict[str, str] = {
    "py": "python",
    "js": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "tsx": "tsx",
    "java": "java",
    "cs": "c_sharp",
    "go": "go",
    "rs": "rust",
    "rb": "ruby",
    "php": "php",
    "cpp": "cpp",
    "cc": "cpp",
    "cxx": "cpp",
    "c": "c",
    "h": "c",
    "hpp": "cpp",
    "swift": "swift",
    "kt": "kotlin",
    "kts": "kotlin",
    "scala": "scala",
    "lua": "lua",
    "r": "r",
    "jl": "julia",
    "ex": "elixir",
    "exs": "elixir",
    "erl": "erlang",
    "hs": "haskell",
    "ml": "ocaml",
    "fs": "c_sharp",
    "clj": "clojure",
    "lisp": "commonlisp",
    "el": "elisp",
    "vim": "vim",
    "sh": "bash",
    "bash": "bash",
    "zsh": "bash",
    "ps1": "powershell",
    "sql": "sql",
    "html": "html",
    "htm": "html",
    "xml": "xml",
    "css": "css",
    "scss": "scss",
    "json": "json",
    "yaml": "yaml",
    "yml": "yaml",
    "toml": "toml",
    "md": "markdown",
    "markdown": "markdown",
}

FUNCTION_NODE_TYPES: Dict[str, List[str]] = {
    "python": ["function_definition", "async_function_definition"],
    "javascript": ["function_declaration", "function_expression", "arrow_function", "method_definition"],
    "typescript": ["function_declaration", "function_expression", "arrow_function", "method_definition"],
    "tsx": ["function_declaration", "function_expression", "arrow_function", "method_definition"],
    "java": ["method_declaration", "constructor_declaration"],
    "c_sharp": ["method_declaration", "constructor_declaration"],
    "go": ["function_declaration", "method_declaration"],
    "rust": ["function_item", "impl_item"],
    "ruby": ["method", "singleton_method"],
    "php": ["function_definition", "method_declaration"],
    "cpp": ["function_definition", "function_declaration"],
    "c": ["function_definition", "function_declaration"],
}

CLASS_NODE_TYPES: Dict[str, List[str]] = {
    "python": ["class_definition"],
    "javascript": ["class_declaration", "class"],
    "typescript": ["class_declaration", "class"],
    "tsx": ["class_declaration", "class"],
    "java": ["class_declaration", "interface_declaration", "enum_declaration"],
    "c_sharp": ["class_declaration", "interface_declaration", "struct_declaration"],
    "go": ["type_declaration"],
    "rust": ["struct_item", "enum_item", "impl_item", "trait_item"],
    "ruby": ["class", "module"],
    "php": ["class_declaration", "interface_declaration", "trait_declaration"],
    "cpp": ["class_specifier", "struct_specifier"],
    "c": ["struct_specifier", "enum_specifier"],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CodeSymbol:
    """Represents a code symbol (function, class, etc.)."""

    name: str
    kind: str  # "function" | "class" | "method" | "variable" | "import"
    start_line: int
    end_line: int
    parent: Optional[str] = None
    signature: str = ""


@dataclass
class CodeAnalysisResult:
    """Result of analysing a code snippet."""

    language: str
    functions: List[CodeSymbol] = field(default_factory=list)
    classes: List[CodeSymbol] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    variables: List[CodeSymbol] = field(default_factory=list)

    def get_function_names(self) -> Set[str]:
        return {f.name for f in self.functions}

    def get_class_names(self) -> Set[str]:
        return {c.name for c in self.classes}

    def get_all_symbol_names(self) -> Set[str]:
        names = self.get_function_names() | self.get_class_names()
        names |= {v.name for v in self.variables}
        return names


# ---------------------------------------------------------------------------
# Analyser
# ---------------------------------------------------------------------------

class CodeAnalyzer:
    """Analyses code using tree-sitter for robust, multi-language parsing."""

    def __init__(self) -> None:
        self._parser_cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Language detection
    # ------------------------------------------------------------------

    def detect_language(self, filename: str, content: str = "") -> str:
        if not filename:
            return self._detect_from_content(content)
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext in EXTENSION_TO_LANGUAGE:
            return EXTENSION_TO_LANGUAGE[ext]
        return self._detect_from_content(content)

    @staticmethod
    def _detect_from_content(content: str) -> str:
        if not content:
            return "unknown"
        if "#!/usr/bin/env python" in content or "#!/usr/bin/python" in content:
            return "python"
        if "#!/bin/bash" in content or "#!/usr/bin/env bash" in content:
            return "bash"
        if "package main" in content and "func " in content:
            return "go"
        if "fn main()" in content or "impl " in content:
            return "rust"
        if "public class " in content or "public static void main" in content:
            return "java"
        if "def " in content and ":" in content:
            return "python"
        if "function " in content or "const " in content or "=>" in content:
            return "javascript"
        return "unknown"

    @staticmethod
    def get_language_display_name(language: str) -> str:
        names = {
            "python": "Python", "javascript": "JavaScript", "typescript": "TypeScript",
            "tsx": "TypeScript/React", "java": "Java", "c_sharp": "C#", "go": "Go",
            "rust": "Rust", "ruby": "Ruby", "php": "PHP", "cpp": "C++", "c": "C",
            "swift": "Swift", "kotlin": "Kotlin", "scala": "Scala", "bash": "Shell",
            "powershell": "PowerShell", "html": "HTML", "css": "CSS", "json": "JSON",
            "yaml": "YAML", "markdown": "Markdown", "sql": "SQL",
        }
        return names.get(language, language.title() if language else "Code")

    # ------------------------------------------------------------------
    # Main analysis entry
    # ------------------------------------------------------------------

    def analyze(self, code: str, language: str | None = None, filename: str | None = None) -> CodeAnalysisResult:
        if not code:
            return CodeAnalysisResult(language=language or "unknown")
        if not language:
            language = self.detect_language(filename or "", code)
        if language != "unknown" and _TREE_SITTER_AVAILABLE:
            try:
                return self._analyze_with_treesitter(code, language)
            except Exception as exc:
                logger.debug("Tree-sitter analysis failed for %s: %s", language, exc)
        return CodeAnalysisResult(language=language)

    # ------------------------------------------------------------------
    # Tree-sitter internals
    # ------------------------------------------------------------------

    def _get_parser(self, language: str):
        if language not in self._parser_cache:
            try:
                self._parser_cache[language] = get_parser(language)
            except Exception as exc:
                logger.debug("Could not get parser for %s: %s", language, exc)
                return None
        return self._parser_cache[language]

    def _analyze_with_treesitter(self, code: str, language: str) -> CodeAnalysisResult:
        parser = self._get_parser(language)
        if not parser:
            return CodeAnalysisResult(language=language)
        tree = parser.parse(code.encode())
        result = CodeAnalysisResult(language=language)
        func_types = FUNCTION_NODE_TYPES.get(language, [])
        class_types = CLASS_NODE_TYPES.get(language, [])
        self._walk_tree(tree.root_node, result, func_types, class_types, language)
        result.imports = self._extract_imports(tree.root_node, language)
        return result

    def _walk_tree(
        self,
        node,
        result: CodeAnalysisResult,
        func_types: List[str],
        class_types: List[str],
        language: str,
        parent_class: str | None = None,
    ) -> None:
        node_type = node.type
        if node_type in func_types:
            name = self._extract_name(node, language)
            if name:
                result.functions.append(CodeSymbol(
                    name=name,
                    kind="method" if parent_class else "function",
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    parent=parent_class,
                ))
        elif node_type in class_types:
            name = self._extract_name(node, language)
            if name:
                result.classes.append(CodeSymbol(
                    name=name, kind="class",
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                ))
                for child in node.children:
                    self._walk_tree(child, result, func_types, class_types, language, name)
                return
        for child in node.children:
            self._walk_tree(child, result, func_types, class_types, language, parent_class)

    @staticmethod
    def _extract_name(node, language: str) -> Optional[str]:
        for child in node.children:
            if child.type in ("identifier", "name", "property_identifier", "type_identifier"):
                return child.text.decode()
        for child in node.children:
            if child.type in ("declarator", "function_declarator"):
                return CodeAnalyzer._extract_name(child, language)
        return None

    @staticmethod
    def _extract_imports(root_node, language: str) -> List[str]:
        import_types: Dict[str, List[str]] = {
            "python": ["import_statement", "import_from_statement"],
            "javascript": ["import_statement"],
            "typescript": ["import_statement"],
            "java": ["import_declaration"],
            "go": ["import_declaration", "import_spec"],
            "rust": ["use_declaration"],
        }
        types_to_find = import_types.get(language, [])
        imports: List[str] = []

        def walk(node):
            if node.type in types_to_find:
                imports.append(node.text.decode().strip())
            for child in node.children:
                walk(child)

        walk(root_node)
        return imports

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------

    def compare_code(self, old_code: str, new_code: str, filename: str | None = None) -> Dict[str, Any]:
        """Compare two code snippets and return a change description dict."""
        old_result = self.analyze(old_code, filename=filename)
        new_result = self.analyze(new_code, filename=filename)

        old_funcs = old_result.get_function_names()
        new_funcs = new_result.get_function_names()
        old_classes = old_result.get_class_names()
        new_classes = new_result.get_class_names()

        added_funcs = new_funcs - old_funcs
        removed_funcs = old_funcs - new_funcs
        added_classes = new_classes - old_classes
        removed_classes = old_classes - new_classes

        renamed_funcs: list = []
        if len(removed_funcs) == 1 and len(added_funcs) == 1:
            renamed_funcs = [(removed_funcs.pop(), added_funcs.pop())]
            removed_funcs = set()
            added_funcs = set()

        renamed_classes: list = []
        if len(removed_classes) == 1 and len(added_classes) == 1:
            renamed_classes = [(removed_classes.pop(), added_classes.pop())]
            removed_classes = set()
            added_classes = set()

        imports_changed = set(old_result.imports) != set(new_result.imports)

        descriptions: List[str] = []
        for old_name, new_name in renamed_funcs:
            descriptions.append(f"renamed function '{old_name}' to '{new_name}'")
        for old_name, new_name in renamed_classes:
            descriptions.append(f"renamed class '{old_name}' to '{new_name}'")
        for func in added_funcs:
            descriptions.append(f"added function '{func}'")
        for func in removed_funcs:
            descriptions.append(f"removed function '{func}'")
        for cls in added_classes:
            descriptions.append(f"added class '{cls}'")
        for cls in removed_classes:
            descriptions.append(f"removed class '{cls}'")
        if imports_changed and not descriptions:
            descriptions.append("modified imports")
        if not descriptions:
            old_len, new_len = len(old_code), len(new_code)
            if new_len > old_len * 1.2:
                descriptions.append("added content")
            elif new_len < old_len * 0.8:
                descriptions.append("removed content")
            else:
                descriptions.append("modified content")

        return {
            "language": new_result.language,
            "added_functions": list(added_funcs),
            "removed_functions": list(removed_funcs),
            "renamed_functions": renamed_funcs,
            "added_classes": list(added_classes),
            "removed_classes": list(removed_classes),
            "renamed_classes": renamed_classes,
            "imports_changed": imports_changed,
            "description": "; ".join(descriptions),
        }

    def describe_file_creation(self, content: str, filename: str) -> str:
        """Return a human-readable description of a file creation."""
        language = self.detect_language(filename, content)
        display_lang = self.get_language_display_name(language)
        result = self.analyze(content, language, filename)
        simple_name = filename.split("/")[-1].split("\\")[-1]
        func_names = result.get_function_names()
        class_names = result.get_class_names()

        if func_names or class_names:
            parts = [f"Created {display_lang} file '{simple_name}'"]
            if class_names:
                if len(class_names) <= 3:
                    parts.append(f"with classes: {', '.join(sorted(class_names))}")
                else:
                    parts.append(f"with {len(class_names)} classes")
            if func_names:
                if len(func_names) <= 3:
                    parts.append(f"functions: {', '.join(sorted(func_names))}")
                else:
                    parts.append(f"{len(func_names)} functions")
            return " ".join(parts)
        return f"Created {display_lang} file '{simple_name}'"


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_analyzer: Optional[CodeAnalyzer] = None


def get_analyzer() -> CodeAnalyzer:
    """Return (or create) the global :class:`CodeAnalyzer` singleton."""
    global _analyzer
    if _analyzer is None:
        _analyzer = CodeAnalyzer()
    return _analyzer
