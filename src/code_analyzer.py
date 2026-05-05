"""
Code Analyzer - Uses tree-sitter for robust, multi-language code analysis.

Provides deterministic extraction of:
- Function/method definitions
- Class/struct definitions
- Import statements
- Variable declarations

Supports 100+ languages through tree-sitter-languages.
"""

import logging
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Import tree-sitter
from tree_sitter_languages import get_parser, get_language


# Language extension mapping
EXTENSION_TO_LANGUAGE = {
    # Common languages
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
    "fs": "c_sharp",  # F# uses c_sharp parser as fallback
    "clj": "clojure",
    "lisp": "commonlisp",
    "el": "elisp",
    "vim": "vim",
    "sh": "bash",
    "bash": "bash",
    "zsh": "bash",
    "ps1": "powershell",
    "sql": "sql",
    # Markup/data
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

# Node types for function definitions per language
FUNCTION_NODE_TYPES = {
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

# Node types for class definitions per language
CLASS_NODE_TYPES = {
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


@dataclass
class CodeSymbol:
    """Represents a code symbol (function, class, etc.)"""
    name: str
    kind: str  # "function", "class", "method", "variable", "import"
    start_line: int
    end_line: int
    parent: Optional[str] = None  # Enclosing class/module name
    signature: str = ""  # Full signature if available


@dataclass 
class CodeAnalysisResult:
    """Result of analyzing a code snippet."""
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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "functions": [f.name for f in self.functions],
            "classes": [c.name for c in self.classes],
            "imports": self.imports,
            "variables": [v.name for v in self.variables],
        }


class CodeAnalyzer:
    """
    Analyzes code using tree-sitter for robust, multi-language parsing.
    
    Falls back to regex-based analysis if tree-sitter is not available.
    """
    
    def __init__(self):
        self._parser_cache: Dict[str, Any] = {}
    
    def detect_language(self, filename: str, content: str = "") -> str:
        """
        Detect programming language from filename extension.
        
        Args:
            filename: File name or path
            content: Optional file content for heuristic detection
            
        Returns:
            Language identifier (e.g., "python", "javascript")
        """
        if not filename:
            return self._detect_from_content(content)
        
        # Extract extension
        ext = filename.split(".")[-1].lower() if "." in filename else ""
        
        if ext in EXTENSION_TO_LANGUAGE:
            return EXTENSION_TO_LANGUAGE[ext]
        
        # Fall back to content-based detection
        return self._detect_from_content(content)
    
    def _detect_from_content(self, content: str) -> str:
        """Detect language from content heuristics."""
        if not content:
            return "unknown"
        
        # Simple heuristics
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
    
    def get_language_display_name(self, language: str) -> str:
        """Get human-readable language name."""
        names = {
            "python": "Python",
            "javascript": "JavaScript",
            "typescript": "TypeScript",
            "tsx": "TypeScript/React",
            "java": "Java",
            "c_sharp": "C#",
            "go": "Go",
            "rust": "Rust",
            "ruby": "Ruby",
            "php": "PHP",
            "cpp": "C++",
            "c": "C",
            "swift": "Swift",
            "kotlin": "Kotlin",
            "scala": "Scala",
            "bash": "Shell",
            "powershell": "PowerShell",
            "html": "HTML",
            "css": "CSS",
            "json": "JSON",
            "yaml": "YAML",
            "markdown": "Markdown",
            "sql": "SQL",
        }
        return names.get(language, language.title() if language else "Code")
    
    def analyze(self, code: str, language: str = None, filename: str = None) -> CodeAnalysisResult:
        """
        Analyze code and extract symbols.
        
        Args:
            code: Source code string
            language: Language identifier (optional, will be detected)
            filename: Filename for language detection (optional)
            
        Returns:
            CodeAnalysisResult with extracted symbols
        """
        if not code:
            return CodeAnalysisResult(language=language or "unknown")
        
        # Detect language if not provided
        if not language:
            language = self.detect_language(filename or "", code)
        
        # Use tree-sitter for parsing
        if language != "unknown":
            try:
                return self._analyze_with_treesitter(code, language)
            except Exception as e:
                logger.debug(f"Tree-sitter analysis failed for {language}: {e}")
                # Return empty result on failure
                return CodeAnalysisResult(language=language)
        
        return CodeAnalysisResult(language=language)
    
    def _get_parser(self, language: str):
        """Get or create a parser for the language."""
        if language not in self._parser_cache:
            try:
                self._parser_cache[language] = get_parser(language)
            except Exception as e:
                logger.debug(f"Could not get parser for {language}: {e}")
                return None
        return self._parser_cache[language]
    
    def _analyze_with_treesitter(self, code: str, language: str) -> CodeAnalysisResult:
        """Analyze code using tree-sitter."""
        parser = self._get_parser(language)
        if not parser:
            return CodeAnalysisResult(language=language)
        
        tree = parser.parse(code.encode())
        result = CodeAnalysisResult(language=language)
        
        # Get node types for this language
        func_types = FUNCTION_NODE_TYPES.get(language, [])
        class_types = CLASS_NODE_TYPES.get(language, [])
        
        # Walk the tree
        self._walk_tree(tree.root_node, result, func_types, class_types, language)
        
        # Extract imports
        result.imports = self._extract_imports(tree.root_node, language)
        
        return result
    
    def _walk_tree(self, node, result: CodeAnalysisResult, func_types: List[str], 
                   class_types: List[str], language: str, parent_class: str = None):
        """Recursively walk the AST and extract symbols."""
        node_type = node.type
        
        # Check for function definitions
        if node_type in func_types:
            name = self._extract_name(node, language)
            if name:
                symbol = CodeSymbol(
                    name=name,
                    kind="method" if parent_class else "function",
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    parent=parent_class
                )
                result.functions.append(symbol)
        
        # Check for class definitions
        elif node_type in class_types:
            name = self._extract_name(node, language)
            if name:
                symbol = CodeSymbol(
                    name=name,
                    kind="class",
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1
                )
                result.classes.append(symbol)
                # Recurse with this as parent class
                for child in node.children:
                    self._walk_tree(child, result, func_types, class_types, language, name)
                return  # Don't recurse again below
        
        # Recurse into children
        for child in node.children:
            self._walk_tree(child, result, func_types, class_types, language, parent_class)
    
    def _extract_name(self, node, language: str) -> Optional[str]:
        """Extract the name from a function/class definition node."""
        # Look for identifier child
        for child in node.children:
            if child.type == "identifier" or child.type == "name":
                return child.text.decode()
            # For property identifiers (methods)
            if child.type == "property_identifier":
                return child.text.decode()
            # For type identifiers (Go, Rust)
            if child.type == "type_identifier":
                return child.text.decode()
        
        # Some languages have nested structures
        for child in node.children:
            if child.type in ("declarator", "function_declarator"):
                return self._extract_name(child, language)
        
        return None
    
    def _extract_imports(self, root_node, language: str) -> List[str]:
        """Extract import statements."""
        imports = []
        
        import_types = {
            "python": ["import_statement", "import_from_statement"],
            "javascript": ["import_statement"],
            "typescript": ["import_statement"],
            "java": ["import_declaration"],
            "go": ["import_declaration", "import_spec"],
            "rust": ["use_declaration"],
        }
        
        types_to_find = import_types.get(language, [])
        
        def walk(node):
            if node.type in types_to_find:
                imports.append(node.text.decode().strip())
            for child in node.children:
                walk(child)
        
        walk(root_node)
        return imports
    
    def compare_code(self, old_code: str, new_code: str, filename: str = None) -> Dict[str, Any]:
        """
        Compare two code snippets and describe the changes.
        
        Returns a dict with:
        - added_functions: List of new functions
        - removed_functions: List of removed functions
        - renamed_functions: List of (old_name, new_name) tuples
        - added_classes: List of new classes
        - removed_classes: List of removed classes
        - renamed_classes: List of (old_name, new_name) tuples
        - imports_changed: Boolean
        - description: Human-readable change description
        """
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
        
        # Detect renames (1 removed + 1 added with similar context)
        renamed_funcs = []
        if len(removed_funcs) == 1 and len(added_funcs) == 1:
            renamed_funcs = [(removed_funcs.pop(), added_funcs.pop())]
            removed_funcs = set()
            added_funcs = set()
        
        renamed_classes = []
        if len(removed_classes) == 1 and len(added_classes) == 1:
            renamed_classes = [(removed_classes.pop(), added_classes.pop())]
            removed_classes = set()
            added_classes = set()
        
        imports_changed = set(old_result.imports) != set(new_result.imports)
        
        # Build description
        descriptions = []
        
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
            # Fallback: describe size change
            old_len = len(old_code)
            new_len = len(new_code)
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
        """
        Generate a description for a file creation operation.
        
        Returns a human-readable description like:
        "Created Python file 'main.py' with functions: main, helper"
        """
        language = self.detect_language(filename, content)
        display_lang = self.get_language_display_name(language)
        
        # Analyze the content
        result = self.analyze(content, language, filename)
        
        # Get just the filename
        simple_name = filename.split("/")[-1].split("\\")[-1]
        
        # Build description
        parts = []
        
        func_names = result.get_function_names()
        class_names = result.get_class_names()
        
        if func_names or class_names:
            parts.append(f"Created {display_lang} file '{simple_name}'")
            
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
        else:
            return f"Created {display_lang} file '{simple_name}'"


# Global analyzer instance
_analyzer = None

def get_analyzer() -> CodeAnalyzer:
    """Get or create the global code analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = CodeAnalyzer()
    return _analyzer


def analyze_code(code: str, filename: str = None) -> CodeAnalysisResult:
    """Convenience function to analyze code."""
    return get_analyzer().analyze(code, filename=filename)


def compare_code(old_code: str, new_code: str, filename: str = None) -> Dict[str, Any]:
    """Convenience function to compare code."""
    return get_analyzer().compare_code(old_code, new_code, filename)


def describe_file_creation(content: str, filename: str) -> str:
    """Convenience function to describe file creation."""
    return get_analyzer().describe_file_creation(content, filename)


# Test the analyzer
if __name__ == "__main__":
    # Test Python code
    python_code = '''
def hello_world():
    print("Hello, World!")

class Calculator:
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b

def main():
    calc = Calculator()
    print(calc.add(1, 2))
'''
    
    result = analyze_code(python_code, "example.py")
    print(f"\nPython analysis:")
    print(f"  Language: {result.language}")
    print(f"  Functions: {result.get_function_names()}")
    print(f"  Classes: {result.get_class_names()}")
    
    # Test JavaScript code
    js_code = '''
function greet(name) {
    return `Hello, ${name}!`;
}

const multiply = (a, b) => a * b;

class Person {
    constructor(name) {
        this.name = name;
    }
    
    sayHello() {
        return greet(this.name);
    }
}
'''
    
    result = analyze_code(js_code, "example.js")
    print(f"\nJavaScript analysis:")
    print(f"  Language: {result.language}")
    print(f"  Functions: {result.get_function_names()}")
    print(f"  Classes: {result.get_class_names()}")
    
    # Test comparison
    old_code = "def add_numbers(a, b):\n    return a + b"
    new_code = "def perform_add(a, b):\n    return a + b"
    
    comparison = compare_code(old_code, new_code, "math.py")
    print(f"\nCode comparison:")
    print(f"  Description: {comparison['description']}")
    print(f"  Renamed functions: {comparison['renamed_functions']}")
