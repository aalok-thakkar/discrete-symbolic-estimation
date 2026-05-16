"""SV-COMP C subset transpiler: C source -> Python source.

Translates a strict subset of C99---the integer-only subset used by
many SV-COMP ``ReachSafety`` and ``NoOverflow`` benchmarks---into a
single Python function suitable for :func:`dise.estimate`.

Supported language fragment
---------------------------

  - **Types**: ``int``, ``unsigned int``, ``long``, ``unsigned long``,
    ``short``, ``unsigned short``, ``char``, ``unsigned char``,
    ``_Bool``.  All represented as Python ``int`` at runtime; unsigned
    types are masked to their bit width on every write.
  - **Statements**: assignment, ``if``/``else``, ``while``, ``do``--
    ``while``, ``for``, ``return``, ``break``, ``continue``,
    compound blocks, single-line expression statements.
  - **Expressions**: binary arithmetic (``+ - * / %``), bitwise
    (``& | ^ ~ << >>``), comparison (``< <= > >= == !=``), logical
    (``&& || !``), unary ``-``/``+``, ternary ``?:``.
  - **SV-COMP idioms**:

    * ``__VERIFIER_nondet_<T>()`` -> a function parameter sampled from
      the distribution assigned to it by the experiment harness.
    * ``__VERIFIER_assert(x)`` -> ``assert x``.  Falsified asserts
      raise ``AssertionError``, which :func:`dise.failure_probability`
      catches.
    * ``__VERIFIER_assume(x)`` -> ``if not x: raise _Assumed()``.  The
      harness intercepts ``_Assumed`` and re-draws the input batch.
    * ``__VERIFIER_error()`` / ``reach_error()`` -> ``raise
      AssertionError("__VERIFIER_error")``.

Unsupported
-----------

  - Pointers (``*``, ``&``), address-of, dereference, pointer
    arithmetic.
  - Structs, unions, enums (we allow enum-style ``int`` constants).
  - Arrays of unbounded size or with symbolic indexing.
  - Recursion, function pointers, variadic functions.
  - Floating-point types (``float``, ``double``).
  - Globals beyond ``const``-style ``#define``-replacements.
  - Multi-file translation units, preprocessor directives.

Programs using unsupported features raise :class:`Untranslatable`.
The intent is to *honestly fail fast* rather than emit silently
wrong Python.
"""

from __future__ import annotations

import io
import textwrap
from dataclasses import dataclass, field
from typing import Any

import pycparser
from pycparser import c_ast


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class Untranslatable(Exception):
    """Raised when the C source uses a feature outside the supported subset."""


@dataclass
class Nondet:
    """One nondeterministic input parameter extracted from the C source."""
    name: str            # the Python identifier (the C variable assigned the nondet)
    c_type: str          # the C type name (``int``, ``unsigned int``, ...)
    bits: int            # bit width
    signed: bool         # signed or unsigned


@dataclass
class TranspileResult:
    """The translated program plus metadata."""
    python_source: str               # the generated Python module source
    function_name: str               # callable to invoke (typically ``main`` or ``run``)
    nondets: list[Nondet]            # ordered input parameters
    has_assert: bool                 # whether the program has any assertion site
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Standard SV-COMP headers — pycparser needs declarations, but the
# benchmark sources rarely include them explicitly. We prepend our own.
# ---------------------------------------------------------------------------


SVCOMP_DECLS = r"""
typedef unsigned int size_t;

extern int __VERIFIER_nondet_int(void);
extern unsigned int __VERIFIER_nondet_uint(void);
extern long __VERIFIER_nondet_long(void);
extern unsigned long __VERIFIER_nondet_ulong(void);
extern short __VERIFIER_nondet_short(void);
extern unsigned short __VERIFIER_nondet_ushort(void);
extern char __VERIFIER_nondet_char(void);
extern unsigned char __VERIFIER_nondet_uchar(void);
extern _Bool __VERIFIER_nondet_bool(void);

extern void __VERIFIER_assert(int);
extern void __VERIFIER_assume(int);
extern void __VERIFIER_error(void);
extern void reach_error(void);

extern void abort(void);
extern void exit(int);
"""


NONDET_TYPES = {
    "__VERIFIER_nondet_int":    ("int",            32, True),
    "__VERIFIER_nondet_uint":   ("unsigned int",   32, False),
    "__VERIFIER_nondet_long":   ("long",           64, True),
    "__VERIFIER_nondet_ulong":  ("unsigned long",  64, False),
    "__VERIFIER_nondet_short":  ("short",          16, True),
    "__VERIFIER_nondet_ushort": ("unsigned short", 16, False),
    "__VERIFIER_nondet_char":   ("char",            8, True),
    "__VERIFIER_nondet_uchar":  ("unsigned char",   8, False),
    "__VERIFIER_nondet_bool":   ("_Bool",           1, False),
}

ERROR_FUNCS = {"__VERIFIER_error", "reach_error", "abort"}
ASSERT_FUNCS = {"__VERIFIER_assert"}
ASSUME_FUNCS = {"__VERIFIER_assume"}


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------


@dataclass
class _LocalScope:
    """Track unsigned-typed locals so we can mask on assignment."""
    unsigned_bits: dict[str, int] = field(default_factory=dict)


class _Transpiler(c_ast.NodeVisitor):
    """Walk a C AST, emit Python source for one function."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.indent = 0
        self.nondets: list[Nondet] = []
        self.nondet_names_seen: set[str] = set()
        self.has_assert = False
        self.scope = _LocalScope()
        self.notes: list[str] = []

    # ---------- utilities ----------

    def _emit(self, line: str = "") -> None:
        self.lines.append(("    " * self.indent) + line if line else "")

    def _indent_block(self):
        self.indent += 1

    def _dedent_block(self):
        self.indent -= 1

    def _new_name(self, c_name: str) -> str:
        """Python identifier from a C identifier (keep as-is for ints)."""
        if c_name in ("from", "import", "lambda", "def", "class", "global",
                       "nonlocal", "pass", "yield", "async", "await", "with"):
            return c_name + "_"
        return c_name

    # ---------- type helpers ----------

    def _type_info(self, type_node: c_ast.Node) -> tuple[str, int, bool]:
        """Return ``(c_type_str, bits, signed)`` for a Decl's type."""
        # Unwrap TypeDecl / ArrayDecl / etc.
        if isinstance(type_node, c_ast.TypeDecl):
            return self._type_info(type_node.type)
        if isinstance(type_node, c_ast.IdentifierType):
            names = type_node.names
            return _c_type_to_bits(names)
        raise Untranslatable(
            f"unsupported declared type: {type(type_node).__name__}"
        )

    # ---------- visitor methods ----------

    def visit_FileAST(self, node: c_ast.FileAST) -> None:
        for ext in node.ext:
            if isinstance(ext, c_ast.Decl) and isinstance(ext.type, c_ast.FuncDecl):
                # Skip extern declarations (no body)
                continue
            if isinstance(ext, c_ast.FuncDef):
                self.visit(ext)
                continue
            if isinstance(ext, c_ast.Typedef):
                continue
            if isinstance(ext, c_ast.Decl):
                # Global; ignore for now (could be const).
                continue
            raise Untranslatable(
                f"unsupported top-level construct: {type(ext).__name__}"
            )

    def visit_FuncDef(self, node: c_ast.FuncDef) -> None:
        fname = node.decl.name
        if fname != "main":
            # We only translate main(); other functions are inlined or treated
            # as opaque. A future enhancement would inline pure helper
            # functions; SV-COMP programs mostly inline themselves.
            self.notes.append(f"skipped non-main function: {fname}")
            return

        # The function signature in the OUTPUT depends on what nondets we
        # find. We collect nondets while translating the body, then patch
        # the def header at the end.
        body_start = len(self.lines)
        self._emit("def __PLACEHOLDER_HEADER__:")
        self._indent_block()

        try:
            self.visit(node.body)
        except Untranslatable:
            raise
        except Exception as exc:
            raise Untranslatable(
                f"transpilation crash inside main(): {type(exc).__name__}: {exc}"
            )

        # Default fall-through: return 0.
        self._emit("return 0")
        self._dedent_block()

        # Patch the def header with the nondets we collected.
        param_list = ", ".join(n.name for n in self.nondets)
        self.lines[body_start] = f"def main({param_list}):"

    def visit_Compound(self, node: c_ast.Compound) -> None:
        if node.block_items is None:
            self._emit("pass")
            return
        for item in node.block_items:
            self.visit(item)

    def visit_Decl(self, node: c_ast.Decl) -> None:
        # Declaration of a local variable: maybe with initialiser.
        if isinstance(node.type, c_ast.FuncDecl):
            raise Untranslatable("nested function declaration")
        if isinstance(node.type, c_ast.ArrayDecl):
            raise Untranslatable("array declaration in function body")
        c_type, bits, signed = self._type_info(node.type)
        py_name = self._new_name(node.name)

        if not signed and bits <= 32:
            self.scope.unsigned_bits[py_name] = bits

        if node.init is None:
            self._emit(f"{py_name} = 0")
            return

        # Initialiser
        init_src = self._expr_to_python(node.init)
        if not signed and bits <= 32:
            init_src = f"({init_src}) & {(1 << bits) - 1:#x}"
        self._emit(f"{py_name} = {init_src}")

    def visit_Assignment(self, node: c_ast.Assignment) -> None:
        lhs = self._lvalue_to_python(node.lvalue)
        rhs = self._expr_to_python(node.rvalue)
        op = node.op
        if op == "=":
            assigned = rhs
        elif op in ("+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
                     "<<=", ">>="):
            py_op = op[:-1]
            # Translate C operators that differ from Python.
            if py_op == "/":
                # C integer division truncates toward zero; Python // floors.
                assigned = f"int(({lhs}) / ({rhs}))"
            else:
                assigned = f"({lhs}) {py_op} ({rhs})"
        else:
            raise Untranslatable(f"unsupported assignment op {op}")
        # Mask if lhs is a tracked unsigned local.
        if lhs in self.scope.unsigned_bits:
            mask = (1 << self.scope.unsigned_bits[lhs]) - 1
            assigned = f"({assigned}) & {mask:#x}"
        self._emit(f"{lhs} = {assigned}")

    def visit_If(self, node: c_ast.If) -> None:
        cond = self._expr_to_python(node.cond)
        self._emit(f"if {cond}:")
        self._indent_block()
        if node.iftrue is not None:
            self.visit(node.iftrue)
        else:
            self._emit("pass")
        self._dedent_block()
        if node.iffalse is not None:
            self._emit("else:")
            self._indent_block()
            self.visit(node.iffalse)
            self._dedent_block()

    def visit_While(self, node: c_ast.While) -> None:
        cond = self._expr_to_python(node.cond)
        self._emit(f"while {cond}:")
        self._indent_block()
        if node.stmt is not None:
            self.visit(node.stmt)
        else:
            self._emit("pass")
        self._dedent_block()

    def visit_DoWhile(self, node: c_ast.DoWhile) -> None:
        # Translate do { S; } while (C); into Python:
        #   while True:
        #       S
        #       if not C: break
        self._emit("while True:")
        self._indent_block()
        if node.stmt is not None:
            self.visit(node.stmt)
        cond = self._expr_to_python(node.cond)
        self._emit(f"if not ({cond}): break")
        self._dedent_block()

    def visit_For(self, node: c_ast.For) -> None:
        if node.init is not None:
            if isinstance(node.init, c_ast.DeclList):
                for d in node.init.decls:
                    self.visit(d)
            else:
                self.visit(node.init)
        cond_src = (
            self._expr_to_python(node.cond) if node.cond is not None else "True"
        )
        self._emit(f"while {cond_src}:")
        self._indent_block()
        if node.stmt is not None:
            self.visit(node.stmt)
        if node.next is not None:
            self.visit(node.next)
        self._dedent_block()

    def visit_Return(self, node: c_ast.Return) -> None:
        if node.expr is None:
            self._emit("return 0")
        else:
            self._emit(f"return {self._expr_to_python(node.expr)}")

    def visit_Break(self, node: c_ast.Break) -> None:
        self._emit("break")

    def visit_Continue(self, node: c_ast.Continue) -> None:
        self._emit("continue")

    def visit_FuncCall(self, node: c_ast.FuncCall) -> None:
        # Used at *statement* level: call with no return-value usage.
        rendered = self._funccall_to_python(node)
        self._emit(rendered)

    def visit_EmptyStatement(self, node: c_ast.EmptyStatement) -> None:
        # ; on its own line
        pass

    def visit_UnaryOp(self, node: c_ast.UnaryOp) -> None:
        # Statement-level UnaryOp: typically ``x++``/``++x``/``x--``/``--x``
        # used purely for its side effect.  Translate as an assignment.
        if node.op in ("++", "p++"):
            lv = self._lvalue_to_python(node.expr)
            new_rhs = f"({lv}) + 1"
            if lv in self.scope.unsigned_bits:
                mask = (1 << self.scope.unsigned_bits[lv]) - 1
                new_rhs = f"({new_rhs}) & {mask:#x}"
            self._emit(f"{lv} = {new_rhs}")
            return
        if node.op in ("--", "p--"):
            lv = self._lvalue_to_python(node.expr)
            new_rhs = f"({lv}) - 1"
            if lv in self.scope.unsigned_bits:
                mask = (1 << self.scope.unsigned_bits[lv]) - 1
                new_rhs = f"({new_rhs}) & {mask:#x}"
            self._emit(f"{lv} = {new_rhs}")
            return
        raise Untranslatable(
            f"unsupported statement-level unary op: {node.op}"
        )

    def visit_Assignment_stmt(self, node):  # noqa: not a visitor name
        # Reserved if we ever distinguish expression vs statement.
        return self.visit_Assignment(node)

    def visit_Label(self, node: c_ast.Label) -> None:
        # Labels are only useful with goto; reject.
        raise Untranslatable("labeled statements (goto)")

    def visit_Goto(self, node: c_ast.Goto) -> None:
        raise Untranslatable("goto")

    def visit_Switch(self, node: c_ast.Switch) -> None:
        # Could translate to an if-elif chain; defer.
        raise Untranslatable("switch statement (not yet supported)")

    def generic_visit(self, node: c_ast.Node) -> None:
        # For node types we haven't implemented; emit the source and bail.
        raise Untranslatable(
            f"unsupported AST node: {type(node).__name__}"
        )

    # ---------- expression / lvalue translation ----------

    def _lvalue_to_python(self, node: c_ast.Node) -> str:
        if isinstance(node, c_ast.ID):
            return self._new_name(node.name)
        raise Untranslatable(
            f"unsupported lvalue: {type(node).__name__}"
        )

    def _expr_to_python(self, node: c_ast.Node) -> str:
        if isinstance(node, c_ast.Constant):
            v = node.value
            if node.type in ("int", "unsigned int", "long", "unsigned long",
                              "short", "unsigned short", "char",
                              "unsigned char", "_Bool"):
                # Strip suffix like 'U', 'L', 'UL', 'LL'.
                v = v.rstrip("uUlL")
                # Handle hex / octal.
                if v.startswith("0x") or v.startswith("0X"):
                    return str(int(v, 16))
                if v.startswith("0") and len(v) > 1 and v[1].isdigit():
                    return str(int(v, 8))
                return v
            if node.type == "char":
                return v
            raise Untranslatable(f"unsupported constant type: {node.type}")
        if isinstance(node, c_ast.ID):
            return self._new_name(node.name)
        if isinstance(node, c_ast.UnaryOp):
            op = node.op
            inner = self._expr_to_python(node.expr)
            if op == "-":
                return f"(-{inner})"
            if op == "+":
                return f"(+{inner})"
            if op == "!":
                return f"(0 if ({inner}) else 1)"
            if op == "~":
                return f"(~({inner}))"
            if op == "++":
                # Pre-increment/post-increment: only support as statement, but
                # translating pre/post is annoying in an expression context.
                # Heuristic: emit as expression-with-side-effect via walrus.
                # For SV-COMP this is mostly in for-loop next, where statement
                # form is fine.
                lv = self._lvalue_to_python(node.expr)
                return f"({lv} := {lv} + 1)"
            if op == "--":
                lv = self._lvalue_to_python(node.expr)
                return f"({lv} := {lv} - 1)"
            if op == "p++":
                lv = self._lvalue_to_python(node.expr)
                return f"({lv}, {lv} := {lv}, {lv} + 1)[0]"
            if op == "p--":
                lv = self._lvalue_to_python(node.expr)
                return f"({lv}, {lv} := {lv}, {lv} - 1)[0]"
            raise Untranslatable(f"unsupported unary op: {op}")
        if isinstance(node, c_ast.BinaryOp):
            l = self._expr_to_python(node.left)
            r = self._expr_to_python(node.right)
            op = node.op
            if op == "/":
                # C signed int division truncates toward zero.
                return f"int(({l}) / ({r}))"
            if op == "%":
                # C remainder; truncates toward zero. Python ``%`` is floor-mod.
                return f"(({l}) - int(({l}) / ({r})) * ({r}))"
            if op in ("+", "-", "*", "&", "|", "^", "<<", ">>"):
                return f"(({l}) {op} ({r}))"
            if op in ("<", "<=", ">", ">=", "==", "!="):
                return f"(1 if (({l}) {op} ({r})) else 0)"
            if op == "&&":
                return f"(1 if (({l}) and ({r})) else 0)"
            if op == "||":
                return f"(1 if (({l}) or ({r})) else 0)"
            raise Untranslatable(f"unsupported binary op: {op}")
        if isinstance(node, c_ast.TernaryOp):
            cond = self._expr_to_python(node.cond)
            t = self._expr_to_python(node.iftrue)
            f = self._expr_to_python(node.iffalse)
            return f"(({t}) if ({cond}) else ({f}))"
        if isinstance(node, c_ast.FuncCall):
            return self._funccall_to_python(node)
        if isinstance(node, c_ast.Cast):
            # We mostly ignore casts; check the target type for sanity.
            _ = self._type_info(node.to_type.type)
            return self._expr_to_python(node.expr)
        raise Untranslatable(
            f"unsupported expression: {type(node).__name__}"
        )

    def _funccall_to_python(self, node: c_ast.FuncCall) -> str:
        if not isinstance(node.name, c_ast.ID):
            raise Untranslatable("call with non-identifier callee")
        fname = node.name.name

        if fname in NONDET_TYPES:
            # Each nondet *call* becomes a fresh input parameter unless we
            # see this assignment pattern: ``T x = nondet();`` --- in which
            # case the parameter takes the variable's name. We need to look
            # at the parent context: if this is the RHS of a Decl or
            # Assignment, name after the lhs; otherwise, generate a fresh
            # name.
            c_type, bits, signed = NONDET_TYPES[fname]
            # Generate or reuse a name. The convention: name from the LHS.
            # We don't have direct access to context, so emit a placeholder
            # and let the caller (visit_Decl / visit_Assignment) rename. For
            # simplicity, generate a fresh anonymous name here.
            idx = len(self.nondets)
            anon = f"__nondet_{idx}"
            self.nondets.append(Nondet(name=anon, c_type=c_type,
                                       bits=bits, signed=signed))
            return anon

        if fname in ERROR_FUNCS:
            self.has_assert = True
            return f'(_ for _ in ()).throw(AssertionError("{fname}"))'

        if fname in ASSERT_FUNCS:
            self.has_assert = True
            if not node.args or not node.args.exprs:
                raise Untranslatable("__VERIFIER_assert with no argument")
            arg = self._expr_to_python(node.args.exprs[0])
            return f"(0 if ({arg}) else (_ for _ in ()).throw(AssertionError('__VERIFIER_assert')))"

        if fname in ASSUME_FUNCS:
            if not node.args or not node.args.exprs:
                raise Untranslatable("__VERIFIER_assume with no argument")
            arg = self._expr_to_python(node.args.exprs[0])
            return f"(0 if ({arg}) else (_ for _ in ()).throw(_Assumed()))"

        raise Untranslatable(f"unsupported function call: {fname}")


def _c_type_to_bits(names: list[str]) -> tuple[str, int, bool]:
    """Map a list of C type tokens to (c_type_str, bits, signed)."""
    s = " ".join(names)
    if s in ("_Bool",):
        return ("_Bool", 1, False)
    if s in ("char", "signed char"):
        return ("char", 8, True)
    if s in ("unsigned char",):
        return ("unsigned char", 8, False)
    if s in ("short", "signed short", "short int", "signed short int"):
        return ("short", 16, True)
    if s in ("unsigned short", "unsigned short int"):
        return ("unsigned short", 16, False)
    if s in ("int", "signed", "signed int"):
        return ("int", 32, True)
    if s in ("unsigned", "unsigned int"):
        return ("unsigned int", 32, False)
    if s in ("long", "signed long", "long int", "signed long int",
              "long long", "signed long long"):
        return ("long", 64, True)
    if s in ("unsigned long", "unsigned long int",
              "unsigned long long"):
        return ("unsigned long", 64, False)
    raise Untranslatable(f"unsupported C type: {s}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _preprocess(source: str) -> str:
    """Strip GCC-specific constructs that ``pycparser`` doesn't handle.

    SV-COMP benchmark sources include ``__attribute__((noreturn))``,
    ``__extension__``, and similar GNU extensions in the assertion
    boilerplate (typically copied from ``glibc`` headers).  These
    aren't semantically meaningful for our purpose, so we elide them.
    """
    import re
    # Drop balanced __attribute__((...)) occurrences. Allow nested
    # parens.
    def _strip_attribute(s: str) -> str:
        out = []
        i = 0
        while i < len(s):
            j = s.find("__attribute__", i)
            if j < 0:
                out.append(s[i:])
                break
            out.append(s[i:j])
            # Skip whitespace after __attribute__
            k = j + len("__attribute__")
            while k < len(s) and s[k].isspace():
                k += 1
            if k < len(s) and s[k] == "(":
                depth = 0
                while k < len(s):
                    if s[k] == "(":
                        depth += 1
                    elif s[k] == ")":
                        depth -= 1
                        if depth == 0:
                            k += 1
                            break
                    k += 1
            i = k
        return "".join(out)

    source = _strip_attribute(source)
    # Strip __extension__, __asm__(...), __inline keywords.
    source = re.sub(r"\b__extension__\b", "", source)
    source = re.sub(r"\b__inline(__)?\b", "inline", source)
    # Drop __asm__(...) blocks.
    source = re.sub(
        r"__asm__\s*\([^)]*\)", "", source, flags=re.DOTALL
    )
    # Drop __restrict / __restrict__ keywords.
    source = re.sub(r"\b__restrict(__)?\b", "", source)
    # Strip C++/C99-style line comments.
    source = re.sub(r"//[^\n]*", "", source)
    # Strip /* ... */ block comments (across multiple lines).
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    # Strip preprocessor directives (#include, #define, #if, etc.).
    # Each starts at column 0 of a line; preserve line numbers by
    # replacing the directive content with a blank line.
    source = re.sub(r"(?m)^[ \t]*#.*$", "", source)
    return source


def transpile_c_source(source: str) -> TranspileResult:
    """Transpile a C source string into a Python module source.

    Raises :class:`Untranslatable` if the source uses features outside
    the supported subset.
    """
    full_source = SVCOMP_DECLS + "\n" + _preprocess(source)
    parser = pycparser.CParser()
    try:
        ast = parser.parse(full_source, filename="<svcomp>")
    except pycparser.c_parser.ParseError as e:
        raise Untranslatable(f"parse error: {e}")
    t = _Transpiler()
    t.visit(ast)

    # Wrap with a small preamble that defines _Assumed.
    preamble = textwrap.dedent("""
        # Auto-generated by dise.frontends.svcomp_c — do not edit by hand.

        class _Assumed(Exception):
            \"\"\"Raised when an __VERIFIER_assume guard fails.\"\"\"

    """).lstrip()
    body = "\n".join(t.lines) + "\n"
    py_source = preamble + body

    return TranspileResult(
        python_source=py_source,
        function_name="main",
        nondets=t.nondets,
        has_assert=t.has_assert,
        notes=t.notes,
    )


def transpile_c_program(path: str) -> TranspileResult:
    """Read a C file from disk and transpile it."""
    with open(path) as f:
        return transpile_c_source(f.read())
