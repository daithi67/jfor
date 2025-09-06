#!/usr/bin/env python3
# jfor.py — tiny "Johnson FOR" DSL with a real parser (no eval)
# Features:
#   - for i = A to B by C do ... end        (inclusive end; C defaults to 1)
#   - for x in EXPR do ... end              (iterate any sequence)
#   - if EXPR then ... [else ...] end
#   - NAME = EXPR
#   - print EXPR
#   - Expressions: ints, floats, strings, lists, names, + - * / // % **,
#                  comparisons, == != < <= > >=, and/or/not, unary +/-
#   - Comments: lines starting with '#' are ignored
#   - Whitespace-insensitive (line-based; blocks delimited by 'do/end' or 'then/else/end')

from __future__ import annotations
import sys
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict

# --------------------------
# Tokenizer
# --------------------------

@dataclass
class Tok:
    kind: str
    value: str
    pos: Tuple[int, int]  # (line, col)

KEYWORDS = {
    "for", "in", "to", "by", "do", "end",
    "if", "then", "else",
    "and", "or", "not",
    "print",
}

SINGLE = set("[](),:+-*/%<>=!{}")  # braces unused but tolerated
WHITESPACE = " \t\r"

def tokenize(src: str) -> List[Tok]:
    toks: List[Tok] = []
    i = 0
    line, col = 1, 1

    def emit(kind, value, l, c): toks.append(Tok(kind, value, (l, c)))

    while i < len(src):
        ch = src[i]

        if ch == '\n':
            emit("NL", "\n", line, col)
            i += 1; line += 1; col = 1
            continue

        if ch in WHITESPACE:
            i += 1; col += 1
            continue

        if ch == '#':
            # comment to end-of-line
            while i < len(src) and src[i] != '\n':
                i += 1; col += 1
            continue

        # string literal (single or double quotes)
        if ch in ('"', "'"):
            quote = ch
            l0, c0 = line, col
            i += 1; col += 1
            buf = []
            while i < len(src):
                ch = src[i]
                if ch == '\\':
                    if i+1 >= len(src):
                        raise_syntax(l0, c0, "unterminated string")
                    nxt = src[i+1]
                    esc = {'n':'\n','t':'\t','r':'\r','\\':'\\','"':'"',"'" :"'"}.get(nxt, nxt)
                    buf.append(esc); i += 2; col += 2
                    continue
                if ch == quote:
                    i += 1; col += 1
                    emit("STRING", "".join(buf), l0, c0)
                    break
                if ch == '\n':
                    raise_syntax(l0, c0, "newline in string literal")
                buf.append(ch); i += 1; col += 1
            else:
                raise_syntax(l0, c0, "unterminated string")
            continue

        # number (int or float)
        if ch.isdigit() or (ch == '.' and i+1 < len(src) and src[i+1].isdigit()):
            l0, c0 = line, col
            start = i
            saw_dot = (ch == '.')
            i += 1; col += 1
            while i < len(src):
                ch = src[i]
                if ch.isdigit():
                    i += 1; col += 1
                    continue
                if ch == '.' and not saw_dot:
                    saw_dot = True
                    i += 1; col += 1
                    continue
                break
            emit("NUMBER", src[start:i], l0, c0)
            continue

        # identifiers / keywords
        if ch.isalpha() or ch == '_':
            l0, c0 = line, col
            start = i
            i += 1; col += 1
            while i < len(src) and (src[i].isalnum() or src[i] == '_'):
                i += 1; col += 1
            word = src[start:i]
            kind = "KW" if word in KEYWORDS else "NAME"
            emit(kind, word, l0, c0)
            continue

        # operators and punctuation (handle two-char ops first)
        l0, c0 = line, col
        two = src[i:i+2]
        if two in ("<=", ">=", "==", "!=", "//", "**"):
            emit(two, two, l0, c0)
            i += 2; col += 2
            continue
        if ch in SINGLE:
            emit(ch, ch, l0, c0)
            i += 1; col += 1
            continue

        raise_syntax(line, col, f"unexpected character: {ch!r}")

    emit("EOF", "", line, col)
    return toks

def raise_syntax(line, col, msg):
    raise SyntaxError(f"[line {line}, col {col}] {msg}")

# --------------------------
# AST nodes
# --------------------------

class Expr: ...
@dataclass
class ENum(Expr): value: float | int
@dataclass
class EStr(Expr): value: str
@dataclass
class EName(Expr): name: str
@dataclass
class EList(Expr): items: List[Expr]
@dataclass
class EUnary(Expr): op: str; rhs: Expr
@dataclass
class EBin(Expr): op: str; lhs: Expr; rhs: Expr

class Stmt: ...
@dataclass
class SAssign(Stmt): name: str; expr: Expr
@dataclass
class SPrint(Stmt): expr: Expr
@dataclass
class SBlock(Stmt): body: List[Stmt]
@dataclass
class SIf(Stmt): cond: Expr; then_block: SBlock; else_block: Optional[SBlock]
@dataclass
class SForCounter(Stmt):
    var: str; start: Expr; end: Expr; step: Optional[Expr]; body: SBlock
@dataclass
class SForIter(Stmt):
    var: str; iterable: Expr; body: SBlock

# --------------------------
# Pratt expression parser
# --------------------------

class Parser:
    def __init__(self, toks: List[Tok]):
        self.toks = toks
        self.i = 0

    def cur(self) -> Tok: return self.toks[self.i]
    def advance(self) -> Tok:
        t = self.toks[self.i]; self.i += 1; return t
    def match(self, *kinds) -> Optional[Tok]:
        if self.cur().kind in kinds:
            return self.advance()
        return None
    def expect(self, *kinds) -> Tok:
        t = self.cur()
        if t.kind not in kinds:
            raise_syntax(t.pos[0], t.pos[1], f"expected {'/'.join(kinds)}, got {t.kind} {t.value!r}")
        return self.advance()

    # Expression grammar with precedence (lowest to highest):
    # or, and, not (unary), comparisons, + -, *, / // %, ** (right-assoc), unary + -
    def parse_expr(self) -> Expr:
        return self.parse_or()

    def parse_or(self) -> Expr:
        e = self.parse_and()
        while self.match("KW") and self.toks[self.i-1].value == "or":
            op = "or"
            rhs = self.parse_and()
            e = EBin(op, e, rhs)
        return e

    def parse_and(self) -> Expr:
        e = self.parse_not()
        while self.match("KW") and self.toks[self.i-1].value == "and":
            op = "and"
            rhs = self.parse_not()
            e = EBin(op, e, rhs)
        return e

    def parse_not(self) -> Expr:
        if self.match("KW") and self.toks[self.i-1].value == "not":
            return EUnary("not", self.parse_not())
        return self.parse_cmp()

    def parse_cmp(self) -> Expr:
        e = self.parse_add()
        while True:
            t = self.cur().kind
            if t in ("==","!=", "<","<=",">",">="):
                op = self.advance().kind
                rhs = self.parse_add()
                e = EBin(op, e, rhs)
            else:
                break
        return e

    def parse_add(self) -> Expr:
        e = self.parse_mul()
        while True:
            t = self.cur().kind
            if t in ("+","-"):
                op = self.advance().kind
                rhs = self.parse_mul()
                e = EBin(op, e, rhs)
            else:
                break
        return e

    def parse_mul(self) -> Expr:
        e = self.parse_pow()
        while True:
            t = self.cur().kind
            if t in ("*","/","//","%"):
                op = self.advance().kind
                rhs = self.parse_pow()
                e = EBin(op, e, rhs)
            else:
                break
        return e

    def parse_pow(self) -> Expr:
        e = self.parse_unary()
        if self.match("**"):
            rhs = self.parse_pow()  # right-assoc
            e = EBin("**", e, rhs)
        return e

    def parse_unary(self) -> Expr:
        if self.match("+"):
            return EUnary("+", self.parse_unary())
        if self.match("-"):
            return EUnary("-", self.parse_unary())
        return self.parse_atom()

    def parse_atom(self) -> Expr:
        t = self.cur()
        if t.kind == "NUMBER":
            self.advance()
            if "." in t.value:
                return ENum(float(t.value))
            return ENum(int(t.value))
        if t.kind == "STRING":
            self.advance(); return EStr(t.value)
        if t.kind == "NAME":
            self.advance(); return EName(t.value)
        if t.kind == "[":
            return self.parse_list()
        if t.kind == "(":
            self.advance()
            e = self.parse_expr()
            self.expect(")")
            return e
        raise_syntax(t.pos[0], t.pos[1], f"unexpected token in expression: {t.kind} {t.value!r}")

    def parse_list(self) -> Expr:
        l0, c0 = self.cur().pos
        self.expect("[")
        items: List[Expr] = []
        if self.cur().kind != "]":
            items.append(self.parse_expr())
            while self.match(","):
                items.append(self.parse_expr())
        self.expect("]")
        return EList(items)

# --------------------------
# Statement parser (line/block oriented)
# --------------------------

def parse_program(src: str) -> SBlock:
    # Convert NLs into statement boundaries by splitting lines, but we still use tokens to parse.
    toks = tokenize(src)
    # We'll parse statements until EOF. Blocks are delimited by 'do/end' and 'then/else/end'.
    p = Parser(toks)
    stmts, _ = parse_block(p, terminators=("EOF",))
    return SBlock(stmts)

def parse_block(p: Parser, terminators: Tuple[str, ...]) -> Tuple[List[Stmt], Optional[str]]:
    stmts: List[Stmt] = []
    # simple helper to skip blank lines
    def skip_blank_lines():
        while p.match("NL"):
            pass

    skip_blank_lines()
    while True:
        t = p.cur()
        if t.kind in terminators:
            return stmts, t.kind
        if t.kind == "EOF":
            return stmts, "EOF"
        if t.kind == "NL":
            p.advance(); continue

        if t.kind == "KW" and t.value == "print":
            p.advance()
            expr = parse_simple_expr_on_line(p)
            stmts.append(SPrint(expr))
            continue

        if t.kind == "KW" and t.value == "if":
            stmts.append(parse_if(p))
            continue

        if t.kind == "KW" and t.value == "for":
            stmts.append(parse_for(p))
            continue

        # assignment: NAME '=' expr
        if t.kind == "NAME":
            name_tok = p.advance()
            p.expect("=")
            expr = parse_simple_expr_on_line(p)
            stmts.append(SAssign(name_tok.value, expr))
            continue

        raise_syntax(t.pos[0], t.pos[1], f"unexpected token at statement start: {t.kind} {t.value!r}")

def parse_simple_expr_on_line(p: Parser) -> Expr:
    expr = p.parse_expr()
    # consume until end-of-line or EOF or block keywords
    while p.match("NL"):
        pass
    return expr

def parse_if(p: Parser) -> SIf:
    # 'if' EXPR 'then' NL ... ['else' NL ...] 'end'
    if_tok = p.expect("KW"); assert if_tok.value == "if"
    cond = p.parse_expr()
    then_kw = p.expect("KW")
    if then_kw.value != "then":
        raise_syntax(then_kw.pos[0], then_kw.pos[1], "expected 'then'")
    # require end-of-line
    if not p.match("NL"):
        # allow inline 'then' newline-less? We'll still require newline for block clarity
        pass
    then_body, term = parse_block(p, terminators=("KW","EOF"))
    else_block: Optional[SBlock] = None
    if term == "KW" and p.toks[p.i-1].value == "else":
        # consume optional trailing NLs
        while p.match("NL"):
            pass
        else_body, term2 = parse_block(p, terminators=("KW","EOF"))
        if term2 != "KW" or p.toks[p.i-1].value != "end":
            t = p.cur()
            raise_syntax(t.pos[0], t.pos[1], "expected 'end' to close if/else")
        # consume trailing NLs
        while p.match("NL"):
            pass
        else_block = SBlock(else_body)
        return SIf(cond, SBlock(then_body), else_block)
    elif term == "KW" and p.toks[p.i-1].value == "end":
        while p.match("NL"):
            pass
        return SIf(cond, SBlock(then_body), None)
    else:
        t = p.cur()
        raise_syntax(t.pos[0], t.pos[1], "expected 'else' or 'end' after then-block")

def parse_for(p: Parser) -> Stmt:
    # 'for' NAME '=' EXPR 'to' EXPR ['by' EXPR] 'do' NL block 'end'
    # or 'for' NAME 'in' EXPR 'do' NL block 'end'
    for_tok = p.expect("KW"); assert for_tok.value == "for"
    var_tok = p.expect("NAME")
    # decide between counter vs iterator
    if p.match("="):
        # counter form
        start = p.parse_expr()
        to_kw = p.expect("KW")
        if to_kw.value != "to":
            raise_syntax(to_kw.pos[0], to_kw.pos[1], "expected 'to'")
        end = p.parse_expr()
        step: Optional[Expr] = None
        if p.match("KW") and p.toks[p.i-1].value == "by":
            step = p.parse_expr()
        do_kw = p.expect("KW")
        if do_kw.value != "do":
            raise_syntax(do_kw.pos[0], do_kw.pos[1], "expected 'do'")
        # newline before body
        if not p.match("NL"):
            pass
        body, term = parse_block(p, terminators=("KW","EOF"))
        if term != "KW" or p.toks[p.i-1].value != "end":
            t = p.cur()
            raise_syntax(t.pos[0], t.pos[1], "expected 'end' to close for")
        while p.match("NL"):
            pass
        return SForCounter(var_tok.value, start, end, step, SBlock(body))
    else:
        in_kw = p.expect("KW")
        if in_kw.value != "in":
            raise_syntax(in_kw.pos[0], in_kw.pos[1], "expected '=' or 'in' after loop variable")
        iterable = p.parse_expr()
        do_kw = p.expect("KW")
        if do_kw.value != "do":
            raise_syntax(do_kw.pos[0], do_kw.pos[1], "expected 'do'")
        if not p.match("NL"):
            pass
        body, term = parse_block(p, terminators=("KW","EOF"))
        if term != "KW" or p.toks[p.i-1].value != "end":
            t = p.cur()
            raise_syntax(t.pos[0], t.pos[1], "expected 'end' to close for")
        while p.match("NL"):
            pass
        return SForIter(var_tok.value, iterable, SBlock(body))

# --------------------------
# Interpreter
# --------------------------

class RuntimeErrorEx(Exception): ...

class Env(dict):
    pass

def eval_expr(e: Expr, env: Env) -> Any:
    if isinstance(e, ENum):
        return e.value
    if isinstance(e, EStr):
        return e.value
    if isinstance(e, EName):
        if e.name in env:
            return env[e.name]
        raise RuntimeErrorEx(f"name '{e.name}' is not defined")
    if isinstance(e, EList):
        return [eval_expr(x, env) for x in e.items]
    if isinstance(e, EUnary):
        v = eval_expr(e.rhs, env)
        if e.op == "+": return +v
        if e.op == "-": return -v
        if e.op == "not": return not bool(v)
        raise RuntimeErrorEx(f"bad unary op {e.op}")
    if isinstance(e, EBin):
        op = e.op
        if op in ("+","-","*","/","//","%","**"):
            l = eval_expr(e.lhs, env); r = eval_expr(e.rhs, env)
            if op == "+": return l + r
            if op == "-": return l - r
            if op == "*": return l * r
            if op == "/": return l / r
            if op == "//": return l // r
            if op == "%": return l % r
            if op == "**": return l ** r
        if op in ("<","<=",">",">=","==","!="):
            l = eval_expr(e.lhs, env); r = eval_expr(e.rhs, env)
            if op == "<":  return l < r
            if op == "<=": return l <= r
            if op == ">":  return l > r
            if op == ">=": return l >= r
            if op == "==": return l == r
            if op == "!=": return l != r
        if op in ("and","or"):
            if op == "and":
                return bool(eval_expr(e.lhs, env)) and bool(eval_expr(e.rhs, env))
            else:
                return bool(eval_expr(e.lhs, env)) or bool(eval_expr(e.rhs, env))
        raise RuntimeErrorEx(f"bad binary op {op}")
    raise RuntimeErrorEx("unknown expression type")

def run_stmt(s: Stmt, env: Env):
    if isinstance(s, SAssign):
        env[s.name] = eval_expr(s.expr, env); return
    if isinstance(s, SPrint):
        val = eval_expr(s.expr, env)
        print(val)
        return
    if isinstance(s, SBlock):
        for st in s.body:
            run_stmt(st, env)
        return
    if isinstance(s, SIf):
        cond = eval_expr(s.cond, env)
        if cond:
            run_stmt(s.then_block, env)
        elif s.else_block is not None:
            run_stmt(s.else_block, env)
        return
    if isinstance(s, SForCounter):
        start = eval_expr(s.start, env)
        end   = eval_expr(s.end, env)
        step  = eval_expr(s.step, env) if s.step is not None else 1
        if step == 0:
            raise RuntimeErrorEx("for-by step must be nonzero")
        # inclusive end, supports negative step
        if step > 0:
            i = start
            while i <= end:
                env[s.var] = i
                run_stmt(s.body, env)
                i = i + step
        else:
            i = start
            while i >= end:
                env[s.var] = i
                run_stmt(s.body, env)
                i = i + step
        return
    if isinstance(s, SForIter):
        it = eval_expr(s.iterable, env)
        try:
            iterator = iter(it)
        except TypeError:
            raise RuntimeErrorEx("object is not iterable in 'for ... in ...'")
        for v in iterator:
            env[s.var] = v
            run_stmt(s.body, env)
        return
    raise RuntimeErrorEx("unknown statement type")

def run(src: str, env: Optional[Env] = None) -> Env:
    if env is None: env = Env()
    prog = parse_program(src)
    run_stmt(prog, env)
    return env

# --------------------------
# CLI demo
# --------------------------

DEMO = """# demo.jfor — Johnson FOR DSL
print "Counter up:"
for i = 1 to 10 by 3 do
    print i
end

print "Counter down:"
for n = 5 to 1 by -2 do
    print n
end

print "Iterator over list:"
for fruit in ["apple","banana","cherry"] do
    print fruit
end

# basic if/else
x = 7
if x % 2 == 0 then
    print "even"
else
    print "odd"
end

# boolean logic, arithmetic
total = 0
for k = 1 to 5 do
    total = total + k ** 2
end
print total
"""

def main(argv: List[str]):
    if len(argv) >= 2 and argv[1] == "demo":
        run(DEMO, Env())
        return
    if len(argv) >= 2:
        with open(argv[1], "r", encoding="utf-8") as f:
            src = f.read()
        run(src, Env())
        return
    print("Usage:")
    print("  python jfor.py demo            # run built-in demo")
    print("  python jfor.py yourfile.jfor   # run a script file")

if __name__ == "__main__":
    main(sys.argv)
