import re
import time

# ------------------------
# Lexer
# ------------------------
token_spec = [
    ('FUNC', r'\bfunc\b'),
    ('RETURN', r'\breturn\b'),
    ('INCL', r'\binclude\b'),
    ('IF', r'\bif\b'),
    ('ELSE', r'\belse\b'),
    ('WHILE', r'\bwhile\b'),
    ('NUMBER', r'\d+(\.\d+)?'),
    ('BOOL', r'\btrue\b|\bfalse\b'),
    ('STR', r'"[\s\S]*?"'),
    ('IDEN', r'[A-Za-z_][A-Za-z0-9_]*'),
    ('EQUALTO', r'=='),
    ('LESSEQ', r'<='),
    ('GREATEQ', r'>='),
    ('EQUAL', r'='),
    ('LESS', r'<'),
    ('GREAT', r'>'),
    ('PLUS', r'\+'),
    ('MINUS', r'-'),
    ('STAR', r'\*'),
    ('SLASH', r'/'),
    ('PERCENT', r'%'),
    ('DOT', r'\.'),
    ('LPAREN', r'\('),
    ('RPAREN', r'\)'),
    ('LBRACKET', r'\{'),
    ('RBRACKET', r'\}'),
    ('SEMICOLON', r';'),
    ('COMMA', r','),
    ('WHITESPACE', r'\s+'),
]

token_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_spec)
get_token = re.compile(token_regex).match

def lexer(code):
    pos = 0
    tokens = []
    while pos < len(code):
        m = get_token(code, pos)
        if not m:
            raise SyntaxError(f'Unexpected character: {code[pos]}')
        typ = m.lastgroup
        if typ != 'WHITESPACE':
            tokens.append((typ, m.group(typ)))
        pos = m.end()
    return tokens

# ------------------------
# AST Node
# ------------------------
class ASTNode:
    def __init__(self, type_, value=None, children=None):
        self.type = type_
        self.value = value
        self.children = children or []

    def __repr__(self):
        return f"{self.type}({self.value}, {self.children})"

# ------------------------
# Parser
# ------------------------
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def eat(self, token_type):
        tok = self.peek()
        if tok and tok[0] == token_type:
            self.pos += 1
            return tok
        raise SyntaxError(f'Expected {token_type} but got {tok}')

    # ------------------------
    # Parsing
    # ------------------------
    def parseSemicolon(self):
        return self.eat('SEMICOLON')

    def parseExpression(self):
        return self.parseAddSub()

    def parseAddSub(self):
        node = self.parseMulDiv()
        while self.peek() and self.peek()[0] in ('PLUS', 'MINUS'):
            op = self.eat(self.peek()[0])
            right = self.parseMulDiv()
            node = ASTNode('BinOp', op[1], [node, right])
        return node
    
    def parseMulDiv(self):
        node = self.parsePrimary()
        while self.peek() and self.peek()[0] in ('STAR', 'SLASH', 'PERCENT'):
            op = self.eat(self.peek()[0])
            right = self.parsePrimary()
            node = ASTNode('BinOp', op[1], [node, right])
        return node

    def parsePrimary(self):
        tok = self.peek()
        if tok[0] in ('NUMBER', 'BOOL', 'STR'):
            self.pos += 1
            return ASTNode('Literal', tok[1])
        elif tok[0] == 'IDEN':
            self.pos += 1
            node = ASTNode('Identifier', tok[1])
            # Handle dot chains
            while self.peek() and self.peek()[0] == 'DOT':
                self.eat('DOT')
                method = self.eat('IDEN')
                node = ASTNode('Call', f"{node.value}.{method[1]}", [])
            # Handle function call
            if self.peek() and self.peek()[0] == 'LPAREN':
                self.eat('LPAREN')
                args = []
                while self.peek() and self.peek()[0] != 'RPAREN':
                    args.append(self.parseExpression())
                    if self.peek() and self.peek()[0] == 'COMMA':
                        self.eat('COMMA')
                self.eat('RPAREN')
                if node.type == 'Call':
                    node.children.extend(args)
                else:
                    node = ASTNode('Call', node.value, args)
            return node
        elif tok[0] == 'LPAREN':
            self.eat('LPAREN')
            node = self.parseExpression()
            self.eat('RPAREN')
            return node
        else:
            raise SyntaxError(f"Unexpected token in expression: {tok}")

    def parseAssignment(self):
        target = self.eat('IDEN')
        self.eat('EQUAL')
        expr = self.parseExpression()
        self.parseSemicolon()
        return ASTNode('Assign', target[1], [expr])

    def parseFunction(self):
        self.eat('FUNC')
        name = self.eat('IDEN')[1]
        self.eat('LPAREN')
        params = []
        while self.peek() and self.peek()[0] != 'RPAREN':
            params.append(self.eat('IDEN')[1])
            if self.peek() and self.peek()[0] == 'COMMA':
                self.eat('COMMA')
        self.eat('RPAREN')
        self.eat('LBRACKET')
        body = self.parseStatements()
        self.eat('RBRACKET')
        return ASTNode('Function', name, [ASTNode('Params', None, params), body])

    def parseReturn(self):
        self.eat('RETURN')
        expr = self.parseExpression()
        self.parseSemicolon()
        return ASTNode('Return', None, [expr])

    def parseStatement(self):
        tok = self.peek()
        if tok[0] == 'IDEN' and self.pos+1 < len(self.tokens) and self.tokens[self.pos+1][0] == 'EQUAL':
            return self.parseAssignment()
        elif tok[0] == 'FUNC':
            return self.parseFunction()
        elif tok[0] == 'RETURN':
            return self.parseReturn()
        elif tok[0] == 'IF':
            self.eat('IF')
            condition = self.parseExpression()
            self.eat('LBRACKET')
            body = self.parseStatements()
            self.eat('RBRACKET')
            else_body = None
            if self.peek() and self.peek()[0] == 'ELSE':
                self.eat('ELSE')
                self.eat('LBRACKET')
                else_body = self.parseStatements()
                self.eat('RBRACKET')
            children = [condition, body]
            if else_body:
                children.append(else_body)
            return ASTNode('If', None, children)
        elif tok[0] == 'WHILE':
            self.eat('WHILE')
            condition = self.parseExpression()
            self.eat('LBRACKET')
            body = self.parseStatements()
            self.eat('RBRACKET')
            return ASTNode('While', None, [condition, body])
        elif tok[0] == 'INCL':
            self.eat('INCL')
            lib = self.eat('IDEN')
            self.parseSemicolon()
            return ASTNode('Include', lib[1])
        else:
            expr = self.parseExpression()
            self.parseSemicolon()
            return expr

    def parseStatements(self):
        stmts = []
        while self.pos < len(self.tokens) and self.peek()[0] != 'RBRACKET':
            stmts.append(self.parseStatement())
        return ASTNode('Block', None, stmts)

    def parseProgram(self):
        return self.parseStatements()


# ------------------------
# Interpreter
# ------------------------
class Environment:
    def __init__(self):
        self.vars = {}
        self.libs = {}

    def register_lib(self, name, lib_dict):
        self.libs[name] = lib_dict

    def get(self, name):
        if name in self.vars:
            return self.vars[name]
        elif name in self.libs:
            return self.libs[name]
        else:
            raise NameError(f"Unknown identifier: {name}")

    def set(self, name, value):
        self.vars[name] = value


class Interpreter:
    def __init__(self, env):
        self.env = env
        self.functions = {}

    def eval(self, node):
        method = getattr(self, f'eval_{node.type}', None)
        if not method:
            raise NotImplementedError(f"No eval method for {node.type}")
        return method(node)

    def eval_Block(self, node):
        result = None
        for stmt in node.children:
            result = self.eval(stmt)
        return result

    def eval_Literal(self, node):
        if node.value in ('true', 'false'):
            return node.value == 'true'
        elif isinstance(node.value, str) and node.value.startswith('"'):
            return node.value[1:-1]
        elif isinstance(node.value, str) and node.value.replace('.', '', 1).isdigit():
            return float(node.value) if '.' in node.value else int(node.value)
        return node.value

    def eval_Identifier(self, node):
        return self.env.get(node.value)

    def eval_Assign(self, node):
        value = self.eval(node.children[0])
        self.env.set(node.value, value)
        return value

    def eval_BinOp(self, node):
        left = self.eval(node.children[0])
        right = self.eval(node.children[1])
        if node.value == '+': return left + right
        if node.value == '-': return left - right
        if node.value == '*': return left * right
        if node.value == '/': return left / right
        if node.value == '%': return left % right
        if node.value == '==': return left == right
        if node.value == '<=': return left <= right
        if node.value == '>=': return left >= right
        if node.value == '<': return left < right
        if node.value == '>': return left > right
        raise NotImplementedError(f"Operator {node.value}")

    def eval_Call(self, node):
        # dotted calls: io.prnnum
        if '.' in node.value:
            lib_name, func_name = node.value.split('.', 1)
            lib = self.env.get(lib_name)
            args = [self.eval(arg) for arg in node.children]
            return lib[func_name](*args)
        # normal function call
        if node.value in self.functions:
            func_node = self.functions[node.value]
            env_backup = self.env.vars.copy()
            for param, arg in zip(func_node.children[0].children, node.children):
                self.env.set(param, self.eval(arg))
            result = self.eval(func_node.children[1])
            self.env.vars = env_backup
            return result
        raise NameError(f"Unknown function: {node.value}")

    def eval_Function(self, node):
        self.functions[node.value] = node
        return None

    def eval_Return(self, node):
        return self.eval(node.children[0])

    def eval_If(self, node):
        cond = self.eval(node.children[0])
        if cond:
            return self.eval(node.children[1])
        elif len(node.children) > 2:
            return self.eval(node.children[2])
        return None

    def eval_While(self, node):
        cond_node, body_node = node.children
        result = None
        while self.eval(cond_node):
            result = self.eval(body_node)
        return result

    def eval_Include(self, node):
        if node.value not in LIBRARIES:
            raise ImportError(f"Library '{node.value}' not found")
        self.env.register_lib(node.value, LIBRARIES[node.value])
        return None

# ------------------------
# Example libraries
# ------------------------
LIBRARIES = {
    'io': {
        'prnnum': lambda *args: print(*args), # Legacy
        'prnstr': lambda *args: print(*args), # Legacy
        'prn': lambda *args: print(*args),
        'instr': lambda x: input(x),
        'innum': lambda x: float(input(x)),
    },
    'cast': {
        'int': lambda x: int(x),
        'flo': lambda x: float(x),
        'str': lambda x: str(x),
    },
}

LIBRARIES['objects'] = {
    # constructors
    "list": lambda *args: list(args),
    "tuple": lambda *args: tuple(args),
    "set": lambda *args: set(args),

    # dict constructor: expects key1, val1, key2, val2, ...
    "dict": lambda *args: {args[i]: args[i+1] for i in range(0, len(args), 2)},

    # helpers
    "len": lambda x: len(x),
    "get": lambda obj, key: obj[key] if isinstance(obj, (dict, list, tuple)) else (_ for _ in ()).throw(TypeError("Not indexable")),

    # list operations
    "append": lambda lst, val: lst + [val] if isinstance(lst, list) else (_ for _ in ()).throw(TypeError("Not a list")),
    "replace": lambda lst, idx, val: (lst.__setitem__(idx, val) or lst) if isinstance(lst, list) else (_ for _ in ()).throw(TypeError("Not a list")),

    # dict operations
    "dict_set": lambda d, key, val: (d.update({key: val}) or d) if isinstance(d, dict) else (_ for _ in ()).throw(TypeError("Not a dict")),

    # extra helpers
    "keys": lambda d: list(d.keys()) if isinstance(d, dict) else (_ for _ in ()).throw(TypeError("Not a dict")),
    "values": lambda d: list(d.values()) if isinstance(d, dict) else (_ for _ in ()).throw(TypeError("Not a dict")),

    # set operations
    "add": lambda s, val: (s.add(val) or s) if isinstance(s, set) else (_ for _ in ()).throw(TypeError("Not a set")),
    "remove": lambda s, val: (s.remove(val) or s) if isinstance(s, set) else (_ for _ in ()).throw(TypeError("Not a set")),
}

LIBRARIES['time'] = {
    "sleep": lambda seconds: time.sleep(seconds)
}

def remove_comments(code: str) -> str:
    """
    Removes comments starting with # until the end of the line.
    Preserves everything else.
    """
    # Remove # comments
    code = re.sub(r'#.*', '', code)
    return code

# ------------------------
# Example program
# ------------------------
code = """
include io;

a = 0;
b = 1;
while true {
    io.prn(a);
    temp = a + b;
    a = b;
    b = temp;
}
"""

tokens = lexer(remove_comments(code))
parser = Parser(tokens)
ast = parser.parseProgram()
env = Environment()
interpreter = Interpreter(env)
interpreter.eval(ast)
