import sys
from dataclasses import dataclass

from var import Var
from const import Constant, INT, DOUBL, STRIN

INTEGER = 'INTEGER'
DOUBLE = 'DOUBLE'
STRING = 'STRING'
BOOL = 'BOOL'
PLUS = 'PLUS'
MINUS = 'MINUS'
MUL = 'MUL'
DIV = 'DIV'
INTEGER_DIV = 'INTEGER_DIV'
DOUBLE_DIV = 'DOUBLE_DIV'
LPAREN = 'LPAREN'
RPAREN = 'RPAREN'
LBRACE = 'LBRACE'
RBRACE = 'RBRACE'
ID = 'ID'
ASSIGN = 'ASSIGN'
SEMI = 'SEMI'
DOT = 'DOT'
COMMA = 'COMMA'
INTEGER_CONST = 'INTEGER_CONST'
DOUBLE_CONST = 'DOUBLE_CONST'
EOF = 'EOF'
MOD = 'MOD'
LESS = 'LESS'
MORE = 'MORE'
AND = 'AND'
OR = 'OR'
NOT = 'NOT'
EQUAL = 'EQUAL'
LESS_OR_EQUAL = 'LESS_OR_EQUAL'
MORE_OR_EQUAL = 'MORE_OR_EQUAL'
NOT_EQUAL = 'NOT_EQUAL'
BOOL_CONST = 'BOOL_CONST'
STRING_LITERAL = 'STRING_LITERAL'
WHILE = 'WHILE'
IF = 'IF'
ELSE = 'ELSE'
CONTINUE = 'CONTINUE'
BREAK = 'BREAK'
TO_INT = 'TO_INT'
TO_STR = 'TO_STR'
TO_DOUBLE = 'TO_DOUBLE'
TO_BOOL = "TO_BOOL"
SCAN = 'SCAN'
PRINT = 'PRINT'
EXIT = 'EXIT'
TRUE = 'TRUE'
FALSE = 'FALSE'
EEQUAL = 'EEQUAL'


OPERATORS = {
    '+': PLUS,
    '-': MINUS,
    '*': MUL,
    '/': DIV,
    '%': MOD,
    '=': ASSIGN,
    '<': LESS,
    '>': MORE,
    '&&': AND,
    '||': OR,
    '!': NOT,
    '==': EQUAL,
    '<=': LESS_OR_EQUAL,
    '>=': MORE_OR_EQUAL,
    '!=': NOT_EQUAL,
    ';': SEMI,
    '(': LPAREN,
    ')': RPAREN,
    '{': LBRACE,
    '}': RBRACE
}


@dataclass()
class Lexeme(object):
    def __init__(self, type, value, line, column, id = None):
        self._id = id
        self.type = type
        self.value = value
        self.line = line
        self.column = column

    def __str__(self):
        return 'Token({type}, {value})'.format(
            type=self.type,
            value=repr(self.value)
        )

    def __repr__(self):
        return self.__str__()

@dataclass()
class Token_to_parse(object):
    def __init__(self, table_idx, index, line, column, id = None):
        self.id = id
        self.table_index = table_idx
        self.index = index
        self.line = line
        self.column = column

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Token to parse ({type}, {value})'.format(
            type=self.table_index,
            value=repr(self.index)
        )


RESERVED_KEYWORDS = {
    'int': INTEGER,
    'double': DOUBLE,
    'string': STRING,
    'bool': BOOL,
    'while': WHILE,
    'if': IF,
    'else': ELSE,
    'continue': CONTINUE,
    'break': BREAK,
    'to_int': TO_INT,
    'to_str': TO_STR,
    'to_double': TO_DOUBLE,
    'scan': SCAN,
    'prin': PRINT,
    'exit': EXIT,
    'true': TRUE,
    'false': FALSE
}

CONSTANTS = {
    'INTEGER_CONST': INTEGER_CONST,
    'DOUBLE_CONST': DOUBLE_CONST,
    'BOOL_CONST': BOOL_CONST,
    'STRING_LITERAL': STRING_LITERAL,
}


class LexicalError(Exception):
    def __init__(self, txt, line, column, id =None):
        self.message = txt
        self.id = id

        self.line = line
        self.column = column
        super().__init__(txt)

    def __repr__(self) :
        return self.__str__()


    def __str__(self):
        return f"LexerError: {self.message} (line {self.line}, index {self.column})"


class OperandError(LexicalError):
    def __init__(self, char, line, column):
        super().__init__(f"Wrong usage of operand near '{char}'", line, column)

    def __str__(self):
        return f"OperandError: (line {self.line}, index {self.column})"


class FloatNumberError(LexicalError):
    def __init__(self, char, line, column):
        super().__init__(f"Wrong floating number near '{char}'", line, column)

    def __str__(self):
        return f"FloatNumberError: {self.message} (line {self.line}, index {self.column})"


class NumberError(LexicalError):
    def __init__(self, char, line, column):
        super().__init__(f"Wrong number near '{char}'", line, column)

    def __str__(self):
        return f"NumberError: {self.message} (line {self.line}, index {self.column})"


class UnknownSymbol(LexicalError):
    def __init__(self, symbol, line, column):
        super().__init__("unknown symbol: " + symbol, line, column)

    def __str__(self):
        return f"UnknownSymbol: {self.message} (line {self.line}, index {self.column})"


class NoMatchingQuotes(LexicalError):
    def __init__(self, symbol, line, column):
        super().__init__("No matching quotes: " + symbol, line, column)

    def __str__(self):
        return f"NoMatchingQuotes: {self.message} (line {self.line}, index {self.column})"


class Lexer(object):
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.current_char = self.text[self.pos]
        self.cur_line = 1
        self.cur_column = 0

    def peek(self):
        peek_pos = self.pos + 1
        if peek_pos > len(self.text) - 1:
            return None
        else:
            return self.text[peek_pos]

    # def error(self):
    #     raise Exception('Invalid character', f"line: {self.cur_line}, column: {self.cur_column}")

    def advance(self):
        self.pos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None  # Indicates end of input
        else:
            self.current_char = self.text[self.pos]
            self.cur_column += 1

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            if self.current_char == '\n':
                self.cur_line += 1
                self.cur_column = 0
            self.advance()

    def _string_literal(self):
        self.advance()
        result = ''
        prev_char = ''
        parenth_closed = False
        while self.current_char is not None:
            if self.current_char == '"' and prev_char != "\\":
                parenth_closed = True
                self.advance()
                break
            if self.current_char != "\\":
                result += self.current_char
            prev_char = self.current_char
            self.advance()
        if not parenth_closed and self.current_char is None:
            raise NoMatchingQuotes(result, self.cur_line, self.cur_column)
        const = Constant(result, STRIN)

        token = Lexeme('STRING_LITERAL', const, self.cur_line, self.cur_column)
        return token

    def number(self):
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()

        if self.current_char == '.':
            result += self.current_char

            if not self.peek().isdigit():
                raise FloatNumberError(self.current_char, self.cur_line, self.cur_column)
            self.advance()

            while (
                    self.current_char is not None and
                    self.current_char.isdigit()
            ):
                result += self.current_char
                self.advance()
            const = Constant(float(result), DOUBL)
            token = Lexeme('DOUBLE_CONST', const, self.cur_line, self.cur_column)
        elif self.current_char not in OPERATORS.keys() and self.current_char not in (' ', ';'):
            raise NumberError(self.current_char, self.cur_line, self.cur_column)
        else:
            const = Constant(int(result), INT)

            token = Lexeme('INTEGER_CONST', const, self.cur_line, self.cur_column)
        return token

    def _id_and_reserved(self):
        result = ''
        if self.current_char is not None and self.current_char.isalpha:
            result += self.current_char
            self.advance()
            while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
                result += self.current_char
                self.advance()
        if result in RESERVED_KEYWORDS.keys():
            token = Lexeme(RESERVED_KEYWORDS[result], result, self.cur_line, self.cur_column)
        else:
            var = Var(result, None, 0, 0)
            token = Lexeme(ID, var, self.cur_line, self.cur_column)
        return token

    def get_next_token(self):
        while self.current_char is not None:

            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char == '{':
                self.advance()
                return Lexeme(LBRACE, '{', self.cur_line, self.cur_column)

            if self.current_char == '}':
                self.advance()
                return Lexeme(RBRACE, '}', self.cur_line, self.cur_column)

            if self.current_char == '"':
                return self._string_literal()

            if self.current_char.isalpha():
                return self._id_and_reserved()

            if self.current_char.isdigit():
                return self.number()

            if self.current_char == '=':
                if self.peek() == '=':
                    self.advance()
                    if not (self.peek().isalnum() or self.peek() == ' '):
                        raise OperandError(self.current_char, self.cur_line, self.cur_column)
                    self.advance()
                    token = Lexeme(EQUAL, '==', self.cur_line, self.cur_column)
                    print("type = ", token.type)
                else:
                    if not (self.peek().isalnum() or self.peek() == ' '):
                        raise OperandError(self.current_char, self.cur_line, self.cur_column)
                    self.advance()
                    token = Lexeme(ASSIGN, '=', self.cur_line, self.cur_column)
                return token

            if self.current_char == '<':
                if self.peek() == '=':
                    self.advance()
                    if not (self.peek().isalnum() or self.peek() == ' '):
                        raise OperandError(self.current_char, self.cur_line, self.cur_column)
                    self.advance()
                    token = Lexeme(LESS_OR_EQUAL, '<=', self.cur_line, self.cur_column)
                else:
                    if not (self.peek().isalnum() or self.peek() == ' '):
                        raise OperandError(self.current_char, self.cur_line, self.cur_column)
                    self.advance()
                    token = Lexeme(LESS, '<', self.cur_line, self.cur_column)
                return token

            if self.current_char == '>':
                if self.peek() == '=':
                    self.advance()
                    if not (self.peek().isalnum() or self.peek() == ' '):
                        raise OperandError(self.current_char, self.cur_line, self.cur_column)
                    self.advance()
                    token = Lexeme(MORE_OR_EQUAL, '>=', self.cur_line, self.cur_column)
                else:
                    if not (self.peek().isalnum() or self.peek() == ' '):
                        raise OperandError(self.current_char, self.cur_line, self.cur_column)
                    self.advance()
                    token = Lexeme(MORE, '>', self.cur_line, self.cur_column)
                return token

            if self.current_char == ';':
                self.advance()
                return Lexeme(SEMI, ';', self.cur_line, self.cur_column)

            if self.current_char == ',':
                self.advance()
                return Lexeme(COMMA, ',', self.cur_line, self.cur_column)

            if self.current_char == '+':
                if not (self.peek().isalnum() or self.peek() == ' '):
                    raise OperandError(self.current_char, self.cur_line, self.cur_column)
                self.advance()
                return Lexeme(PLUS, '+', self.cur_line, self.cur_column)

            if self.current_char == '%':
                if not (self.peek().isalnum() or self.peek() == ' '):
                    raise OperandError(self.current_char, self.cur_line, self.cur_column)
                self.advance()
                return Lexeme(MOD, '%', self.cur_line, self.cur_column)

            if self.current_char == '!':
                if self.peek() == '=':
                    self.advance()
                    if not (self.peek().isalnum() or self.peek() == ' '):
                        raise OperandError(self.current_char, self.cur_line, self.cur_column)
                    self.advance()
                    token = Lexeme(NOT_EQUAL, '!=', self.cur_line, self.cur_column)
                else:
                    if not (self.peek().isalnum() or self.peek() == ' '):
                        raise OperandError(self.current_char, self.cur_line, self.cur_column)
                    self.advance()
                    token = Lexeme(NOT, '!', self.cur_line, self.cur_column)
                return token

            if self.current_char == '-':
                if not (self.peek().isalnum() or self.peek() == ' '):
                    raise OperandError(self.current_char, self.cur_line, self.cur_column)
                self.advance()
                return Lexeme(MINUS, '-', self.cur_line, self.cur_column)

            if self.current_char == '&':
                if not (self.peek().isalnum() or self.peek() == ' '):
                    raise OperandError(self.current_char, self.cur_line, self.cur_column)
                self.advance()
                return Lexeme(AND, '&', self.cur_line, self.cur_column)

            if self.current_char == '|':
                if not (self.peek().isalnum() or self.peek() == ' '):
                    raise OperandError(self.current_char, self.cur_line, self.cur_column)
                self.advance()
                return Lexeme(OR, '|', self.cur_line, self.cur_column)

            if self.current_char == '*':
                if not (self.peek().isalnum() or self.peek() == ' '):
                    raise OperandError(self.current_char, self.cur_line, self.cur_column)
                self.advance()
                return Lexeme(MUL, '*', self.cur_line, self.cur_column)

            if self.current_char == '/':
                if not (self.peek().isalnum() or self.peek() == ' '):
                    raise OperandError(self.current_char, self.cur_line, self.cur_column)
                self.advance()
                return Lexeme(DIV, '/', self.cur_line, self.cur_column)

            if self.current_char == '(':
                self.advance()
                return Lexeme(LPAREN, '(', self.cur_line, self.cur_column)

            if self.current_char == ')':
                self.advance()
                return Lexeme(RPAREN, ')', self.cur_line, self.cur_column)

            if self.current_char == '.':
                self.advance()
                return Lexeme(DOT, '.', self.cur_line, self.cur_column)

            raise UnknownSymbol(self.current_char, self.cur_line, self.cur_column)

        return Lexeme(EOF, None, self.cur_line, self.cur_column)
