import sys

INTEGER = 'INTEGER'
DOUBLE = 'DOUBLE'
STRING = 'STRING'
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
SCAN = 'SCAN'
PRINT = 'PRINT'
EXIT = 'EXIT'

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


class Token(object):
    def __init__(self, type, value, line, column):
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


RESERVED_KEYWORDS = {
    'int': INTEGER,
    'double': DOUBLE,
    'string': STRING,
    'while': WHILE,
    'if': IF,
    'else': ELSE,
    'continue': CONTINUE,
    'break': BREAK,
    'to_int': TO_INT,
    'to_str': TO_STR,
    'to_double': TO_DOUBLE,
    'scan': SCAN,
    'print': PRINT,
    'exit': EXIT
}

CONSTANTS = {
    'INTEGER_CONST': INTEGER_CONST,
    'DOUBLE_CONST': DOUBLE_CONST,
    'BOOL_CONST': BOOL_CONST,
    'STRING_LITERAL': STRING_LITERAL,
}


class LexerError(Exception):
    def __init__(self, message, line: int, index: int):
        self.message = message
        self.line = line
        self.index = index
        super().__init__(message)


class OperandError(LexerError):
    def __init__(self, char, line, column):
        super().__init__(f"Wrong usage of operand near '{char}' ", line, column)


class FloatNumberError(LexerError):
    def __init__(self, char, line, column):
        super().__init__(f"Wrong floating number near '{char}': ", line, column)


class NumberError(LexerError):
    def __init__(self, char, line, column):
        super().__init__(f"Wrong number near '{char}': ", line, column)


class UnknownSymbol(LexerError):
    def __init__(self, symbol, line, column):
        super().__init__("unknown symbol: " + symbol, line, column)


class NoMatchingQuotes(LexerError):
    def __init__(self, symbol, line, column):
        super().__init__("No matching quotes: " + symbol, line, column)


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
        token = Token('STRING_LITERAL', result, self.cur_line, self.cur_column)
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

            token = Token('DOUBLE_CONST', float(result), self.cur_line, self.cur_column)
        elif self.current_char not in OPERATORS.keys() and self.current_char not in (' ', ';'):
            raise NumberError(self.current_char, self.cur_line, self.cur_column)
        else:
            token = Token('INTEGER_CONST', int(result), self.cur_line, self.cur_column)
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
            token = Token(RESERVED_KEYWORDS[result], result, self.cur_line, self.cur_column)
        else:
            token = Token(ID, result, self.cur_line, self.cur_column)
        return token

    def get_next_token(self):
        while self.current_char is not None:

            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char == '{':
                self.advance()
                return Token(LBRACE, '{', self.cur_line, self.cur_column)

            if self.current_char == '}':
                self.advance()
                return Token(RBRACE, '}', self.cur_line, self.cur_column)

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
                    token = Token(EQUAL, '==', self.cur_line, self.cur_column)
                    print("type = ", token.type)
                else:
                    if not (self.peek().isalnum() or self.peek() == ' '):
                        raise OperandError(self.current_char, self.cur_line, self.cur_column)
                    self.advance()
                    token = Token(ASSIGN, '=', self.cur_line, self.cur_column)
                return token

            if self.current_char == '<':
                if self.peek() == '=':
                    self.advance()
                    if not (self.peek().isalnum() or self.peek() == ' '):
                        raise OperandError(self.current_char, self.cur_line, self.cur_column)
                    self.advance()
                    token = Token(LESS_OR_EQUAL, '<=', self.cur_line, self.cur_column)
                else:
                    if not (self.peek().isalnum() or self.peek() == ' '):
                        raise OperandError(self.current_char, self.cur_line, self.cur_column)
                    self.advance()
                    token = Token(LESS, '<', self.cur_line, self.cur_column)
                return token

            if self.current_char == '>':
                if self.peek() == '=':
                    self.advance()
                    if not (self.peek().isalnum() or self.peek() == ' '):
                        raise OperandError(self.current_char, self.cur_line, self.cur_column)
                    self.advance()
                    token = Token(MORE_OR_EQUAL, '>=', self.cur_line, self.cur_column)
                else:
                    if not (self.peek().isalnum() or self.peek() == ' '):
                        raise OperandError(self.current_char, self.cur_line, self.cur_column)
                    self.advance()
                    token = Token(MORE, '>', self.cur_line, self.cur_column)
                return token

            if self.current_char == ';':
                self.advance()
                return Token(SEMI, ';', self.cur_line, self.cur_column)

            if self.current_char == ',':
                self.advance()
                return Token(COMMA, ',', self.cur_line, self.cur_column)

            if self.current_char == '+':
                if not (self.peek().isalnum() or self.peek() == ' '):
                    raise OperandError(self.current_char, self.cur_line, self.cur_column)
                self.advance()
                return Token(PLUS, '+', self.cur_line, self.cur_column)

            if self.current_char == '%':
                if not (self.peek().isalnum() or self.peek() == ' '):
                    raise OperandError(self.current_char, self.cur_line, self.cur_column)
                self.advance()
                return Token(MOD, '%', self.cur_line, self.cur_column)

            if self.current_char == '!':
                if self.peek() == '=':
                    self.advance()
                    if not (self.peek().isalnum() or self.peek() == ' '):
                        raise OperandError(self.current_char, self.cur_line, self.cur_column)
                    self.advance()
                    token = Token(NOT_EQUAL, '!=', self.cur_line, self.cur_column)
                else:
                    if not (self.peek().isalnum() or self.peek() == ' '):
                        raise OperandError(self.current_char, self.cur_line, self.cur_column)
                    self.advance()
                    token = Token(NOT, '!', self.cur_line, self.cur_column)
                return token

            if self.current_char == '-':
                if not (self.peek().isalnum() or self.peek() == ' '):
                    raise OperandError(self.current_char, self.cur_line, self.cur_column)
                self.advance()
                return Token(MINUS, '-', self.cur_line, self.cur_column)

            if self.current_char == '&':
                if not (self.peek().isalnum() or self.peek() == ' '):
                    raise OperandError(self.current_char, self.cur_line, self.cur_column)
                self.advance()
                return Token(AND, '&', self.cur_line, self.cur_column)

            if self.current_char == '|':
                if not (self.peek().isalnum() or self.peek() == ' '):
                    raise OperandError(self.current_char, self.cur_line, self.cur_column)
                self.advance()
                return Token(OR, '|', self.cur_line, self.cur_column)

            if self.current_char == '*':
                if not (self.peek().isalnum() or self.peek() == ' '):
                    raise OperandError(self.current_char, self.cur_line, self.cur_column)
                self.advance()
                return Token(MUL, '*', self.cur_line, self.cur_column)

            if self.current_char == '/':
                if not (self.peek().isalnum() or self.peek() == ' '):
                    raise OperandError(self.current_char, self.cur_line, self.cur_column)
                self.advance()
                return Token(DIV, '/', self.cur_line, self.cur_column)

            if self.current_char == '(':
                self.advance()
                return Token(LPAREN, '(', self.cur_line, self.cur_column)

            if self.current_char == ')':
                self.advance()
                return Token(RPAREN, ')', self.cur_line, self.cur_column)

            if self.current_char == '.':
                self.advance()
                return Token(DOT, '.', self.cur_line, self.cur_column)

            raise UnknownSymbol(self.current_char, self.cur_line, self.cur_column)

        return Token(EOF, None, self.cur_line, self.cur_column)


def main():
    with open('input.txt') as f:
        text = f.read()
    lexer = Lexer(text)
    symbolTable = SymbolTable(lexer, OPERATORS, RESERVED_KEYWORDS, CONSTANTS)
    try:
        symbolTable.fulfill()
    except LexerError as err:
        print(f"LEXER ERROR: {err.message} line {err.line} column {err.index}")
        sys.exit(1)
    print(symbolTable)

    # operators_table = []
    # variables_table = []
    # constants_table = []
    # key_words_table = []
    # general_table = []
    #
    # tables = ([operators_table, variables_table, constants_table, key_words_table])
    # # op_ind, var_index, const_index, key_index = 0, 0, 0, 0
    #
    # try:
    #     current_token = lexer.get_next_token()
    #     while current_token.type != EOF:
    #         if current_token.type in OPERATORS.values():
    #             if current_token.value not in operators_table:
    #                 operators_table.append(current_token.value)
    #             general_table.append((0, operators_table.index(current_token.value),
    #                                   current_token.line, current_token.column))
    #
    #         elif current_token in RESERVED_KEYWORDS.values():
    #             if current_token.value not in key_words_table:
    #                 key_words_table.append(current_token.value)
    #             general_table.append((1, key_words_table.index(current_token.value),
    #                                   current_token.line, current_token.column))
    #         elif current_token.type in CONSTANTS:
    #             if current_token.value not in constants_table:
    #                 constants_table.append(current_token.value)
    #             general_table.append((2, constants_table.index(current_token.value),
    #                                   current_token.line, current_token.column))
    #         else:
    #             if current_token.value not in variables_table:
    #                 variables_table.append(current_token.value)
    #             general_table.append((3, variables_table.index(current_token.value),
    #                                   current_token.line, current_token.column))
    #
    #         current_token = lexer.get_next_token()
    #
    # except LexerError as err:
    #     print(f"LEXER ERROR: {err.message} line {err.line} column {err.index})")
    #     sys.exit(1)

    # for table in tables:
    #     print("----------------------------------")
    #     for i in range(len(table)):
    #         print(i, "|", table[i])
    # print("General table:", general_table)


class SymbolTable(object):
    def __init__(self, lexer: Lexer, operators, reserved, constants):
        self.lexer = lexer
        self.operators_table = []
        self.variables_table = []
        self.constants_table = []
        self.key_words_table = []
        self.tables = ([self.operators_table,
                        self.variables_table,
                        self.constants_table,
                        self.key_words_table])
        self.general_table = []
        self.operators = operators
        self.reserved_keywords = reserved
        self.constants = constants

    def __str__(self):
        s = ""
        for table in self.tables:
            for item in table:
                s += "{index} | {var}\n".format(index=table.index(item),var=item)
            s += "--------------------------------------\n"
        for item in self.general_table:
            s += "table: {t}, index : {i}, line {l}, column {c}\n".format(t=item[0],
                                                                         i=item[1],
                                                                         l=item[2],
                                                                         c=item[3])
        return s

    # __repr__ = __str__

    def fulfill(self):
        current_token = self.lexer.get_next_token()
        while current_token.type != EOF:
            if current_token.type in self.operators.values():
                self._add_to_table(self.operators_table, current_token)
            elif current_token.type in self.reserved_keywords.values():
                self._add_to_table(self.key_words_table, current_token)
            elif current_token.type in self.constants:
                self._add_to_table(self.constants_table, current_token)
            else:
                self._add_to_table(self.variables_table, current_token)
            current_token = self.lexer.get_next_token()

    def _add_to_table(self, table, token):
        if token.value not in table:
            table.append(token.value)
        self.general_table.append((self.tables.index(table),
                                   table.index(token.value),
                                   token.line,
                                   token.column))



if __name__ == '__main__':
    main()
