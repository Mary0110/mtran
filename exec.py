from const import Constant, INT, DOUBL, STRIN
import lexer

class InterpreterUtils:
    operators_table = None
    keywords_table = None
    constants_table = None

    @classmethod
    def initialize(cls, op, key, const):
        cls.operators_table = op
        cls.keywords_table = key
        cls.constants_table = const

    @staticmethod
    def first_child(node):
        return node.children[0]

    @staticmethod
    def second_child(node):
        return node.children[1]

    @classmethod
    def op(cls, tok, op):
        if tok.table is not cls.operators_table:
            return False

        if isinstance(op, tuple):
            return tok.value() in op
        return tok.value() == op

    @classmethod
    def keyword(cls, tok, keyword):
        if tok.table is not cls.keywords_table:
            return False

        if isinstance(keyword, tuple):
            return tok.value() in keyword
        return tok.value() == keyword

    def is_type_and_const(self, tok, tp):
        return tok.table is self.constants_table and tok.value().type == tp


class Translator:
    OPERATORS = {
        lexer.BOOL: "bool",
        lexer.WHILE: "while",
        lexer.IF: "if",
        lexer.BREAK: "break",
        lexer.EQUAL:"=",
        lexer.NOT_EQUAL: "!=",
        lexer.MUL: "*",
        lexer.DIV: "/",
        lexer.PLUS: "+",
        lexer.MINUS: "-",
        lexer.TO_DOUBLE: "to_double",
        lexer.TO_BOOL: "to_bool",
        lexer.TO_INT: "to_int",
        lexer.MOD: "%",
        lexer.OR: "|",
        lexer.AND: "&",
        lexer.LESS: "<",
        lexer.MORE: ">",
        lexer.LESS_OR_EQUAL: "<=",
        lexer.MORE_OR_EQUAL: ">=",
        lexer.EEQUAL: "==",
        lexer.NOT: "!",
        lexer.SCAN: "scan"
    }

    def __init__(self, parser_nodes, operators, identifiers, keywords,
                 consts):
        self.ops = operators
        self.keys = keywords
        self.consts = consts
        InterpreterUtils.initialize(self.ops, self.keys, self.consts)
        self.vars = identifiers
        self._marks = parser_nodes



    class ExecutionError(Exception):
        def __init__(self, msg, line, column, id = None):
            self.id = id
            self.txt = msg
            self.line = line
            self.column = column
            super().__init__(msg)

        def __repr__(self):
            return "{} ({}:{})".format(self.txt, self.line, self.column)

        def __str__(self):
            return "{} ({}:{})".format(self.txt, self.line, self.column)

    def _run_assignment(self, assignment_node):
        left_node = assignment_node.children[0]
        value_node = assignment_node.children[1]
        value = self.exec(value_node)
        self.vars[left_node.index_in_table].value = value

    def _arr(self, sizes: list, default_value):
        if not sizes:
            return default_value

        ret = [self._arr(sizes[1:], default_value) for _ in range(sizes[0])]
        return ret

    def _run_declare(self, decl_node):
        for var_node in decl_node.children:
            if InterpreterUtils.op(var_node, self.OPERATORS[lexer.EQUAL]):
                self._run_assignment(var_node)
            else:
                var_type = var_node.value().type

                if not isinstance(var_type, list):
                    if var_type == "int":
                        self.vars[var_node.index_in_table].value = 0
                    elif var_type == "double":
                        self.vars[var_node.index_in_table].value = 0.0
                    elif var_type == "string":
                        return ""
                    if var_type == "bool":
                        self.vars[var_node.index_in_table].value = False
                    else:
                        self.vars[var_node.index_in_table].value = None
                else:
                    if var_type == "int":
                        default_value = 0
                    elif var_type == "double":
                        default_value = 0.0
                    elif var_type == "string":
                        default_value = ""
                    elif var_type == "bool":
                        default_value = False
                    else:
                        default_value = None
                    self.vars[var_node.index_in_table].value = self._arr(var_type[1:], default_value)

    def _run_if(self, if_node):
        condition_node = InterpreterUtils.first_child(if_node)
        statement_if_true = InterpreterUtils.second_child(if_node)
        statement_if_false = if_node.children[2] if len(if_node.children) > 2 else None

        execute = self.exec if self.exec(condition_node) else (
            lambda x: None if statement_if_false is None else self.exec(statement_if_false))
        execute(statement_if_true)

    class Break(Exception):
        pass

    def _run_while(self, while_node):
        condition = InterpreterUtils.first_child(while_node)
        body = InterpreterUtils.second_child(while_node)

        while self.exec(condition):
            result = self.exec(body)
            if isinstance(result, Translator.Break):
                break

    def exec(self, node):
        try:
            if node.table[node.index_in_table] == "PROG":
                for stmt_node in node.children:
                    self.exec(stmt_node)
            elif node.table is self._marks and node.table[node.index_in_table] == "BODY":
                for stmt_node in node.children:
                    self.exec(stmt_node)
            elif InterpreterUtils.keyword(node, "prin"):
                print(self.exec(InterpreterUtils.first_child(node)), end='')
            elif node.table is self._marks and node.table[node.index_in_table] == "VAR":
                self._run_declare(node)
            elif InterpreterUtils.op(node,self.OPERATORS[lexer.EQUAL]):
                self._run_assignment(node)
            elif node.table is self.consts and node.table[node.index_in_table].type == STRIN:
                str_to_print =node.table[node.index_in_table].value
                return bytes(str_to_print, "utf-8").decode("unicode_escape")
            elif node.table is self.consts and node.table[node.index_in_table].type == INT:
                return int(node.table[node.index_in_table].value)
            elif node.table is self.consts and node.table[node.index_in_table].type == DOUBL:
                return float(node.table[node.index_in_table].value)
            elif InterpreterUtils.keyword(node, ("true", "false")):
                return node.table[node.index_in_table] == "true"
            elif node.table is self.vars:
                return node.table[node.index_in_table].value
            elif InterpreterUtils.keyword(node, "to_string"):
                result = str(self.exec(InterpreterUtils.first_child(node)))
                if result == "True":
                    return "true"
                elif result == "False":
                    return "false"
                else:
                    return result
            elif InterpreterUtils.keyword(node, self.OPERATORS[lexer.TO_INT]):
                try:
                    return int(self.exec(InterpreterUtils.first_child(node)))
                except Exception:
                    raise Translator.ExecutionError("Cannot convert input to int", node.line, node.column)
            elif InterpreterUtils.keyword(node,  self.OPERATORS[lexer.TO_DOUBLE]):
                try:
                    return float(self.exec(InterpreterUtils.first_child(node)))
                except Exception:
                    raise Translator.ExecutionError("Cannot convert input to double", node.line, node.column)
            elif InterpreterUtils.keyword(node, self.OPERATORS[lexer.TO_BOOL]):
                try:
                    return bool(self.exec(InterpreterUtils.first_child(node)))
                except Exception:
                    raise Translator.ExecutionError("Cannot convert input to bool", node.line, node.column)

            elif InterpreterUtils.keyword(node,  self.OPERATORS[lexer.SCAN]):
                return input()
            elif InterpreterUtils.op(node, self.OPERATORS[lexer.PLUS]):
                try:
                    return self.exec(InterpreterUtils.first_child(node))
                except:
                    return self.exec(InterpreterUtils.first_child(node)) + self.exec(
                        InterpreterUtils.second_child(node))

            elif InterpreterUtils.op(node, self.OPERATORS[lexer.MINUS]):
                if len(node.children) <= 1:
                    return -self.exec(InterpreterUtils.first_child(node))
                else:
                    return self.exec(InterpreterUtils.first_child(node)) - self.exec(
                        InterpreterUtils.second_child(node))
            elif InterpreterUtils.op(node,self.OPERATORS[lexer.MUL]):
                return self.exec(InterpreterUtils.first_child(node)) * self.exec(InterpreterUtils.second_child(node))
            elif InterpreterUtils.op(node, self.OPERATORS[lexer.MOD]):
                return self.exec(InterpreterUtils.first_child(node)) % self.exec(InterpreterUtils.second_child(node))
            elif InterpreterUtils.op(node, self.OPERATORS[lexer.DIV]):
                return self.exec(InterpreterUtils.first_child(node)) / self.exec(InterpreterUtils.second_child(node))
            elif InterpreterUtils.op(node, self.OPERATORS[lexer.AND]):
                return self.exec(InterpreterUtils.first_child(node)) and self.exec(InterpreterUtils.second_child(node))
            elif InterpreterUtils.op(node, self.OPERATORS[lexer.OR]):
                return self.exec(InterpreterUtils.first_child(node)) or self.exec(InterpreterUtils.second_child(node))
            elif InterpreterUtils.op(node, self.OPERATORS[lexer.NOT]):
                return not self.exec(InterpreterUtils.first_child(node))
            elif InterpreterUtils.op(node, self.OPERATORS[lexer.MORE]):
                return self.exec(InterpreterUtils.first_child(node)) > self.exec(InterpreterUtils.second_child(node))
            elif InterpreterUtils.op(node, self.OPERATORS[lexer.LESS]):
                return self.exec(InterpreterUtils.first_child(node)) < self.exec(InterpreterUtils.second_child(node))
            elif InterpreterUtils.op(node, self.OPERATORS[lexer.MORE_OR_EQUAL]):
                return self.exec(InterpreterUtils.first_child(node)) >= self.exec(InterpreterUtils.second_child(node))
            elif InterpreterUtils.op(node, self.OPERATORS[lexer.LESS_OR_EQUAL]):
                return self.exec(InterpreterUtils.first_child(node)) <= self.exec(InterpreterUtils.second_child(node))
            elif InterpreterUtils.op(node, self.OPERATORS[lexer.EEQUAL]):
                return self.exec(InterpreterUtils.first_child(node)) == self.exec(InterpreterUtils.second_child(node))
            elif InterpreterUtils.op(node, self.OPERATORS[lexer.NOT_EQUAL]):
                return self.exec(InterpreterUtils.first_child(node)) != self.exec(InterpreterUtils.second_child(node))
            elif InterpreterUtils.keyword(node, self.OPERATORS[lexer.WHILE]):
                self._run_while(node)
            elif InterpreterUtils.keyword(node, self.OPERATORS[lexer.IF]):
                self._run_if(node)
            elif InterpreterUtils.keyword(node, self.OPERATORS[lexer.BREAK]):
                raise Translator.Break()
        except Exception:
            raise Translator.ExecutionError("not found", node.line, node.column)
