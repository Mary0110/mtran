from dataclasses import dataclass

from lexer import Lexer, EOF, Token_to_parse

@dataclass()
class SymbolTable(object):
    def __init__(self, lexer: Lexer, operators, reserved, constants, name = None, id = None):
        self.id = id
        self.lexer = lexer
        self.operators_table = []
        self.variables_table = []
        self.name = name
        self.constants_table = []
        self.key_words_table = []

        self.general_table = []
        self.operators = operators
        self.reserved_keywords = reserved
        self.constants = constants
        self.tables = ([self.operators_table,
                        self.variables_table,
                        self.constants_table,
                        self.key_words_table])

    def __str__(self):
        s = ""
        s += "operators table:\n"
        s += "--------------------------------------\n"

        for item in self.operators_table:
            s += "{index} | {var}\n".format(index=self.operators_table.index(item), var=item)
        s += "--------------------------------------\n"
        s += "variables table:\n"
        s += "--------------------------------------\n"

        for item in self.variables_table:
            s += "{index} | {var}\n".format(index=self.variables_table.index(item), var=item)
        s += "--------------------------------------\n"
        s += "constants table:\n"
        s += "--------------------------------------\n"

        for item in self.constants_table:
            s += "{index} | {var}\n".format(index=self.constants_table.index(item), var=item)
        s += "--------------------------------------\n"
        s += "reserved keywords table:\n"
        s += "--------------------------------------\n"

        for item in self.key_words_table:
            s += "{index} | {var}\n".format(index=self.key_words_table.index(item), var=item)
        s += "--------------------------------------\n"

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

        self.general_table.append(Token_to_parse(self.tables.index(table),
                                                 table.index(token.value),
                                                 token.line,
                                                 token.column))
