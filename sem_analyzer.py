from const import DOUBL, INT


class SemAnalyzer:
    class SemError(Exception):
        def __init__(self, message, line, column, id=None):
            self.id = id
            self.message = message
            self.line = line
            self.column = column
            super().__init__(message)

        def __str__(self) -> str:
            return "{} ({}:{})".format(self.message, self.line, self.column)

    class WrongModUsage(SemError):
        def __init__(self, line, column):
            super().__init__("mod must work with integer operands", line, column)

        def __str__(self):
            return f"{self.message} ({self.line}:{self.column})"

    def __init__(self, operators, variables, keywords,
                 consts):
        self._operators = operators
        self._vars = variables
        self._keywords = keywords
        self._consts = consts

    def passing(self, node):
        if node.table[node.index_in_table] == "/":
            _operator = node.children[1]
            if float(_operator.table[_operator.index_in_table].value) == 0:
                raise SemAnalyzer.DivisionByZero(_operator.line, _operator.column)
        elif node.table[node.index_in_table] == "%":
            self._check_int_operations(node.children[0])
            self._check_int_operations(node.children[1])
        else:
            for child in node.children:
                self.passing(child)

    def _check_int_operations(self, _node):
        if _node.table is self._consts and _node.table[_node.index_in_table].type == DOUBL:
            raise SemAnalyzer.WrongModUsage(_node.line, _node.column)
        elif _node.table is self._keywords and (_node.table[_node.index_in_table] == "to_double" or
                                                _node.table[_node.index_in_table] == "to_int"):
            raise SemAnalyzer.WrongModUsage(_node.line, _node.column)
        elif _node.table is self._vars and _node.table[_node.index_in_table].type == "double":
            raise SemAnalyzer.WrongModUsage(_node.line, _node.column)
        else:
            for child in _node.children:
                self._check_int_operations(child)

    class DivisionByZero(SemError):
        def __init__(self, line, column):
            super().__init__("division by zero", line, column)

        def __str__(self):
            return f"{self.message} ({self.line}:{self.column})"
