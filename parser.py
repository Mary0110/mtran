import sys
from var import Var
from const import Constant, DOUBL, INT, STRIN

type_keywords = ("int", "double", "string", "bool")

class CheckType:
    parser_tables = None

    @classmethod
    def initialize(cls, _parser_tables):
        cls.parser_tables = _parser_tables

    @classmethod
    def is_table_index(cls, tok, table_name):
        r = tok.table_index == cls.parser_tables.tables.index(getattr(cls.parser_tables, table_name))
        a = getattr(cls.parser_tables, table_name)
        return r

    @classmethod
    def is_in_table(cls, tok, table_name, values):
        k = cls.is_table_index(tok, table_name) and cls.parser_tables.tables[tok.table_index][
            tok.index] in values
        v = cls.parser_tables.tables[tok.table_index][
            tok.index]
        return cls.is_table_index(tok, table_name) and cls.parser_tables.tables[tok.table_index][
            tok.index] in values

    @classmethod
    def is_operator(cls, tok, op):
        return cls.is_table_index(tok, 'operators') and cls.parser_tables.tables[tok.table_index][
            tok.index] in op

    @classmethod
    def is_addop(cls, tok):
        return cls.is_in_table(tok, 'operators', ('+', '-'))

    @classmethod
    def is_mulop(cls, tok):
        return cls.is_in_table(tok, 'operators', ('*', '/', '%'))

    @classmethod
    def is_keyword(cls, tok, keyword):
        return cls.is_in_table(tok, 'keywords', keyword)

    @classmethod
    def is_variable(cls, tok):
        return cls.is_table_index(tok, 'vars')

    @classmethod
    def is_var_of_type(cls, tok, type_):
        return cls.is_variable(tok) and cls.parser_tables.tables[tok.table_index][tok.index].type == type_


class ParserTables:
    def __init__(self, ops_tbl, variables_tbl, keywords_tbl, consts_tbl, parser_nodes_tbl, tables, id = None):
        self.operators = ops_tbl
        self.vars = variables_tbl
        self.keywords = keywords_tbl
        self.name = None
        self.consts = consts_tbl
        self.tables = tables
        self.parser_nodes = parser_nodes_tbl
        self.tables.append(self.parser_nodes)

    def get_nodes(self):
        return self.parser_nodes

    def get_operands(self):
        return self.operators

    def get_vars(self):
        return self.vars

    def get_keywords(self):
        return self.keywords

    def get_consts(self):
        return self.consts


class ParserUtils:
    @staticmethod
    def eat_op(tok, op):
        if not CheckType.is_operator(tok, op):
            raise Parser.Expected(str(op), tok.line, tok.column)

    @staticmethod
    def check_is_keyword(tok, keyword):
        if not CheckType.is_keyword(tok, keyword):
            raise Parser.Expected(str(keyword), tok.line, tok.column)

    @staticmethod
    def print_tree(node, level=0):
        if not node:
            return

        print('-' * level + str(node))
        for child in node.children:
            ParserUtils.print_tree(child, level + 1)


class Parser:
    def __init__(self, general_table, tables):
        self.tokens = general_table
        self.cu_tok_idx = 0
        self.pr_lvl = 0
        self.blocks_on_lvls = [1]
        self.visibility_list = [(0, 1)]

        self.parser_tables = ParserTables(ops_tbl=tables[0], variables_tbl=tables[1], keywords_tbl=tables[3],
                                          consts_tbl=tables[2],
                                          parser_nodes_tbl=["PROG", "VAR", "BODY"],
                                          tables=tables)
        self.num_of_cond = 0
        CheckType.initialize(self.parser_tables)


    class SyntaxError(Exception):
        def __init__(self, text, line, column, id = None):
            self.id = id
            self.error_text = text
            self.line = line
            self.column = column
            super().__init__(text)

        def __repr__(self):
            return f"SyntaxError({self.error_text!r}, {self.line}, {self.column})"
        def __str__(self):
            return f"{self.args[0]} ({self.line}:{self.column})"

    class ExpectedComparisonOperator(SyntaxError):
        def __init__(self, text, line, column):
            super().__init__(text, line, column)

        def __str__(self):
            return f"Expected a comparison operator at line {self.line}, column {self.column}."

    class BooleanParamErr(SyntaxError):
        def __init__(self, msg: str, line: int, col):
            super().__init__(msg, line, col)

        def __str__(self) -> str:
            res = f"{super().__str__()}\n"
            return res

    class Expected(SyntaxError):
        def __init__(self, expected, line, index):
            super().__init__(f"{expected} expected", line, index)

        def __str__(self):
            return f"expected at line {self.line}, column {self.column}"

    class EndOfFile(SyntaxError):
        def __init__(self, what_is_unexpected, line, index):
            super().__init__(f"end of file {what_is_unexpected}", line, index)

        def __str__(self):
            return f"{self.msg} at line {self.line}, column {self.column}"

    class CompErr(SyntaxError):
        def __init__(self, line, index):
            super().__init__("cannot compare string and number", line, index)

        def __str__(self):
            return f"{self.msg} at line {self.line}, column {self.column}"

    class DoubleDeclaration(SyntaxError):
        def __init__(self, name, line, index):
            super().__init__(f"double declaration of {name}", line, index)

        def __str__(self):
            return f"{self.msg} at line {self.line}, column {self.column}"

    class DeclarationError(SyntaxError):
        def __init__(self, name, line, index):
            super().__init__(f"using not declared variable {name}", line, index)

        def __str__(self):
            return f"{self.msg} at line {self.line}, column {self.index}"

    class InvalidVarType(SyntaxError):
        def __init__(self, tp, expected_type, line, index):
            super().__init__(f"{tp} variable cannot be used in this expression ({expected_type} expected)", line, index)

        def __str__(self):
            return f"{self.msg} at line {self.line}, column {self.index}"

    class ForbiddenStatement(SyntaxError):
        def __init__(self, stmt, line, index):
            super().__init__(f"{stmt} cannot be used in this block", line, index)

        def __str__(self):
            return f"{self.msg} at line {self.line}, column {self.index}"

    def advance(self):
        self.cu_tok_idx += 1

    class Node:
        def __init__(self, tbl=None, index_in_tbl=None, children=None, line=0, column=0):
            if children is None:
                children = []

            self.children = children
            self.table = tbl
            self.index_in_table = index_in_tbl
            self.line = line
            self.column = column

        def __str__(self):
            return "NODE(" + str(self.table[self.index_in_table]) + ")"

    def _empty_tokens_list(self):
        return self.cu_tok_idx >= len(self.tokens)

    def _add_children(self, node, child1, child2=None):
        if(child2 is None):
            node.children = [child1]

        else:
            node.children=([child1, child2])
    def _present_token(self):
        if self._empty_tokens_list():
            last_tok = self.tokens[-1]
            line = last_tok.line
            index = last_tok.index + len(self.parser_tables.tables[last_tok.table_index][last_tok.index])
            raise Parser.EndOfFile("EOF", line, index)
        r = self.tokens[self.cu_tok_idx]
        return self.tokens[self.cu_tok_idx]

    def _parse_operator(self):
        tok = self._present_token()
        res_node = Parser.Node(self.parser_tables.tables[tok.table_index], tok.index, line=tok.line, column=tok.column)
        self.advance()
        return res_node

    def get_scope_and_revised_indices(self,visibility_list):
        scope_idx = len(visibility_list) - 1
        revised_idx = -1
        return scope_idx, revised_idx

    def _parse_variable_in_use(self, type=None):
        tok = self._present_token()

        res_node = Parser.Node(self.parser_tables.tables[tok.table_index], tok.index, line=tok.line, column=tok.column)

        if res_node.table[res_node.index_in_table].type is None:
            raise Parser.DeclarationError(res_node.table[res_node.index_in_table].name, res_node.line,
                                          res_node.column)
        scope_idx, revised_idx = self.get_scope_and_revised_indices(self.visibility_list)

        while scope_idx >= 0:
            pr_bl = self.visibility_list[scope_idx]
            cur_var = Var(res_node.table[res_node.index_in_table].name,
                          res_node.table[res_node.index_in_table].type, pr_bl[0], pr_bl[1])

            found_var = False
            for i, var in enumerate(self.parser_tables.vars):
                if var == cur_var:
                    revised_idx = i
                    found_var = True
                    break

            if found_var:
                break
            else:
                scope_idx -= 1

        if revised_idx < 0:
            raise Parser.DeclarationError(res_node.table[res_node.index_in_table].name, res_node.line,
                                          res_node.column)
        res_node.index_in_table = revised_idx

        self.advance()
        return res_node

    def _is_to_int_double_bool(self, tok):
        return tok.table_index == self.parser_tables.tables.index(self.parser_tables.keywords) and \
               self.parser_tables.tables[tok.table_index][tok.index] in ("to_int", "to_double", "to_bool")

    def _parse_to_int_double_bool_arg(self):
        self._check_is_open_paren()
        self.num_of_cond += 1
        res = self._parse_string()
        # TODO: Handle conversion from bool, int, and float
        self._check_is_close_paren()
        return res

    def _parse_oper(self):
        expr1 = self._parse_mathexpr()
        self.num_of_cond += 1
        while CheckType.is_mulop(self._present_token()):
            operator_node = self._parse_operator()
            expr2 = self._parse_mathexpr()
            self._add_children(operator_node, expr1, expr2)
            return operator_node

        return expr1

    def _parse_str_param(self):
        old_tok = self._present_token()
        old_token_index = self.cu_tok_idx

        res_node, tp = self._try_parse(self._parse_bool_expr, "bool")
        if res_node:
            return res_node, tp

        self.cu_tok_idx = old_token_index

        res_node, tp = self._try_parse(self._parse_string, "string")
        if res_node:
            return res_node, tp

        self.cu_tok_idx = old_token_index

        res_node, tp = self._try_parse(self._parse_num_expr, "numeric")
        if res_node:
            return res_node, tp

        raise Parser.BooleanParamErr(
            "Invalid expression: this is neither a string, arithmetic, nor boolean exp",
            old_tok.line, old_tok.column
        )

    def _parse_num_expr(self):
        tok = self._present_token()
        if CheckType.is_addop(tok):
            sign = self._parse_operator()
        else:
            sign = None

        term = self._parse_oper()

        if sign is None:
            term1 = term
        else:
            self._add_children(sign, term)
            term1 = sign
        while CheckType.is_addop(self._present_token()):
            add_op = self._parse_operator()
            rhs = self._parse_oper()
            self._add_children(add_op, term1, rhs)
            term1 = add_op

        return term1

    def _try_parse(self, func, tp):
        try:
            res_node = func()
            return res_node, tp
        except Parser.SyntaxError as e:
            return None, None

    def _parse_mathexpr(self):
        tok = self._present_token()

        if tok.table_index == self.parser_tables.tables.index(self.parser_tables.vars):
            var_types = ("int", "double")
            res_node = self._parse_variable_in_use(var_types)
        elif tok.table_index == self.parser_tables.tables.index(self.parser_tables.operators) and \
            self.parser_tables.tables[tok.table_index][tok.index] == '(':
            self.advance()
            res_node = self._parse_num_expr()
            self._check_is_close_paren()
        elif tok.table_index == self.parser_tables.tables.index(self.parser_tables.keywords) and \
                self.parser_tables.tables[tok.table_index][tok.index] in ("to_int", "to_double"):
            res_node = self._parse_to_int_double_bool()
        else:
            if self._is_number_token(tok):
                res_node = self._parse_number()
            else:
                raise Parser.Expected("number", tok.line, tok.column)

        return res_node

    def _is_number_token(self, tok):
        return tok.table_index == self.parser_tables.tables.index(self.parser_tables.consts) and \
               tok.index < len(self.parser_tables.consts) and \
               self.parser_tables.consts[tok.index].type in (DOUBL, INT)

    def _parse_number(self):
        tok = self._present_token()
        res_node = Parser.Node(self.parser_tables.consts, tok.index, None, tok.line, tok.column)
        self.advance()
        return res_node

    def _parse_scan(self):
        tok = self._present_token()
        if not CheckType.is_keyword(tok, "scan"):
            raise Parser.Expected("scan", tok.line, tok.column)

        op = Parser.Node(self.parser_tables.tables[tok.table_index], tok.index, line=tok.line, column=tok.column)
        self.advance()

        if not CheckType.is_operator(self._present_token(), "("):
            raise Parser.Expected("(", tok.line, self._present_token().column)
        self.advance()

        if not CheckType.is_operator(self._present_token(), ")"):
            raise Parser.Expected(")", tok.line, self._present_token().column)
        self.advance()

        return op

    def _parse_string(self):
        tok = self._present_token()

        if tok.table_index is self.parser_tables.tables.index(self.parser_tables.vars):
            res_node = self._parse_variable_in_use("string")
        elif CheckType.is_keyword(tok, "to_str"):
            tok = self._present_token()
            op = Parser.Node(self.parser_tables.tables[tok.table_index], tok.index, line=tok.line, column=tok.column)

            self.advance()
            self._check_is_open_paren()

            expr, _ = self._parse_str_param()

            self._check_is_close_paren()
            self._add_children(op, expr)
            res_node = op
        elif CheckType.is_keyword(tok, "scan"):
            res_node = self._parse_scan()
        else:
            if tok.table_index is not self.parser_tables.tables.index(self.parser_tables.consts) or \
                    self.parser_tables.tables[tok.table_index][
                        tok.index].type != STRIN:
                raise Parser.Expected("string", tok.line, tok.column)

            res_node = Parser.Node((self.parser_tables.consts), tok.index, None, tok.line, tok.column)
            self.advance()

        return res_node

    def _parse_comparison_operands(self):
        old_token_index = self.cu_tok_idx
        res_node, res_kind = None, None

        # Try to parse an arithmetic expression
        try:
            res_node = self._parse_num_expr()
            res_kind = "numeric"
        except Parser.SyntaxError as err1:
            self.cu_tok_idx = old_token_index

            # Try to parse a string expression
            try:
                res_node = self._parse_string()
                res_kind = "string"
            except Parser.SyntaxError as err2:
                pass
        possible_err_msg = "error while parsing comparison part"
        # Raise an error if neither an arithmetic nor a string expression could be parsed
        if res_node is None:
            raise Parser.BooleanParamErr(
                possible_err_msg, self._present_token().line, self._present_token().column
            )

        return res_node, res_kind

    def _parse_rel_op(self):
        left_operand, left_operand_type = self._parse_comparison_operands()
        operator = self._parse_comparison_operator()
        self.num_of_cond += 1
        right_operand, right_operand_type = self._parse_comparison_operands()

        self._check_operand_types(left_operand_type, right_operand_type, operator.line, operator.column)

        comp_op_node = Parser.Node(operator.table, operator.index_in_table, line=operator.line, column=operator.column)
        self._add_children(comp_op_node, left_operand, right_operand)
        return comp_op_node

    def _parse_comparison_operator(self):
        tok = self._present_token()
        operators = ('==', '>=', '<=', '>', '<', '!=')
        try:
            ParserUtils.eat_op(tok, operators)
        except Parser.SyntaxError:
            raise Parser.ExpectedComparisonOperator("Comparison error", tok.line, tok.column)
        operator_node = self._parse_operator()
        return operator_node

    def _check_operand_types(self, left_type, right_type, line, column):
        if left_type != right_type:
            raise Parser.CompErr(line, column)

    def _parse_to_int_double_bool(self):
        tok = self._present_token()
        if not self._is_to_int_double_bool(tok):
            raise Parser.Expected("to_int, to_double or to_bool", tok.line, tok.column)

        op_node = Parser.Node(self.parser_tables.tables[tok.table_index], tok.index, line=tok.line, column=tok.column)
        self.advance()

        arg_node = self._parse_to_int_double_bool_arg()
        self._add_children(op_node, arg_node)
        return op_node

    def _parse_bool_as_param(self):
        factor = self._parse_bool()
        while self._present_token() == '&':
            _and = self._parse_and_operator()
            self._add_children(_and, factor,  self._parse_bool())
            factor = _and
        return factor

    def _parse_and_operator(self):
        op_tok = self._present_token()
        self.num_of_cond += 1
        ParserUtils.eat_op(op_tok, '&')
        op = Parser.Node(self.parser_tables.tables[op_tok.table_index], op_tok.index, line=op_tok.line,
                         column=op_tok.column)
        self.advance()
        return op

    def _parse_initialization(self, type_node, identifier_node):
        if not CheckType.is_operator(self._present_token(), '='):
            return identifier_node

        assignment_node = self._parse_operator()
        var_type = type_node.table[type_node.index_in_table]
        rhs = self._parse_rhs_expression(var_type)
        self._add_children(assignment_node,identifier_node,rhs)
        return assignment_node

    def _parse_rhs_expression(self, var_type):
        if var_type == "bool":
            return self._parse_bool_expr()
        elif var_type == "int" or var_type == "double":
            return self._parse_num_expr()
        else:
            return self._parse_string()

    def _parse_bool_expr(self):
        left_term = self._parse_bool_as_param()
        self.num_of_cond += 1
        while CheckType.is_operator(self._present_token(), '|'):
            tok = self._present_token()
            res_node = Parser.Node(self.parser_tables.tables[tok.table_index], tok.index, line=tok.line,
                                   column=tok.column)
            self.advance()
            operator = res_node
            right_term = self._parse_bool_as_param()
            self._add_children(operator, left_term, right_term)
            left_term = operator

        return left_term

    def _parse_declare_var(self):
        tok = self._present_token()

        ParserUtils.check_is_keyword(tok, type_keywords)
        self.num_of_cond += 1
        var_type = Parser.Node(self.parser_tables.tables[tok.table_index], tok.index, line=tok.line, column=tok.column)
        self.advance()

        var_declaration = Parser.Node((self.parser_tables.get_nodes()), self.parser_tables.parser_nodes.index("VAR"),
                                      line=var_type.line, column=var_type.column)
        self._add_children(var_declaration, var_type)

        # result = self._parse_var_in_declaration(var_type)
        tok = self._present_token()

        # create a new node from the current token
        res_node = Parser.Node(self.parser_tables.tables[tok.table_index], tok.index, line=tok.line, column=tok.column)

        # get the current block level
        curr_block = self.visibility_list[-1]

        # check if the node represents a variable declaration
        if res_node.table[res_node.index_in_table].type is not None:

            # check for double declaration in the current block
            for var in self.parser_tables.vars:
                if res_node.table[res_node.index_in_table].name == var.name and curr_block[0] == var.closure_lvl and \
                        curr_block[1] == var.num_of_closure:
                    raise Parser.DoubleDeclaration(res_node.table[res_node.index_in_table].name, res_node.line,
                                                   res_node.column)

            # add the new variable to the local variable table
            self.parser_tables.vars.append(
                Var(res_node.table[res_node.index_in_table].name, None, curr_block[0], curr_block[1]))
            res_node.index_in_table = len(self.parser_tables.vars) - 1
        self.set_node_type_and_block_info(res_node, var_type, curr_block)

        # set the type and block information for the node

        # move to the next token and return the node
        self.advance()
        result = res_node

        var_declaration.children.append(self._parse_initialization(var_type, result))

        while not self._empty_tokens_list():
            present_token = self._present_token()
            if CheckType.is_operator(present_token, ','):
                self.advance()
            else:
                break

            tok = self._present_token()

            # create a new node from the current token
            res_node = Parser.Node(self.parser_tables.tables[tok.table_index], tok.index, line=tok.line,
                                   column=tok.column)

            # get the current block level
            curr_block = self.visibility_list[-1]

            # check if the node represents a variable declaration
            if res_node.table[res_node.index_in_table].type is not None:

                # check for double declaration in the current block
                for var in self.parser_tables.vars:
                    if res_node.table[res_node.index_in_table].name == var.name and curr_block[0] == var.closure_lvl and \
                            curr_block[1] == var.num_of_closure:
                        raise Parser.DoubleDeclaration(res_node.table[res_node.index_in_table].name, res_node.line,
                                                       res_node.column)

                # add the new variable to the local variable table
                self.parser_tables.vars.append(
                    Var(res_node.table[res_node.index_in_table].name, None, curr_block[0], curr_block[1]))
                res_node.index_in_table = len(self.parser_tables.vars) - 1

            # set the type and block information for the node
            self.set_node_type_and_block_info(res_node, var_type, curr_block)

            # move to the next token and return the node
            self.advance()

            result = res_node

            var_declaration.children.append(self._parse_initialization(var_type, result))

        self._check_is_semicolon()

        return var_declaration

    def set_node_type_and_block_info(self, node, var_type, curr_block):
        node.table[node.index_in_table].type = var_type.table[var_type.index_in_table]
        node.table[node.index_in_table].closure_lvl = curr_block[0]
        self.num_of_cond += 1
        node.table[node.index_in_table].num_of_closure = curr_block[1]

    def _parse_bool(self):
        tok = self._present_token()

        if CheckType.is_operator(tok, '!'):
            negation_node = self._parse_operator()
        else:
            negation_node = None

        tok = self._present_token()

        if CheckType.is_keyword(tok, "to_double"):
            parsed_node = self._parse_to_int_double_bool()
            self.num_of_cond += 1
        elif (tok.table_index == self.parser_tables.tables.index(self.parser_tables.vars)
              and self.parser_tables.tables[tok.table_index][tok.index].type == "bool"):
            parsed_node = self._parse_variable_in_use("bool")
        elif CheckType.is_operator(tok, '('):
            self.advance()
            parsed_node = self._parse_bool_expr()
            self._check_is_close_paren()

            bool_op = parsed_node

            if negation_node is None:
                return bool_op
            self._add_children(negation_node, bool_op)
            return negation_node

        elif CheckType.is_keyword(tok, ("true", "false")):
            tok = self._present_token()
            parsed_node = Parser.Node((self.parser_tables.keywords), tok.index, None, tok.line, tok.column)
            self.advance()
        else:
            parsed_node = self._parse_rel_op()

        bool_op = parsed_node

        if negation_node is None:
            return bool_op
        self._add_children(negation_node, bool_op)
        return negation_node

    def _check_is_open_paren(self):
        ParserUtils.eat_op(self._present_token(), '(')
        self.advance()

    def _check_is_close_paren(self):
        ParserUtils.eat_op(self._present_token(), ')')
        self.advance()

    def _check_is_open_br(self):
        ParserUtils.eat_op(self._present_token(), '{')
        self.advance()

    def _check_is_close_br(self):
        ParserUtils.eat_op(self._present_token(), '}')
        self.advance()

    def _check_is_semicolon(self):
        ParserUtils.eat_op(self._present_token(), ';')
        self.advance()

    def _parse_curly_braces(self):
        self._check_is_open_br()
        self.pr_lvl += 1
        self._update_blocks_on_lvls()

        self.visibility_list.append((self.pr_lvl, self.blocks_on_lvls[self.pr_lvl]))

        res = self._parse_compound_statement()

        self._check_is_close_br()
        self.pr_lvl -= 1
        self.visibility_list.pop()

        return res

    def _parse_print(self):
        tok = self._present_token()

        node = Parser.Node(self.parser_tables.tables[tok.table_index], tok.index, line=tok.line, column=tok.column)

        self.advance()
        self._check_is_open_paren()
        liter = self._parse_string()
        self._check_is_close_paren()
        self._check_is_semicolon()
        self._add_children(node, liter)
        return node

    def _append_new_block_level(self):
        self.blocks_on_lvls.append(0)

    def _increment_block_count(self):
        self.blocks_on_lvls[self.pr_lvl] += 1

    def _update_blocks_on_lvls(self):
        if self.pr_lvl >= len(self.blocks_on_lvls):
            self._append_new_block_level()
        self._increment_block_count()

    def _parse_compound_statement(self):
        tok = self._present_token()
        st_node = Parser.Node(self.parser_tables.get_nodes(),
                              self.parser_tables.get_nodes().index("BODY"),
                              line=tok.line, column=tok.column)

        while not self._empty_tokens_list():
            present_token = self._present_token()
            if CheckType.is_operator(present_token, '}'):
                break
            else:
                st_node.children.append(self.get_next_node())

        return st_node

    def _parse_if(self):
        if_token = self._present_token()
        _if = Parser.Node(self.parser_tables.tables[if_token.table_index], if_token.index, line=if_token.line,
                          column=if_token.column)
        self.advance()

        self._check_is_open_paren()
        self.num_of_cond += 1
        bool_expr = self._parse_bool_expr()
        self._check_is_close_paren()
        then_node = self.get_next_node()
        self._add_children(_if, bool_expr,then_node)


        if self._empty_tokens_list():
            return _if

        # Check for else-if statement
        while CheckType.is_keyword(self._present_token(), "else"):
            self.advance()
            if not CheckType.is_keyword(self._present_token(), "if"):
                else_node = self.get_next_node()
                _if.children.append(else_node)
                break
            else_if_node = self._parse_else_if()
            _if.children.append(else_if_node)

        return _if

    def _parse_cont_or_br(self):
        tok = self._present_token()
        res = Parser.Node(self.parser_tables.tables[tok.table_index], tok.index, line=tok.line, column=tok.column)
        self.advance()
        self._check_is_semicolon()

        return res

    def _parse_else_if(self):
        res = self._create_else_node()
        self.num_of_cond += 1
        self.advance()

        condition_node_elseif = self._parse_condition_node()
        self._check_is_close_paren()

        _else_if = self.get_next_node()
        self._add_children(res, condition_node_elseif, _else_if)
        return res

    def _create_else_node(self):
        return Parser.Node(
            self.parser_tables.tables[self._present_token().table_index],
            self._present_token().index,
            line=self._present_token().line,
            column=self._present_token().column
        )

    def _parse_condition_node(self):
        self._check_is_open_paren()

        condition_node = self._parse_bool_expr()
        return condition_node

    def _parse_while(self):
        tok = self._present_token()
        while_node = Parser.Node(
            self.parser_tables.tables[tok.table_index], tok.index,
            line=tok.line, column=tok.column
        )
        self.advance()

        condition_node = self._parse_while_condition()

        statement = self._parse_while_statement()
        self._add_children(while_node, condition_node,statement)

        return while_node

    def _parse_while_condition(self):
        self.advance()

        condition_node = self._parse_bool_expr()

        self._check_is_close_paren()

        return condition_node

    def _parse_while_statement(self):

        self._is_processing_cycle = True
        statement = self.get_next_node()
        self._is_processing_cycle = False

        return statement

    def get_next_node(self):
        tok = self._present_token()

        if CheckType.is_keyword(tok, "break") and not self._is_processing_cycle:
            raise Parser.ForbiddenStatement(self.parser_tables.tables[tok.table_index][tok.index], tok.line, tok.column)

        switch = {
            "prin": self._parse_print,
            "if": self._parse_if,
            "while": self._parse_while,
            "continue": self._parse_cont_or_br,
            "break": self._parse_cont_or_br,
            "{": self._parse_curly_braces
        }
        try:
            res_node = switch.get(self.parser_tables.tables[tok.table_index][tok.index], None)
        except TypeError:
            res_node = None
        if res_node is not None:
            res_node = res_node()
        else:
            if CheckType.is_variable(tok):
                # parse variable to be assigned
                ident_node = self._parse_variable_in_use()

                # parse assignment operator
                assignment_operator = self._parse_operator()

                # determine the type of the variable
                var_type = ident_node.table[ident_node.index_in_table].type

                # parse the right-hand side of the assignment based on variable type

                if var_type == "bool":
                    right_part = self._parse_bool_expr()
                elif var_type == "int" or var_type == "double":
                    right_part = self._parse_num_expr()
                else:
                    right_part = self._parse_string()

                self._add_children(assignment_operator, ident_node, right_part)

                res_node = assignment_operator
                self._check_is_semicolon()
            else:
                res_node = self._parse_declare_var()

        return res_node

    def parse_program(self):
        pn = self.parser_tables.get_nodes()
        prog_node = Parser.Node(pn, pn.index("PROG"))

        while not self._empty_tokens_list():
            prog_node.children.append(self.get_next_node())

        return prog_node
