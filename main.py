import sys

from lexer import Lexer, OPERATORS, RESERVED_KEYWORDS, CONSTANTS, LexicalError
from parser import Parser, ParserUtils
from sem_analyzer import SemAnalyzer
from symbol_table import SymbolTable
from exec import Translator


def main():
    with open('input.txt') as f:
        text = f.read()

        # Lexical Analysis
    lexer = Lexer(text)
    symbolTable = SymbolTable(lexer, OPERATORS, RESERVED_KEYWORDS, CONSTANTS)
    try:
        symbolTable.fulfill()
    except LexicalError as err:
        print(f"LEXER ERROR: {err.message} line {err.line} column {err.column}")
        sys.exit(1)
    print(symbolTable)

    # Syntax Analysis
    parser = Parser(
        symbolTable.general_table,
        symbolTable.tables)
    try:
        syntax_tree = parser.parse_program()
    except Parser.SyntaxError as err:
        print(f"PARSER ERROR:\n{err}")
        sys.exit(1)
    print("parser result:")
    ParserUtils.print_tree(syntax_tree)

    # Semantic Analysis
    semantic_analyzer = SemAnalyzer(
        parser.parser_tables.get_operands(),
        parser.parser_tables.get_vars(),
        parser.parser_tables.get_keywords(),
        parser.parser_tables.get_consts()
    )
    try:
        semantic_analyzer.passing(syntax_tree)
    except SemAnalyzer.SemError as err:
        print(f"SEMANTIC ERROR:\n{err}")
        sys.exit(1)

    interpreter = Translator(
        parser.parser_tables.get_nodes(),
        parser.parser_tables.get_operands(),
        parser.parser_tables.get_vars(),
        parser.parser_tables.get_keywords(),
        parser.parser_tables.get_consts()    )
    try:
        interpreter.exec(syntax_tree)
    except Translator.ExecutionError as err:
        print(f"\n\n\nRUNTIME ERROR:\n{err}")
        exit(1)


if __name__ == '__main__':
    main()
