class Var:
    def __init__(self, name, _type, close_lvl, num_of_close, txt = None):
        self.name = name
        self.type = _type
        self.closure_lvl = close_lvl
        self.txt = txt
        self.num_of_closure = num_of_close

    def __str__(self):
        return f"Symbol: {self.name}, Type: {self.type}, Closure Level: {self.closure_lvl}"

    def __eq__(self, other):
        if not isinstance(other, Var):
            return False
        return self.name == other.name and \
               self.closure_lvl == other.closure_lvl and \
               self.num_of_closure == other.num_of_closure

    def __repr__(self):
        return f"Variable(name='{self.name}', type='{self.type}', closure" \
               f"={self.closure_lvl}, " \
               f"closure_num={self.num_of_closure})"