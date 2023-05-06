INT, DOUBL, STRIN = 0, 1, 2


class Constant:

    def __init__(self, value, tp):
        self.value = value
        self.type = tp

    def __str__(self):
        if self.type == INT:
            return str(self.value)
        elif self.type == DOUBL:
            return str(round(self.value, 2))
        elif self.type == STRIN:
            return f"'{self.value}'"
        else:
            raise ValueError(f"Unknown constant type: {self.type}")

    def __eq__(self, other):
        if isinstance(other, Constant):
            return self.value == other.value and self.type == other.type
        return False

    def __repr__(self):
        return f"Constant({self.value}, {self.type})"