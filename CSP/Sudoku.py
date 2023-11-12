from typing import Set, Dict

from CSP import CSP, Variable, Value


class Sudoku(CSP):
    def __init__(self, MRV=True, LCV=True):
        super().__init__(MRV=MRV, LCV=LCV)
        self._variables = set(Cell(n) for n in range(81))

    @property
    def variables(self) -> Set["Cell"]:
        """Return the set of variables in this CSP."""
        return self._variables

    def getCell(self, x: int, y: int) -> "Cell":
        """Get the  variable corresponding to the cell on (x, y)"""
        for var in self.variables:
            if var.row == x and var.col == y:
                return var

    def neighbors(self, var: "Cell") -> Set["Cell"]:
        """Return all variables related to var by some constraint."""
        return self.variables - {var}

    def isValidPairwise(
        self, var1: "Cell", val1: Value, var2: "Cell", val2: Value
    ) -> bool:
        """Return whether this pairwise assignment is valid with the constraints of the csp."""
        if var1 == var2:
            return False

        if (var1.row == var2.row or var1.col == var2.col) and val1 == val2:
            return False

        if var1.row // 3 == var2.row // 3 and var1.col // 3 == var2.col // 3 and val1 == val2:
            return False

        return True

    def assignmentToStr(self, assignment: Dict["Cell", Value]) -> str:
        """Formats the assignment of variables for this CSP into a string."""
        s = ""
        for y in range(9):
            if y != 0 and y % 3 == 0:
                s += "---+---+---\n"
            for x in range(9):
                if x != 0 and x % 3 == 0:
                    s += "|"

                cell = self.getCell(x, y)
                s += str(assignment.get(cell, " "))
            s += "\n"
        return s

    def parseAssignment(self, path: str) -> Dict["Cell", Value]:
        """Gives an initial assignment for a Sudoku board from file."""
        initialAssignment = dict()

        with open(path, "r") as file:
            for y, line in enumerate(file.readlines()):
                if line.isspace():
                    continue
                assert y < 9, "Too many rows in sudoku"

                for x, char in enumerate(line):
                    if char.isspace():
                        continue

                    assert x < 9, "Too many columns in sudoku"

                    var = self.getCell(x, y)
                    val = int(char)

                    if val == 0:
                        continue

                    assert val > 0 and val < 10, f"Impossible value in grid"
                    initialAssignment[var] = val
        return initialAssignment


class Cell(Variable):
    def __init__(self, n):
        super().__init__()
        self.row = n % 9
        self.col = n // 9
        self._n = n

    @property
    def startDomain(self) -> Set[Value]:
        """Returns the set of initial values of this variable (not taking constraints into account)."""
        return set(range(1, 10))

    def __repr__(self):
        return f"({self.row}, {self.col})"
    def __eq__(self, other):
        return self._n == other._n

    def __hash__(self):
        return hash(self._n)
