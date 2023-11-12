import random
import copy

from typing import Set, Dict, List, TypeVar, Optional
from abc import ABC, abstractmethod

from util import monitor


Value = TypeVar("Value")


class Variable(ABC):
    @property
    @abstractmethod
    def startDomain(self) -> Set[Value]:
        """Returns the set of initial values of this variable (not taking constraints into account)."""
        pass

    def __copy__(self):
        return self

    def __deepcopy__(self, memodict={}):
        return self


class CSP(ABC):
    def __init__(self, MRV=True, LCV=True):
        self.MRV = MRV
        self.LCV = LCV

    @property
    @abstractmethod
    def variables(self) -> Set[Variable]:
        """Return the set of variables in this CSP.
        Abstract method to be implemented for specific instances of CSP problems.
        """
        pass

    def remainingVariables(self, assignment: Dict[Variable, Value]) -> Set[Variable]:
        """Returns the variables not yet assigned."""
        return self.variables.difference(assignment.keys())

    @abstractmethod
    def neighbors(self, var: Variable) -> Set[Variable]:
        """Return all variables related to var by some constraint.
        Abstract method to be implemented for specific instances of CSP problems.
        """
        pass

    def assignmentToStr(self, assignment: Dict[Variable, Value]) -> str:
        """Formats the assignment of variables for this CSP into a string."""
        s = ""
        for var, val in assignment.items():
            s += f"{var} = {val}\n"
        return s

    def isComplete(self, assignment: Dict[Variable, Value]) -> bool:
        """Return whether the assignment covers all variables.
        :param assignment: dict (Variable -> value)
        """
        return len(assignment) == len(self.variables)

    @abstractmethod
    def isValidPairwise(
        self, var1: Variable, val1: Value, var2: Variable, val2: Value
    ) -> bool:
        """Return whether this pairwise assignment is valid with the constraints of the csp.
        Abstract method to be implemented for specific instances of CSP problems.
        """
        pass

    def isValid(self, assignment: Dict[Variable, Value]) -> bool:
        """Return whether the assignment is valid (i.e. is not in conflict with any constraints).
        You only need to take binary constraints into account.
        Hint: use `CSP::neighbors` and `CSP::isValidPairwise` to check that all binary constraints are satisfied.
        Note that constraints are symmetrical, so you don't need to check them in both directions.
        """
        for var in assignment:
            for neighbor in self.neighbors(var):
                if neighbor not in assignment:
                    continue
                if not self.isValidPairwise(
                    var, assignment[var], neighbor, assignment[neighbor]
                ):
                    return False
        return True

    def solveBruteForce(
        self, initialAssignment: Dict[Variable, Value] = dict()
    ) -> Optional[Dict[Variable, Value]]:
        """Called to solve this CSP with brute force technique.
        Initializes the domains and calls `CSP::_solveBruteForce`."""
        domains = domainsFromAssignment(initialAssignment, self.variables)
        return self._solveBruteForce(initialAssignment, domains)

    @monitor
    def _solveBruteForce(
        self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]]
    ) -> Optional[Dict[Variable, Value]]:
        """Implement the actual backtracking algorithm to brute force this CSP.
        Use `CSP::isComplete`, `CSP::isValid`, `CSP::selectVariable` and `CSP::orderDomain`.
        :return: a complete and valid assignment if one exists, None otherwise.
        """
        if self.isComplete(assignment):
            return assignment
        var = self.selectVariable(assignment, domains)
        for value in self.orderDomain(assignment, domains, var):
            assignment[var] = value
            if self.isValid(assignment):
                result = self._solveBruteForce(assignment, domains)
                if result is not None:
                    return result
            del assignment[var]
        return None

    def solveForwardChecking(
        self, initialAssignment: Dict[Variable, Value] = dict()
    ) -> Optional[Dict[Variable, Value]]:
        """Called to solve this CSP with forward checking.
        Initializes the domains and calls `CSP::_solveForwardChecking`."""
        domains = domainsFromAssignment(initialAssignment, self.variables)
        for var in set(initialAssignment.keys()):
            domains = self.forwardChecking(initialAssignment, domains, var)
        return self._solveForwardChecking(initialAssignment, domains)

    @monitor
    def _solveForwardChecking(
        self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]]
    ) -> Optional[Dict[Variable, Value]]:
        """Implement the actual backtracking algorithm with forward checking.
        Use `CSP::forwardChecking` and you should no longer need to check if an assignment is valid.
        :return: a complete and valid assignment if one exists, None otherwise.
        """
        if self.isComplete(assignment):
            return assignment
        var = self.selectVariable(assignment, domains)
        for value in self.orderDomain(assignment, domains, var):
            if self.isValid(assignment):
                assignment[var] = value
                new_domains = self.forwardChecking(assignment, domains, var)
                if new_domains is not None:
                    result = self._solveForwardChecking(assignment, new_domains)
                    if result is not None:
                        return result
                del assignment[var]
        return None

    def forwardChecking(
        self,
        assignment: Dict[Variable, Value],
        domains: Dict[Variable, Set[Value]],
        variable: Variable,
    ) -> Dict[Variable, Set[Value]] or None:
        """Implement the forward checking algorithm from the theory lectures.

        :param domains: current domains.
        :param assignment: current assignment.
        :param variable: The variable that was just assigned (only need to check changes).
        :return: the new domains after enforcing all constraints.
        """
        new_domains = copy.copy(domains)
        for neighbor in self.neighbors(variable):
            if neighbor in assignment:
                continue
            new_set = set()
            for value in new_domains[neighbor]:
                if self.isValidPairwise(
                    variable, assignment[variable], neighbor, value
                ):
                    new_set.add(value)
            new_domains[neighbor] = new_set
            if len(new_set) == 0:
                return None
        return new_domains

    def selectVariable(
        self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]]
    ) -> Variable:
        """Implement a strategy to select the next variable to assign."""
        if not self.MRV:
            return random.choice(list(self.remainingVariables(assignment)))

        min_var = None
        min_len = float("inf")
        for var in self.remainingVariables(assignment):
            if len(domains[var]) < min_len:
                min_var = var
                min_len = len(domains[var])
        return min_var

    def orderDomain(
        self,
        assignment: Dict[Variable, Value],
        domains: Dict[Variable, Set[Value]],
        var: Variable,
    ) -> List[Value]:
        """Implement a smart ordering of the domain values."""
        if not self.LCV:
            return list(domains[var])

        # TODO: Rewrite to make this more efficient
        def count_prunes(value):
            count = 0
            for neighbor in self.neighbors(var):
                if neighbor in assignment:
                    continue
                for neighbor_value in domains[neighbor]:
                    if not self.isValidPairwise(var, value, neighbor, neighbor_value):
                        count += 1
            return count

        return sorted(domains[var], key=lambda value: count_prunes(value))

    def solveAC3(
        self, initialAssignment: Dict[Variable, Value] = dict()
    ) -> Optional[Dict[Variable, Value]]:
        """Called to solve this CSP with AC3.
        Initializes domains and calls `CSP::_solveAC3`."""
        domains = domainsFromAssignment(initialAssignment, self.variables)
        for var in set(initialAssignment.keys()):
            domains = self.ac3(initialAssignment, domains, var)
        return self._solveAC3(initialAssignment, domains)

    @monitor
    def _solveAC3(
        self, assignment: Dict[Variable, Value], domains: Dict[Variable, Set[Value]]
    ) -> Optional[Dict[Variable, Value]]:
        """
        Implement the actual backtracking algorithm with AC3.
        Use `CSP::ac3`.
        :return: a complete and valid assignment if one exists, None otherwise.
        """
        if self.isComplete(assignment):
            return assignment
        var = self.selectVariable(assignment, domains)
        original_domain = copy.deepcopy(domains)

        for value in self.orderDomain(assignment, domains, var):
            assignment[var] = value
            domains[var] = {value}
            new_domains = self.ac3(assignment, domains, var)
            if new_domains is not None:
                result = self._solveAC3(assignment, new_domains)
                if result is not None:
                    return result
            del assignment[var]
            domains = original_domain


        return None

    def remove_inconsistent_values(self, domains, tail, head):
        removed = False
        to_remove = set()
        for x in domains[tail]:
            if not any(self.isValidPairwise(tail, x, head, y) for y in domains[head]):
                to_remove.add(x)
                removed = True
        for x in to_remove:
            domains[tail].remove(x)
        return removed

    def ac3(
        self,
        assignment: Dict[Variable, Value],
        domains: Dict[Variable, Set[Value]],
        variable: Variable,
    ) -> Dict[Variable, Set[Value]] or None:
        """Implement the AC3 algorithm from the theory lectures.

        :param domains: current domains.
        :param assignment: current assignment.
        :param variable: The variable that was just assigned (only need to check changes).
        :return: the new domains ensuring arc consistency.
        """

        queue = [(variable, neighbor) for neighbor in self.neighbors(variable) if neighbor != variable]
        while queue:
            (head, tail) = queue.pop(0)
            if self.remove_inconsistent_values(domains, tail, head):
                if len(domains[tail]) == 0:
                    return None
                for neighbor in self.neighbors(tail):
                    queue.append((neighbor, tail))
        return domains


def domainsFromAssignment(
    assignment: Dict[Variable, Value], variables: Set[Variable]
) -> Dict[Variable, Set[Value]]:
    """Fills in the initial domains for each variable.
    Already assigned variables only contain the given value in their domain.
    """
    domains = {v: v.startDomain for v in variables}
    for var, val in assignment.items():
        domains[var] = {val}
    return domains
