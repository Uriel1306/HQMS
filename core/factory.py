from environment.RDP7 import RDP7_Problem
from environment.SimplePoly import SimplePoly_Problem
from environment.RDCP1 import RDCP1_Problem

class ProblemFactory:
    @staticmethod
    def create_problem(problem_name: str):
        if problem_name.upper() == "RDP7":
            return RDP7_Problem()
        elif problem_name.upper() == "SIMPLEPOLY":
            return SimplePoly_Problem()
        elif problem_name.upper() == "RDCP1":
            return RDCP1_Problem()
        raise ValueError(f"Unknown problem: {problem_name}")