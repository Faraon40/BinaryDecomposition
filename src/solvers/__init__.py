"""Integer Linear Programming (ILP) solvers for optimal solutions.

Available solvers:
- CBC solver (open-source, via PuLP)
- Gurobi solver (commercial, faster - optional)

Note: Gurobi requires separate installation:
    pip install gurobipy
    Free academic license: https://www.gurobi.com/academia/
"""

from src.solvers.ilp_solver_cbc import run_ilp as run_ilp_cbc

# Conditionally import Gurobi if available
try:
    from src.solvers.ilp_solver_gurobi import run_ilp as run_ilp_gurobi
    __all__ = ["run_ilp_cbc", "run_ilp_gurobi"]
except ImportError:
    __all__ = ["run_ilp_cbc"]