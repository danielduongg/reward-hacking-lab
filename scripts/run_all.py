"""Run the full pipeline: overoptimization curve -> training dynamics -> mitigations."""
import runpy
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
for step in ["01_overoptimization_curve.py", "02_training_dynamics.py", "03_mitigations.py"]:
    print(f"\n=== {step} ===")
    sys.argv = [step]
    runpy.run_path(str(HERE / step), run_name="__main__")
