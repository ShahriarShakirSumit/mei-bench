"""Copy the tables and figures that come from our original run (see results/original_run/README.md)
into results/tables and results/figures, before the other scripts run."""
import shutil
from repro_paths import RESULTS, TABLES, FIGURES
src = RESULTS / "original_run"
for sub, dst in (("tables", TABLES), ("figures", FIGURES)):
    for f in sorted((src / sub).iterdir()):
        shutil.copyfile(f, dst / f.name)
print("copied", sum(1 for s in ("tables", "figures") for _ in (src / s).iterdir()), "files from the original run")
