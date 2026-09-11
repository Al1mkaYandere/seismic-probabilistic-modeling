"""The run's threshold, panel file and output directory come from the environment.

Why a subprocess in every test: several modules bind their own constants from
``config`` AT IMPORT TIME (``spatial_diagnostics.PROCESSED_DATA``,
``probabilistic_evaluation.OUTPUT_*``, ``tail_metrics.PREDICTION_STORE``).
Assigning to ``config`` from inside a running test would move some of the
pipeline and leave the rest pointing at the published files - which is exactly
the failure this parametrisation exists to prevent. So each test starts a fresh
interpreter with the environment already set, the way a real parallel run does,
and checks what the code then actually READS and WRITES - not what the module
constants say.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PUBLISHED_OUTPUTS = REPO_ROOT / "outputs"
PUBLISHED_PANEL = REPO_ROOT / "data" / "processed" / "spatiotemporal_grid.csv"


SWEEP_SOURCE = """
import importlib, json, pkgutil, sys
sys.path.insert(0, '.')
from pathlib import Path

import src
from src import config
swept, skipped, paths = [], [], []
for info in pkgutil.iter_modules(src.__path__):
    try:
        module = importlib.import_module('src.' + info.name)
    except ImportError:
        skipped.append(info.name)
        continue
    swept.append(info.name)
    for attr, value in vars(module).items():
        if isinstance(value, Path):
            paths.append((info.name + '.' + attr, str(value)))

print(json.dumps({'swept': swept, 'skipped': skipped, 'paths': paths,
                  'out': str(config.OUTPUT_DIR)}))
"""

def _run(code: str, env_overrides: dict[str, str] | None = None) -> dict:
    """Run ``code`` in a fresh interpreter; it prints one JSON object."""
    env = dict(os.environ)
    for key in ("SPM_M_C", "SPM_PANEL_FILE", "SPM_OUTPUT_DIR"):
        env.pop(key, None)
    env.update(env_overrides or {})
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def published_fingerprint() -> dict[str, str]:
    """What a published file looks like: content, last write, size.

    The last-write time is not decoration. A step that writes over a published
    file with the SAME bytes - which happens whenever its inputs were not
    perturbed - is invisible to a content comparison, and a real review found
    exactly that: a second write inside ``run_mc_estimation`` landed on the
    published copy and every test stayed green. The nanosecond mtime catches
    the write regardless of what was written.
    """
    out: dict[str, str] = {}
    for path in sorted(PUBLISHED_OUTPUTS.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        out[str(path.relative_to(PUBLISHED_OUTPUTS))] = (
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}:{stat.st_mtime_ns}:{stat.st_size}"
        )
    return out


def test_unset_environment_reproduces_the_published_layout() -> None:
    """With nothing set, the run must land exactly where the published one did."""
    out = _run(
        "import json, sys; sys.path.insert(0, '.');"
        "from src import config;"
        "print(json.dumps({'m_c': config.M_C, 'panel': config.PROCESSED_DATA_FILE,"
        " 'out': str(config.OUTPUT_DIR), 'fig': str(config.FIGURES_DIR),"
        " 'comparison': str(config.MODEL_COMPARISON_CSV)}))"
    )
    assert out["m_c"] == 3.0
    assert out["panel"] == "spatiotemporal_grid.csv"
    assert Path(out["out"]) == PUBLISHED_OUTPUTS
    assert Path(out["fig"]) == PUBLISHED_OUTPUTS / "figures"
    assert Path(out["comparison"]) == PUBLISHED_OUTPUTS / "model_comparison.csv"


def test_threshold_override_actually_cuts_the_catalogue(tmp_path: Path) -> None:
    """SPM_M_C must reach the one place that cuts the catalogue, and ETAS with it."""
    code = (
        "import json, sys; sys.path.insert(0, '.');"
        "import pandas as pd;"
        "from src import config;"
        "from src.grid_builder import filter_to_completeness, load_raw_data;"
        "from src.etas_baseline import _m_c;"
        "raw = load_raw_data();"
        "print(json.dumps({'m_c': config.M_C, 'etas_m_c': _m_c(),"
        " 'total': len(raw), 'kept': len(filter_to_completeness(raw))}))"
    )
    at_3 = _run(code)
    at_45 = _run(code, {"SPM_M_C": "4.5"})

    assert at_3["m_c"] == 3.0 and at_3["etas_m_c"] == 3.0
    assert at_3["kept"] == at_3["total"], "at 3.0 nothing may be dropped"

    assert at_45["m_c"] == 4.5 and at_45["etas_m_c"] == 4.5, "ETAS must see the same threshold"
    assert at_45["kept"] < at_3["kept"], "raising the threshold must drop events"
    assert at_45["kept"] == int((pd.read_csv(REPO_ROOT / "data/raw/usgs_central_asia_raw.csv")["mag"] >= 4.5).sum())


def test_output_override_writes_elsewhere_and_leaves_published_files_alone(tmp_path: Path) -> None:
    """A real pipeline step must write into the override, not into outputs/."""
    published = PUBLISHED_OUTPUTS / "mc_estimate.csv"
    before = published.read_bytes() if published.exists() else None

    out = _run(
        "import json, sys; sys.path.insert(0, '.');"
        "from src import config;"
        "from src.mc_estimation import run_mc_estimation;"
        "run_mc_estimation();"
        "print(json.dumps({'out': str(config.OUTPUT_DIR)}))",
        {"SPM_OUTPUT_DIR": str(tmp_path / "elsewhere")},
    )

    written = Path(out["out"]) / "mc_estimate.csv"
    assert written.exists(), "the step wrote nowhere the override points to"
    assert Path(out["out"]) == tmp_path / "elsewhere"
    after = published.read_bytes() if published.exists() else None
    assert after == before, "the published copy was touched by an overridden run"


def test_panel_override_changes_which_panel_is_loaded(tmp_path: Path) -> None:
    """SPM_PANEL_FILE must reach the loader, and the published panel stays put."""
    fake = pd.read_csv(PUBLISHED_PANEL, nrows=40).copy()
    fake["Y"] = 777                      # a value the real panel cannot contain
    panel_dir = tmp_path / "processed"
    panel_dir.mkdir()
    (panel_dir / "panel_under_test.csv").write_text(fake.to_csv(index=False))

    code = (
        "import json, sys; sys.path.insert(0, '.');"
        "from src import config;"
        f"config.PROCESSED_DATA_PATH = __import__('pathlib').Path({str(panel_dir)!r});"
        "from src.dl_modeling import _load_df;"
        "df = _load_df();"
        "print(json.dumps({'file': config.PROCESSED_DATA_FILE, 'rows': len(df),"
        " 'y_values': sorted(set(int(v) for v in df['Y']))}))"
    )
    out = _run(code, {"SPM_PANEL_FILE": "panel_under_test.csv"})

    assert out["file"] == "panel_under_test.csv"
    assert out["y_values"] == [777], "the loader read some other panel"
    assert PUBLISHED_PANEL.exists()


def test_empty_variable_means_the_default_not_a_crash() -> None:
    """``SPM_M_C=`` in a shell exports an EMPTY string, not an absent variable.

    Treating that as a value would make ``float("")`` blow up at import and take
    the whole run with it, so an empty override must read as "not set".
    """
    out = _run(
        "import json, sys; sys.path.insert(0, '.');"
        "from src import config;"
        "print(json.dumps({'m_c': config.M_C, 'panel': config.PROCESSED_DATA_FILE,"
        " 'out': str(config.OUTPUT_DIR)}))",
        {"SPM_M_C": "", "SPM_PANEL_FILE": "", "SPM_OUTPUT_DIR": ""},
    )
    assert out["m_c"] == 3.0
    assert out["panel"] == "spatiotemporal_grid.csv"
    assert Path(out["out"]) == PUBLISHED_OUTPUTS


def test_no_module_holds_a_path_into_the_published_tree(tmp_path: Path) -> None:
    """The general form of the defect: ANY module may not escape the override.

    The three tests above walk three concrete code paths. This one imports every
    module under ``src/`` in an overridden run and inspects every module-level
    ``Path`` it holds - the constants that get bound AT IMPORT TIME and are
    therefore the easy place for a hard-coded path to hide. None of them may
    point inside the published outputs directory or at the published panel.
    """
    code = SWEEP_SOURCE
    override = tmp_path / "run-elsewhere"
    out = _run(code, {"SPM_OUTPUT_DIR": str(override),
                      "SPM_PANEL_FILE": "panel_under_test.csv",
                      "SPM_M_C": "4.5"})

    # A sweep that silently imports nothing would pass no matter what, so the
    # modules that actually write files have to be among the ones it looked at.
    swept = set(out["swept"])
    assert {"tail_metrics", "spatial_diagnostics", "probabilistic_evaluation",
            "dl_modeling", "validation", "modeling", "calibration", "mc_estimation",
            "grid_builder", "poisson_analysis", "visualizer"} <= swept, f"not swept: {swept}"
    # data_ingestion is the one module that cannot be imported here: it pulls in
    # ``requests``, deliberately absent because the pipeline runs offline.
    assert set(out["skipped"]) <= {"data_ingestion"}, f"unexpectedly unimportable: {out['skipped']}"
    published = str(PUBLISHED_OUTPUTS.resolve())
    offenders = [
        (name, value) for name, value in out["paths"]
        if value == published or value.startswith(published + "/") or value == str(PUBLISHED_PANEL)
    ]
    assert not offenders, f"paths escaping the override: {offenders}"


def test_directory_bootstrap_follows_the_override(tmp_path: Path) -> None:
    """``ensure_project_directories`` must create the RUN's directories.

    Kept in its own interpreter that imports nothing but ``config`` and
    ``utils``: ``validation.py`` creates the figures directory as an import
    side effect, so a test that sweeps every module would find the directory
    already there and pass no matter what the bootstrap did.
    """
    override = tmp_path / "bootstrap-here"
    out = _run(
        "import json, sys; sys.path.insert(0, '.');"
        "from src import config;"
        "from src.utils import ensure_project_directories;"
        "before = config.FIGURES_DIR.is_dir();"
        "ensure_project_directories();"
        "print(json.dumps({'before': before, 'after': config.FIGURES_DIR.is_dir(),"
        " 'fig': str(config.FIGURES_DIR)}))",
        {"SPM_OUTPUT_DIR": str(override)},
    )
    assert out["before"] is False, "the test cannot tell whether the bootstrap did anything"
    assert out["after"] is True, "the bootstrap did not create the configured figures directory"
    assert Path(out["fig"]) == override / "figures"
    assert (override / "figures").is_dir()


def test_an_overridden_run_never_writes_into_the_published_outputs(tmp_path: Path) -> None:
    """The end-to-end form: RUN the writing steps, then prove nothing published moved.

    The module sweep above only sees paths bound at module level. A path built
    inside a function escapes it - and ``run_calibration``, ``run_tail_metrics``
    and ``run_probabilistic_evaluation`` are called by no other test at all, so
    such a path would be invisible to the whole suite.

    Two details decide whether this test has teeth, and the first version of it
    had neither:

    * only the steps' INPUTS are placed in the override. Copying everything
      would put the produced files there in advance, and "the file appeared"
      would then be true no matter where the step actually wrote.
    * those inputs are PERTURBED. With unperturbed inputs a step that writes
      into the published directory writes the very same bytes that are already
      there, and a byte comparison sees nothing. Perturbed inputs make any such
      write show up as a changed published file.
    """
    fingerprint = published_fingerprint

    before = fingerprint()
    assert len(before) > 10, "the published outputs are missing - this test would prove nothing"

    override = tmp_path / "run-here"
    (override / "figures").mkdir(parents=True)
    inputs = ("test_predictions.csv", "calibration_predictions.csv",
              "model_comparison.csv", "etas_test_predictions.csv")
    for name in inputs:
        shutil.copy2(PUBLISHED_OUTPUTS / name, override / name)

    perturbed = pd.read_csv(override / "calibration_predictions.csv")
    perturbed["mu_pred"] = perturbed["mu_pred"] * 1.5      # so any output differs
    perturbed.to_csv(override / "calibration_predictions.csv", index=False)

    produced = ("mc_estimate.csv", "calibration_summary.csv", "tail_evaluation.csv",
                "moran_residuals.csv", "exceedance_calibration.csv",
                "probabilistic_scores.csv", "score_comparisons.csv")
    for name in produced:
        assert not (override / name).exists(), "a produced file was pre-placed - the check would be empty"

    out = _run(
        "import json, sys; sys.path.insert(0, '.');"
        "from src import config;"
        "from src.calibration import run_calibration;"
        "from src.tail_metrics import run_tail_metrics;"
        "from src.spatial_diagnostics import run_spatial_diagnostics;"
        "from src.probabilistic_evaluation import run_probabilistic_evaluation;"
        "from src.mc_estimation import run_mc_estimation;"
        "ran = [];"
        "[ (f(), ran.append(f.__name__)) for f in ("
        "  run_mc_estimation, run_calibration, run_tail_metrics,"
        "  run_spatial_diagnostics, run_probabilistic_evaluation) ];"
        "print(json.dumps({'ran': ran, 'out': str(config.OUTPUT_DIR)}))",
        {"SPM_OUTPUT_DIR": str(override)},
    )

    assert len(out["ran"]) == 5, f"not every step ran: {out['ran']}"
    missing = [name for name in produced if not (override / name).exists()]
    assert not missing, f"written somewhere other than the override: {missing}"

    after = fingerprint()
    changed = sorted(k for k in before if after.get(k) != before[k])
    assert not changed, f"an overridden run rewrote published files: {changed}"
    assert set(after) == set(before), f"published file list changed: {set(after) ^ set(before)}"


def test_the_fitting_steps_also_write_only_into_the_override(tmp_path: Path) -> None:
    """Same guarantee for the two steps that fit the classical models.

    Kept apart from the test above for one concrete reason: ``run_modeling``
    REWRITES ``model_comparison.csv`` from scratch, and the pipeline appends the
    deep-learning and ETAS rows to that file afterwards. Running it in the same
    directory as the scoring steps would hand them a truncated table. Here it
    gets an empty directory of its own, which also gives the "the file appeared"
    check its teeth: nothing is pre-placed.
    """
    fingerprint = published_fingerprint

    before = fingerprint()
    override = tmp_path / "fitting-here"

    out = _run(
        "import json, sys; sys.path.insert(0, '.');"
        "import pandas as pd;"
        "from src import config;"
        "from src.poisson_analysis import run_poisson_analysis;"
        "from src.modeling import run_modeling;"
        "panel = pd.read_csv(config.PROCESSED_DATA_PATH / config.PROCESSED_DATA_FILE);"
        "run_poisson_analysis(panel);"
        "run_modeling();"
        "print(json.dumps({'out': str(config.OUTPUT_DIR)}))",
        {"SPM_OUTPUT_DIR": str(override)},
    )

    assert Path(out["out"]) == override
    for produced in ("model_comparison.csv", "test_predictions.csv", "poisson_cell_stats.csv"):
        assert (override / produced).exists(), f"{produced} was not written into the override"

    after = fingerprint()
    changed = sorted(k for k in before if after.get(k) != before[k])
    assert not changed, f"a fitting step rewrote published files: {changed}"
    assert set(after) == set(before), f"published file list changed: {set(after) ^ set(before)}"


def test_no_module_spells_the_output_directory_by_hand() -> None:
    """A spelling guard for the steps too slow to run inside the suite.

    What it is: the two tests above prove BEHAVIOURALLY that seven pipeline
    steps write only where they are told. Five more - ``run_dl_modeling``,
    ``run_walk_forward``, ``run_alpha_identifiability`` and the two plotting
    helpers - train neural networks and take about a minute and a half, far too
    long to run on every test invocation, so no behavioural test covers them.

    What it is NOT: proof that those steps write in the right place. It only
    catches the accident that produced every escape found so far - a path built
    by spelling the directory name out in the code. Deliberate obfuscation walks
    past it, and so does a path assembled from pieces. The real check for those
    steps is the whole-pipeline run with the override, done by hand and recorded.

    Prose is exempt on purpose: several modules DISCUSS the output directory in
    their docstrings, and forbidding that would push people into writing worse
    documentation to please a test.
    """
    def code_strings(source: str) -> list[tuple[int, str]]:
        """Every string literal in the file except docstrings."""
        tree = ast.parse(source)
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                body = getattr(node, "body", [])
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                        and isinstance(body[0].value.value, str):
                    docstrings.add(id(body[0].value))
        return [
            (n.lineno, n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings
        ]

    offenders: dict[str, list[str]] = {}
    for path in sorted((REPO_ROOT / "src").glob("*.py")):
        if path.name == "config.py":
            continue
        hits = [f"line {line}: {value!r}" for line, value in code_strings(path.read_text())
                if "outputs" in value]
        if hits:
            offenders[path.name] = hits

    assert not offenders, (
        "a module spells the output directory instead of taking it from config: "
        f"{offenders}"
    )


def test_saving_the_panel_follows_the_configured_name_and_place(tmp_path: Path) -> None:
    """``save_processed_data`` is the only function that writes the panel.

    No test called it at all, and it is the one write that could destroy the
    published panel: a run at another completeness threshold saves its own panel
    through this function, so a hard-coded filename here would overwrite the
    panel the published results were built from.
    """
    published_stat = PUBLISHED_PANEL.stat()
    published_hash = hashlib.sha256(PUBLISHED_PANEL.read_bytes()).hexdigest()

    panel_dir = tmp_path / "processed"
    panel_dir.mkdir()

    out = _run(
        "import json, sys; sys.path.insert(0, '.');"
        "import pandas as pd;"
        "from pathlib import Path;"
        "from src import config;"
        f"config.PROCESSED_DATA_PATH = Path({str(panel_dir)!r});"
        "from src.grid_builder import save_processed_data;"
        "df = pd.read_csv(Path(config.PROJECT_ROOT) / 'data' / 'processed' /"
        " 'spatiotemporal_grid.csv', nrows=25);"
        "save_processed_data(df);"
        "print(json.dumps({'name': config.PROCESSED_DATA_FILE,"
        " 'written': sorted(p.name for p in Path(config.PROCESSED_DATA_PATH).iterdir())}))",
        {"SPM_PANEL_FILE": "panel_for_this_run.csv"},
    )

    assert out["name"] == "panel_for_this_run.csv"
    assert out["written"] == ["panel_for_this_run.csv"], (
        f"the panel was saved somewhere else: {out['written']}"
    )
    assert (panel_dir / "panel_for_this_run.csv").exists()
    assert len(pd.read_csv(panel_dir / "panel_for_this_run.csv")) == 25

    assert hashlib.sha256(PUBLISHED_PANEL.read_bytes()).hexdigest() == published_hash
    assert PUBLISHED_PANEL.stat().st_mtime_ns == published_stat.st_mtime_ns, (
        "the published panel was written over"
    )
