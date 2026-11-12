import os
import subprocess
import tarfile

from conftest import REPO


def test_pack_results_contents_exclude_weights(spaced_tmp):
    runs, results, out = spaced_tmp / "runs", spaced_tmp / "results", spaced_tmp / "out"
    out.mkdir()
    for name in ("d11n_s0", "w11n_s0"):
        (runs / name / "weights").mkdir(parents=True)
        for f in ("args.yaml", "env.json", "env_resume_1.json", "results.csv", "confusion_matrix.png",
                  "train_batch0.jpg", "weights/best.pt"):
            (runs / name / f).write_text("x")
        (runs / f"{name}.log").write_text("log")
    (results / "d11n_s0").mkdir(parents=True)
    (results / "d11n_s0" / "metrics.json").write_text("{}")
    (results / "compare.json").write_text("{}")
    env = dict(os.environ, RUNS_DIR=str(runs), RESULTS_DIR=str(results), OUT_DIR=str(out))
    subprocess.run(["bash", str(REPO / "scripts" / "pack_results.sh")], env=env, check=True, capture_output=True)
    (archive,) = out.glob("results_*_*.tar.gz")
    names = set(tarfile.open(archive).getnames())
    assert not any(n.endswith(".pt") for n in names)
    assert not any(n.endswith("train_batch0.jpg") for n in names)  # only the listed run files
    for run in ("d11n_s0", "w11n_s0"):
        for f in ("args.yaml", "env.json", "env_resume_1.json", "results.csv", "confusion_matrix.png"):
            assert any(n.endswith(f"{run}/{f}") for n in names)
        assert any(n.endswith(f"{run}.log") for n in names)
    assert any(n.endswith("results/compare.json") for n in names)
    assert any(n.endswith("d11n_s0/metrics.json") for n in names)
    assert not list(out.glob(".pack_list.*"))
