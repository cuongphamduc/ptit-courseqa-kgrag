"""Orchestrates Stages 5-6 on the single GPU, strictly sequentially (no concurrent GPU jobs, so the latencies logged by
06_run.py are clean). Every step is resumable; completed outputs are skipped by the underlying scripts.

  build     LightRAG graphs (05c) after the S5 extraction (05b) has finished
  tune      dev tuning 05d: k -> s5grid -> s5facts -> s4mode
  freeze    freeze the dev-chosen configuration (after the dev results were checked)
  test      frozen config: retrieval S1-S5, then generation S0-S5 x 5 SLMs, PTIT test and ViQuAD test
  ablate    S5 ablations with the primary SLM on both test sets
  eval      06_eval + 06_stats
usage: 06_pipeline.py build tune | freeze test ablate eval
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import data, llm, systems  # noqa: E402

PY = sys.executable
S = os.path.join(data.WORK, "scripts")
RES = os.path.join(data.ROOT, "results")


def sh(*args):
    print(time.strftime("%H:%M:%S"), " ".join(args), flush=True)
    subprocess.run([PY, *args], check=True)


def n_lines(p):
    return sum(1 for _ in open(p)) if os.path.exists(p) else 0


def build():
    while subprocess.run(["pgrep", "-u", os.environ.get("USER", "admin"), "-f", "05b_build_kg_extract"], capture_output=True).stdout.strip():
        time.sleep(120)
    sh(os.path.join(S, "05c_build_lightrag.py"), "ptit", "viquad_test")


def tune():
    for step in ("k", "s5grid", "s5facts", "s4mode"):
        if os.path.exists(os.path.join(RES, "dev", f"tune_{step}.json")):
            continue
        sh(os.path.join(S, "05d_tune_dev.py"), step)


def freeze():
    sh(os.path.join(S, "05d_tune_dev.py"), "freeze")


def run_test(system_list, slms):
    cfg = json.load(open(os.path.join(RES, "configs", "systems_config.json")))
    assert cfg.get("frozen"), "config not frozen"
    for ds in ("ptit", "viquad"):
        nq = len(data.questions(ds, "test"))
        for s in system_list:
            if s != "S0" and n_lines(os.path.join(RES, "retrieval", f"{ds}_test", f"{s}.jsonl")) < nq:
                sh(os.path.join(S, "06_run.py"), "retrieve", "--system", s, "--dataset", ds, "--split", "test")
    for slm in slms:
        for ds in ("ptit", "viquad"):
            nq = len(data.questions(ds, "test"))
            for s in system_list:
                if n_lines(os.path.join(RES, "predictions", f"{ds}_test", f"{s}__{slm}.jsonl")) < nq:
                    sh(os.path.join(S, "06_run.py"), "generate", "--system", s, "--slm", slm, "--dataset", ds, "--split", "test")


def test():
    run_test(systems.SYSTEMS, list(llm.SLMS))


def ablate():
    cfg = json.load(open(os.path.join(RES, "configs", "systems_config.json")))
    abl = [a for a in systems.ABLATIONS if a != f"S5-depth{cfg['S5']['depth']}"]  # the chosen depth is S5 itself
    run_test(abl, [llm.PRIMARY])


def evaluate():
    sh(os.path.join(S, "06_eval.py"), "ptit_test", "viquad_test")
    sh(os.path.join(S, "06_stats.py"), "ptit_test", "viquad_test")


if __name__ == "__main__":
    for step in sys.argv[1:]:
        {"build": build, "tune": tune, "freeze": freeze, "test": test, "ablate": ablate, "eval": evaluate}[step]()
