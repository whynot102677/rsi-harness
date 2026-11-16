"""Run the frozen P1 paired baseline and P2 validation-gated RSI pilot."""

import argparse
import fcntl
import json
import logging
import re
from collections import Counter

import yaml

from .api_client import APIProblem, AttemptLog, DeepSeek, phase_usage
from .dataset_loader import MANIFEST, ROOT, build_manifest, load_tasks
from .harness import ask, evaluate_fixed, generate, repair, summarize, task_prompt
from .sandbox_runner import Sandbox, SandboxProblem
from .smoke_test import atomic_write

RESULTS = ROOT / "results"


def read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, lambda stream: stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n"))


class Traces:
    def __init__(self, path):
        self.path = path
        self.rows = read_json(path, {})

    def get(self, key):
        return self.rows.get(key)

    def put(self, key, value):
        self.rows[key] = value
        write_json(self.path, self.rows)


def require_p0(manifest):
    folder = RESULTS / "p0"
    rows_file = folder / "smoke_results.jsonl"
    rows = [json.loads(line) for line in rows_file.read_text().splitlines()] if rows_file.exists() else []
    events = AttemptLog(folder / "api_attempts.jsonl", 8).events()
    probes = {event["attempt"] for event in events if event["event"] == "attempt" and event["purpose"] == "probe"}
    models = {event["model"] for event in events if event["attempt"] in probes and event.get("status") == "ok"}
    smoke_attempts = {
        event["attempt"]: event["task_id"] for event in events
        if event["event"] == "attempt" and event["purpose"] == "smoke"
    }
    successful_calls = {event["attempt"] for event in events if event.get("status") == "ok"}
    completed = {
        row["task_id"]: row["model"] for row in rows
        if row["model"] in models and row["execution_status"] in ("passed", "failed")
        and row["failure_type"] in (None, "syntax_error", "runtime_error", "assertion_failure", "timeout")
        and row["api_attempt"] in successful_calls
        and smoke_attempts.get(row["api_attempt"]) == row["task_id"]
        and row["execution_time"] is not None
    }
    if set(completed) != set(manifest["smoke_task_ids"]) or len(set(completed.values())) != 1:
        raise APIProblem("P0 needs a successful API probe and all three real Docker smoke evaluations")
    return next(iter(completed.values()))


def check_ids(config, manifest):
    p1 = config["p1_task_ids"]
    batches = config["p2_experience_batches"]
    batch_ids = [task_id for batch in batches for task_id in batch]
    if (
        len(p1) != 10 or len(set(p1)) != 10
        or set(p1) - set(manifest["splits"]["experience"])
        or set(p1) & set(manifest["smoke_task_ids"])
        or len(batches) != 2 or sorted(batch_ids) != sorted(p1)
        or not all(batches) or len(manifest["splits"]["validation"]) != 10
    ):
        raise ValueError("Frozen P1/P2 allocation does not match the P0 manifest")


def freeze(path, config, model_config, manifest, image, response_model):
    snapshot = {
        "design": config, "inference_and_sandbox": model_config, "thinking": "disabled",
        "dataset_sha256": manifest["sha256"], "docker_image_id": image, "response_model": response_model,
        "validation_task_ids": manifest["splits"]["validation"],
    }
    existing = read_json(path)
    if existing is not None and existing != snapshot:
        raise ValueError("Experiment settings differ from the saved run snapshot")
    write_json(path, snapshot)


def run_p1(tasks, config, client, sandbox, folder):
    ids = config["p1_task_ids"]
    h0 = config["h0"]
    traces = Traces(folder / "traces.json")
    direct, fixed = {}, {}
    try:
        for task_id in ids:
            task = tasks[task_id]
            prompt = task_prompt(task, h0)
            first = generate(task, prompt, "p1/shared-initial", client, sandbox, traces)
            direct[task_id] = [first]
            fixed[task_id] = [
                first, *repair(task, prompt, first, "p1/fixed-repair", client, sandbox, traces),
            ]
            print(f"P1 {task_id} A={first['execution_status']} B={fixed[task_id][-1]['execution_status']}", flush=True)
    finally:
        write_json(folder / "trials.json", {"A": direct, "B": fixed})
        write_json(folder / "summary.json", {
            "complete": len(direct) == len(fixed) == len(ids),
            "shared_initial_generation": True,
            "A": summarize(direct, len(ids)), "B": summarize(fixed, len(ids)),
            "api_usage": client.log.usage(),
        })


def diagnose(batch, trials):
    first_failures = Counter()
    repair_outcomes = Counter()
    for task_id in batch:
        steps = trials[str(task_id)]
        first_failures[steps[0]["failure_type"] or "passed"] += 1
        if len(steps) == 2:
            repair_outcomes[steps[-1]["failure_type"] or "passed"] += 1
    return {
        "experience_tasks": len(batch), "first_outcomes": dict(first_failures),
        "repair_outcomes": dict(repair_outcomes),
    }


def proposal_prompt(harness, diagnosis):
    return (
        "You maintain a reusable Python coding Harness. Propose exactly one new general rule "
        "to improve correctness from these aggregate experience outcomes. If evidence is sparse, "
        "suggest a general self-check and say that the evidence is sparse. Return one JSON object "
        'with string fields "rule" and "rationale". The rule must be a single plain English '
        "sentence under 200 characters, containing only letters, spaces, commas, apostrophes, "
        "and a final period. Never include a specific task, answer, example, identifier, or code.\n"
        f"Current Harness: {json.dumps(harness, sort_keys=True)}\n"
        f"Aggregate experience: {json.dumps(diagnosis, sort_keys=True)}"
    )


def parse_proposal(text, current):
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[8:-4].strip()
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"rule", "rationale"}:
        raise ValueError("Candidate response must contain rule and rationale")
    rule, rationale = value["rule"], value["rationale"]
    if not isinstance(rule, str) or not re.fullmatch(r"[A-Za-z][A-Za-z ,']{9,198}\.", rule):
        raise ValueError("Candidate rule failed the plain English format check")
    if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 1000:
        raise ValueError("Candidate rationale is missing or too long")
    if rule.casefold() in {line.casefold() for line in current["rules"] + current["skills"]}:
        raise ValueError("Candidate repeats an existing rule")
    return value


def validate(baseline, candidate, token_limit):
    before = summarize(baseline, 10)
    after = summarize(candidate, 10)
    if len(baseline) != 10 or set(baseline) != set(candidate):
        raise ValueError("Acceptance requires all ten paired validation tasks")
    regressions = [
        task_id for task_id, steps in baseline.items()
        if steps[-1]["execution_status"] == "passed" and candidate[task_id][-1]["execution_status"] != "passed"
    ]
    gains = [
        task_id for task_id, steps in candidate.items()
        if steps[-1]["execution_status"] == "passed" and baseline[task_id][-1]["execution_status"] != "passed"
    ]
    old_tokens = before["input_tokens"] + before["output_tokens"]
    new_tokens = after["input_tokens"] + after["output_tokens"]
    accepted = bool(gains and not regressions and new_tokens * 100 <= old_tokens * (100 + token_limit))
    return {
        "accepted": accepted, "gains": gains, "regressions": regressions,
        "token_increase_percent": round(100 * (new_tokens - old_tokens) / old_tokens, 3),
        "baseline": before, "candidate": after,
        "max_extra_tokens_percent": token_limit,
    }


def run_p2(tasks, ids, config, client, sandbox, folder):
    p1_folder = folder.parent / "p1"
    p1_summary = read_json(p1_folder / "summary.json", {})
    if not p1_summary.get("complete"):
        raise APIProblem("P1 must complete both methods before P2")
    trials = read_json(p1_folder / "trials.json")["B"]
    traces = Traces(folder / "traces.json")
    h0 = config["h0"]
    current = h0
    version = "H0"
    trace_version = "H0"
    versions = folder / "versions"
    write_json(versions / "H0.json", h0)
    if not (folder / "active.json").exists():
        write_json(folder / "active.json", {"version": version, "harness": current})
    validated = 0
    for index, batch in enumerate(config["p2_experience_batches"], 1):
        round_path = folder / f"round_{index}.json"
        previous = read_json(round_path, {})
        if previous.get("state") == "validated":
            version, current = previous["active_version"], previous["active_harness"]
            trace_version = previous["active_trace_version"]
            write_json(folder / "active.json", {"version": version, "harness": current})
            validated += 1
            continue
        if previous.get("state") == "proposal_rejected":
            continue
        worst_case_attempts = 21 if validated else 41
        if index == 2 and not previous and client.log.limit - client.log.usage()["attempts"] < worst_case_attempts:
            print("P2 round 2 skipped to preserve room for a full validation pair", flush=True)
            break
        diagnosis = diagnose(batch, trials)
        round_data = {
            "round": index, "parent_version": version, "experience_task_ids": batch,
            "diagnosis": diagnosis, "state": "proposing",
        }
        if not previous:
            write_json(round_path, round_data)
        proposal = ask(client, proposal_prompt(current, diagnosis), purpose=f"p2/round-{index}/proposal")
        round_data["proposal_attempt"] = proposal.attempt
        try:
            change = parse_proposal(proposal.text, current)
        except ValueError as error:
            round_data.update(state="proposal_rejected", error=str(error), active_version=version)
            write_json(round_path, round_data)
            continue

        candidate_id = f"C{index}"
        candidate = {"rules": current["rules"], "skills": [*current["skills"], change["rule"]]}
        write_json(versions / f"{candidate_id}.json", candidate)
        round_data.update(state="validating", candidate_version=candidate_id, difference=change)
        write_json(round_path, round_data)
        baseline_scores, candidate_scores = {}, {}
        baseline_trace_version = trace_version
        for position, task_id in enumerate(ids):
            # Alternate first-round arm order. A previously scored version reuses its exact trace.
            pair = [(baseline_trace_version, current, baseline_scores), (candidate_id, candidate, candidate_scores)]
            if position % 2:
                pair.reverse()
            for arm, harness, scores in pair:
                scores[task_id] = evaluate_fixed(
                    tasks[task_id], harness, f"p2/validation/{arm}", client, sandbox, traces,
                )
            print(f"P2 round {index} validated task {task_id}", flush=True)
        decision = validate(baseline_scores, candidate_scores, config["max_extra_validation_tokens_percent"])
        validated += 1
        if decision["accepted"]:
            version = f"H{index}"
            current = candidate
            trace_version = candidate_id
            write_json(versions / f"{version}.json", current)
        round_data.update(
            state="validated", decision=decision, active_version=version, active_harness=current,
            active_trace_version=trace_version,
            validation_trial_keys={"baseline": f"p2/validation/{baseline_trace_version}",
                                   "candidate": f"p2/validation/{candidate_id}"},
            ungated_candidate_version=candidate_id,
        )
        write_json(round_path, round_data)
        write_json(folder / "active.json", {"version": version, "harness": current})
        print(f"P2 round {index}: {'accepted' if decision['accepted'] else 'rolled back'} ({version})", flush=True)
    if not validated:
        raise APIProblem("No candidate completed validation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("p1", "p2", "all", "usage"))
    args = parser.parse_args()
    for name in ("openai", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    if args.phase == "usage":
        print(json.dumps(phase_usage(RESULTS), indent=2))
        return
    config = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text())
    model_config = yaml.safe_load((ROOT / "configs/p0.yaml").read_text())
    manifest = read_json(MANIFEST)
    try:
        response_model = require_p0(manifest)
        check_ids(config, manifest)
        full_data = load_tasks()
        if build_manifest(full_data) != manifest:
            raise ValueError("P0 manifest differs from the pinned MBPP split")
        ids = manifest["splits"]["validation"]
        tasks = {task_id: full_data[task_id] for task_id in config["p1_task_ids"] + ids}
        sandbox = Sandbox(model_config["docker_image"], model_config["execution_timeout"])
        image = sandbox.prepare()
        with (RESULTS / "experiments.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            freeze(RESULTS / "experiment_config.json", config, model_config, manifest, image, response_model)
            phases = ("p1", "p2") if args.phase == "all" else (args.phase,)
            try:
                for phase in phases:
                    folder = RESULTS / phase
                    log = AttemptLog(folder / "api_attempts.jsonl", config[f"{phase}_attempt_limit"], phase=phase)
                    client = DeepSeek(model_config, log, env_file=ROOT / ".env", expected_model=response_model)
                    if phase == "p1":
                        run_p1(tasks, config, client, sandbox, folder)
                    else:
                        run_p2(tasks, ids, config, client, sandbox, folder)
            finally:
                write_json(RESULTS / "experiment_usage.json", phase_usage(RESULTS))
    except (APIProblem, SandboxProblem, OSError, ValueError) as error:
        parser.exit(1, f"Experiment stopped: {error}\n")


if __name__ == "__main__":
    main()
