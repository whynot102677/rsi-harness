"""Run the once-frozen formal experiment, with task checkpoints and paired analysis."""

import argparse
import fcntl
import itertools
import json
import logging
import math
import random
from collections import defaultdict

import yaml

from .api_client import APIProblem, AttemptLog, BudgetStop, DeepSeek, INPUT_MICROYUAN, OUTPUT_MICROYUAN, phase_usage
from .dataset_loader import MANIFEST, ROOT, load_selected
from .experiments import RESULTS, Traces, read_json, write_json
from .formal import FROZEN, HASH_FILE, P25, freeze_formal, sha, write_csv
from .harness import MODEL_FAILURES, generate, repair, task_prompt
from .sandbox_runner import Sandbox, SandboxProblem

P3 = RESULTS / "p3"


def verify_frozen():
    if not FROZEN.is_file() or not HASH_FILE.is_file() or sha(FROZEN) != HASH_FILE.read_text().split()[0]:
        raise ValueError("Missing or altered formal protocol")
    config = yaml.safe_load(FROZEN.read_text())
    if config["dataset"]["manifest_sha256"] != sha(MANIFEST):
        raise ValueError("Frozen dataset manifest has changed")
    for filename, digest in config["source_sha256"].items():
        if sha(ROOT / "src" / filename) != digest:
            raise ValueError(f"Frozen experiment code has changed: {filename}")
    return config


def variant_groups(groups):
    canonical = {}
    source = {}
    for label in ("B", "C", "D"):
        signature = json.dumps([groups[label]["harness"], groups[label]["max_repairs"]], sort_keys=True)
        canonical.setdefault(signature, label)
        source[label] = canonical[signature]
    return source


def evaluate_arm(task, label, group, client, sandbox, traces):
    base = f"p3/{label}"
    prompt = task_prompt(task, group["harness"])
    stages = (
        ("initial", lambda: generate(task, prompt, f"{base}/initial", client, sandbox, traces)),
        ("repair", lambda: repair(task, prompt, first, f"{base}/repair", client, sandbox, traces)[0]),
    )
    first = second = None
    for stage, call in stages:
        if stage == "repair" and first["failure_type"] not in MODEL_FAILURES:
            break
        for evaluation_attempt in range(2):
            try:
                step = call()
                break
            except BudgetStop:
                raise
            except APIProblem as error:
                if any(term in str(error) for term in ("HTTP 401", "HTTP 402", "HTTP 403", "differs from the P0")):
                    raise
                return {"initial": first, "repair": second, "incomplete": "api_error",
                        "reason": str(error), "incomplete_stage": stage}
            except SandboxProblem as error:
                if "Docker evaluation stopped" not in str(error):
                    raise
                if evaluation_attempt == 1:
                    return {"initial": first, "repair": second, "incomplete": "infrastructure_error",
                            "reason": str(error), "incomplete_stage": stage}
        if stage == "initial":
            first = step
        else:
            second = step
    return {"initial": first, "repair": second, "incomplete": None, "reason": None, "incomplete_stage": None}


def request_totals(events):
    outcomes = [event for event in events if event["event"] == "outcome"]
    inputs = sum(event.get("input_tokens") or 0 for event in outcomes)
    outputs = sum(event.get("output_tokens") or 0 for event in outcomes)
    return {
        "api_requests": sum(event["event"] == "attempt" for event in events),
        "input_tokens": inputs,
        "output_tokens": outputs,
        "api_seconds": round(sum(event.get("api_time", 0) for event in outcomes), 3),
        "estimated_cny": (inputs * INPUT_MICROYUAN + outputs * OUTPUT_MICROYUAN) / 1_000_000,
    }


def result_rows(ids, arms, groups, sources, events):
    outcomes = {event["attempt"]: event for event in events if event["event"] == "outcome"}
    calls = defaultdict(list)
    for event in events:
        if event["event"] != "attempt":
            continue
        _, group, stage = event["purpose"].split("/")
        calls[group, stage, event["task_id"]].append(event)
        if event["attempt"] in outcomes:
            calls[group, stage, event["task_id"]].append(outcomes[event["attempt"]])
    rows = []
    for task_id in ids:
        for label, group in groups.items():
            source = "B" if label == "A" else sources[label]
            trial = arms.get(f"{source}:{task_id}")
            if not trial:
                continue
            first = trial["initial"]
            second = None if label == "A" else trial["repair"]
            final = second or first
            incomplete = trial["incomplete"] if label != "A" or not first else None
            first_calls = calls[source, "initial", task_id]
            repair_calls = calls[source, "repair", task_id] if label != "A" else []
            totals = request_totals(first_calls + repair_calls)
            docker_time = sum(step["execution_time"] for step in (first, second) if step)
            rows.append({
                "task_id": task_id, "group": label, "method": group["method"],
                "harness_version": group["version"], "score_source_group": source,
                "reuse": ("initial_only" if label == "A" else "full") if source != label else "",
                "completion_status": "incomplete" if incomplete else "scored",
                "incomplete_reason": trial["reason"] if incomplete else None,
                "first_status": first["execution_status"] if first else None,
                "first_pass": first["execution_status"] == "passed" if first else None,
                "first_failure_type": first["failure_type"] if first else None,
                "first_assertions_passed": first["tests_passed"] if first else None,
                "first_attempt_ref": first["attempt"] if first else None,
                "repair_requested": bool(repair_calls),
                "repair_pass": second["execution_status"] == "passed" if second else None,
                "repair_attempt_ref": second["attempt"] if second else None,
                "final_pass": final["execution_status"] == "passed" if final and not incomplete else None,
                "final_assertions_passed": final["tests_passed"] if final and not incomplete else None,
                "assertions_total": final["tests_total"] if final else None,
                "failure_type": incomplete or (final["failure_type"] if final else None),
                "model": final["model"] if final else None,
                **totals, "docker_seconds": round(docker_time, 3),
                "elapsed_api_plus_docker_seconds": round(totals["api_seconds"] + docker_time, 3),
            })
    return rows


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def method_rows(rows, groups, ids, optimization):
    table = []
    for label, group in groups.items():
        arm = [row for row in rows if row["group"] == label]
        complete = [row for row in arm if row["completion_status"] == "scored"]
        first = [row for row in arm if row["first_pass"] is not None]
        repairs = [row for row in arm if row["repair_pass"] is not None]
        inputs = sum(row["input_tokens"] for row in arm)
        outputs = sum(row["output_tokens"] for row in arm)
        cost = sum(row["estimated_cny"] for row in arm)
        optimized = optimization.get(label, {"tokens": 0, "cny": 0})
        denominator = len(ids)
        table.append({
            "group": label, "method": group["method"], "harness_version": group["version"],
            "planned": denominator, "scored": len(complete), "incomplete": denominator - len(complete),
            "api_incomplete": sum(row["failure_type"] == "api_error" for row in arm),
            "infrastructure_incomplete": sum(row["failure_type"] == "infrastructure_error" for row in arm),
            "first_scored": len(first), "first_successes": sum(row["first_pass"] for row in first),
            "first_success_rate": ratio(sum(row["first_pass"] for row in first), len(first)),
            "final_successes": sum(row["final_pass"] for row in complete),
            "final_success_rate": ratio(sum(row["final_pass"] for row in complete), len(complete)),
            "repairs_scored": len(repairs), "successful_repairs": sum(row["repair_pass"] for row in repairs),
            "repair_success_rate": ratio(sum(row["repair_pass"] for row in repairs), len(repairs)),
            "task_input_tokens": inputs, "task_output_tokens": outputs,
            "task_tokens_per_planned_task": (inputs + outputs) / denominator,
            "attributed_task_api_calls": sum(row["api_requests"] for row in arm),
            "attributed_task_cny": round(cost, 6),
            "optimization_tokens": optimized["tokens"], "optimization_cny": optimized["cny"],
            "optimization_tokens_per_planned_task": optimized["tokens"] / denominator,
            "all_in_tokens_per_planned_task": (inputs + outputs + optimized["tokens"]) / denominator,
            "all_in_cny_per_planned_task": (cost + optimized["cny"]) / denominator,
        })
    return table


def quantile(values, probability):
    rank = (len(values) - 1) * probability
    lower = math.floor(rank)
    upper = math.ceil(rank)
    return values[lower] + (values[upper] - values[lower]) * (rank - lower)


def paired_rows(rows, bootstrap):
    by_group = defaultdict(dict)
    for row in rows:
        if row["final_pass"] is not None:
            by_group[row["group"]][row["task_id"]] = int(row["final_pass"])
    output = []
    for left, right in itertools.combinations("ABCD", 2):
        a, b = by_group[left], by_group[right]
        ids = sorted(a.keys() & b.keys())
        deltas = [b[i] - a[i] for i in ids]
        rng = random.Random(bootstrap["seed"])
        alpha = (1 - bootstrap["confidence"]) / 2
        distribution = sorted(
            sum(rng.choices(deltas, k=len(ids))) * 100 / len(ids)
            for _ in range(bootstrap["resamples"])
        ) if ids else []
        output.append({
            "left": left, "right": right, "paired_n": len(ids),
            "left_successes": sum(a[i] for i in ids), "right_successes": sum(b[i] for i in ids),
            "both_successes": sum(a[i] and b[i] for i in ids),
            "left_only": deltas.count(-1), "right_only": deltas.count(1),
            "right_minus_left_pp": 100 * sum(deltas) / len(ids) if ids else None,
            "ci95_low_pp": quantile(distribution, alpha) if distribution else None,
            "ci95_high_pp": quantile(distribution, 1 - alpha) if distribution else None,
        })
    return output


def iteration_rows(config):
    pilot = [read_json(RESULTS / "p2" / f"round_{i}.json") for i in (1, 2)]
    decisions = read_json(P25 / "decisions.json")
    records = []
    baseline = pilot[0]["decision"]["baseline"]
    entries = [("P2", "H0", "baseline", None, None)]
    for item in pilot:
        entries.append(("P2", item.get("candidate_version", f"proposal {item['round']}"), item["state"],
                        item.get("decision"), item.get("error")))
    for item in decisions:
        entries.append(("P2.5", item["version"], item["state"], item.get("decision"),
                        item.get("reason") or ", ".join(item.get("reasons", []))))
    for phase, version, state, decision, note in entries:
        metrics = decision["candidate"] if decision else baseline if state == "baseline" else None
        if decision and not note and not decision["accepted"]:
            note = ", ".join(
                label for label, failed in (
                    ("no_new_pass", not decision["gains"]), ("regression", bool(decision["regressions"])),
                    ("extra_tokens_above_limit", decision["token_increase_percent"] > 25),
                ) if failed
            )
        records.append({
            "phase": phase, "version": version, "state": state,
            "validation_passes": metrics["final_passes"] if metrics else None,
            "validation_n": metrics["completed"] if metrics else None,
            "gains": len(decision["gains"]) if decision else None,
            "regressions": len(decision["regressions"]) if decision else None,
            "total_tokens": metrics["input_tokens"] + metrics["output_tokens"] if metrics else None,
            "token_change_percent": decision["token_increase_percent"] if decision else None,
            "gate_accepted": decision["accepted"] if decision else None,
            "frozen_C": version == config["groups"]["C"]["version"],
            "frozen_D": version == config["groups"]["D"]["version"],
            "decision_reason": note,
        })
    return records


def export_results(config, checkpoint, log):
    groups = config["groups"]
    ids = config["dataset"]["splits"]["test"]
    sources = variant_groups(groups)
    formal = phase_usage(RESULTS, formal=True)
    optimization = formal["p25"]["by_activity"]
    shared = [optimization[key] for key in ("experience_execution", "rule_proposal") if key in optimization]
    verified = [*shared, *([optimization["candidate_validation"]] if "candidate_validation" in optimization else [])]
    assigned = {label: {"tokens": sum(x["input_tokens"] + x["output_tokens"] for x in buckets),
                        "cny": round(sum(x["estimated_peak_cny"] for x in buckets), 6)}
                for label, buckets in (("C", shared), ("D", verified))}
    rows = result_rows(ids, checkpoint.rows, groups, sources, log.events())
    if rows:
        write_csv(P3 / "task_level_results.csv", rows)
    methods = method_rows(rows, groups, ids, assigned)
    write_csv(P3 / "method_comparison.csv", methods)
    pairs = paired_rows(rows, config["protocol"]["bootstrap"])
    write_csv(P3 / "paired_comparisons.csv", pairs)
    selected = [("cross_task_update", "B", "C"), ("validation_gate", "C", "D"),
                ("verified_vs_fixed", "B", "D")]
    write_csv(P3 / "ablation_results.csv", [
        {"mechanism": name, **next(pair for pair in pairs if pair["left"] == left and pair["right"] == right),
         "equivalent_configuration": sources[left] == sources[right]}
        for name, left, right in selected
    ])
    write_csv(P3 / "iteration_history.csv", iteration_rows(config))
    write_json(P3 / "usage_summary.json", {
        "frozen_sha256": sha(FROZEN), "actual_usage": formal,
        "historical_pilot_usage": phase_usage(RESULTS),
        "new_actual_total": {field: sum(stage[field] for stage in formal.values()) for field in (
            "attempts", "attempts_with_usage", "input_tokens", "output_tokens", "estimated_peak_cny")},
        "optimization_attributed_by_method": assigned, "score_source": {"A": "B", **sources},
        "expected_task_method_rows": len(ids) * len(groups), "saved_task_method_rows": len(rows),
        "missing_or_incomplete": sum(method["incomplete"] for method in methods),
    })


def run():
    config = verify_frozen()
    inference = config["inference"]
    sandbox = Sandbox(inference["docker_image"], inference["execution_timeout"])
    if sandbox.prepare() != config["docker_image_id"]:
        raise SandboxProblem("Docker image differs from frozen image")
    ids = config["dataset"]["splits"]["test"]
    if len(ids) != 40 or len(set(ids)) != 40:
        raise ValueError("Expected exactly 40 distinct formal test IDs")
    tasks = load_selected(ids)
    log = AttemptLog(P3 / "api_attempts.jsonl", 450, phase="p3")
    client = DeepSeek(inference, log, env_file=ROOT / ".env", expected_model=config["response_model"])
    traces = Traces(P3 / "traces.json")
    checkpoint = Traces(P3 / "arm_checkpoints.json")
    groups = config["groups"]
    sources = list(dict.fromkeys(variant_groups(groups).values()))
    try:
        for index, task_id in enumerate(ids):
            ordered = sources[index % len(sources):] + sources[:index % len(sources)]
            for label in ordered:
                key = f"{label}:{task_id}"
                if checkpoint.get(key) is None:
                    result = evaluate_arm(tasks[task_id], label, groups[label], client, sandbox, traces)
                    checkpoint.put(key, result)
            print(f"P3 task {task_id} ({index + 1}/{len(ids)}) checkpointed", flush=True)
    finally:
        export_results(config, checkpoint, log)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    args = parser.parse_args()
    for name in ("openai", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    try:
        with (RESULTS / "experiments.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if args.stage == "run":
                run()
            elif FROZEN.exists() or HASH_FILE.exists() or (P3 / "api_attempts.jsonl").exists():
                raise ValueError("Formal protocol is already frozen or has results")
            else:
                manifest = read_json(MANIFEST)
                pilot = read_json(RESULTS / "experiment_config.json")
                prepared = read_json(P25 / "freeze_inputs.json")
                if prepared is None:
                    raise ValueError("P2.5 must finish before the freeze")
                freeze_formal(manifest, pilot["inference_and_sandbox"], prepared["image"], prepared["model"],
                              pilot["design"]["h0"], prepared["decisions"])
                print(sha(FROZEN), flush=True)
    except (APIProblem, SandboxProblem, OSError, ValueError) as error:
        parser.exit(1, f"Formal experiment stopped: {error}\n")


if __name__ == "__main__":
    main()
