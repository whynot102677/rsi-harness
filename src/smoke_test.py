"""Prepare P0 results, probe DeepSeek, or run the three fixed MBPP smoke tasks."""

import argparse
import ast
import csv
import fcntl
import json
import logging
import os
import re
import tempfile
from pathlib import Path

import yaml

from .api_client import APIProblem, AttemptLog, BudgetStop, DeepSeek
from .dataset_loader import MANIFEST, ROOT, build_manifest, load_tasks
from .sandbox_runner import Sandbox, SandboxProblem

RESULTS = ROOT / "results/p0"
FIELDS = (
    "task_id", "requested_model", "model", "execution_status", "tests_passed",
    "tests_total", "pass_rate", "failure_type", "execution_time", "input_tokens",
    "output_tokens", "api_time", "api_attempt", "docker_image_id",
)


def atomic_write(path, write):
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, prefix=".p0-", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            write(stream)
            stream.flush()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def save_results(rows, output=RESULTS):
    output.mkdir(parents=True, exist_ok=True)

    def write_jsonl(stream):
        for row in rows:
            stream.write(json.dumps(row) + "\n")

    def write_csv(stream):
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    atomic_write(output / "smoke_results.jsonl", write_jsonl)
    atomic_write(output / "smoke_summary.csv", write_csv)


def make_row(task, requested_model, status, *, failure=None):
    return dict.fromkeys(FIELDS) | {
        "task_id": task["task_id"],
        "requested_model": requested_model,
        "execution_status": status,
        "tests_total": len(task["test_list"]),
        "failure_type": failure,
    }


def prompt_for(task):
    # In the frozen experiment tasks, any helpers precede the exported function.
    names = [node.name for node in ast.parse(task["code"]).body if isinstance(node, ast.FunctionDef)]
    name = names[-1]
    return (
        f"{task['text']}\nDefine a top-level function named {name}. "
        "Return only Python source code, with any standard-library imports required."
    )


def extract_source(text):
    text = text.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:python|py)?[ \t]*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
        if not match:
            raise ValueError("Unrecognized code fence")
        text = match.group(1).strip()
    if not text:
        raise ValueError("Empty candidate")
    return text


def run_smoke(tasks, config, log, *, env_file=None):
    rows = []
    sandbox = Sandbox(config["docker_image"], config["execution_timeout"])
    image_id = None
    try:
        image_id = sandbox.prepare()
        client = DeepSeek(config, log, env_file=env_file)
    except (SandboxProblem, APIProblem) as error:
        failure = "sandbox_initialization_failure" if isinstance(error, SandboxProblem) else "api_configuration_failure"
        previous = RESULTS / "smoke_results.jsonl"
        prior_rows = [json.loads(line) for line in previous.read_text().splitlines()] if previous.exists() else []
        preflight_failures = {"sandbox_initialization_failure", "api_configuration_failure"}
        if all(
            row["execution_status"] == "blocked"
            and row["failure_type"] in preflight_failures
            and row["api_attempt"] is None
            for row in prior_rows
        ):
            for task in tasks:
                row = make_row(task, config["model"], "blocked", failure=failure)
                row["docker_image_id"] = image_id
                rows.append(row)
            save_results(rows, RESULTS)
        raise

    generated = RESULTS / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        row = make_row(task, config["model"], "failed")
        row["docker_image_id"] = image_id
        try:
            completion = client.complete(prompt_for(task), purpose="smoke", task_id=task["task_id"])
        except BudgetStop:
            raise
        except APIProblem as error:
            row["failure_type"] = "api_failure"
            print(f"Task {task['task_id']} API failure ({error})")
        else:
            for field in ("model", "input_tokens", "output_tokens", "api_time"):
                row[field] = getattr(completion, field)
            row["api_attempt"] = completion.attempt
            try:
                source = extract_source(completion.text)
            except ValueError:
                row["failure_type"] = "code_extraction_failure"
            else:
                # A distinct file for each candidate preserves retry evidence.
                path = generated / f"task-{task['task_id']}-attempt-{completion.attempt}.py"
                atomic_write(path, lambda stream: stream.write(source + "\n"))
                try:
                    evaluation = sandbox.evaluate(source, task["test_setup_code"], task["test_list"])
                except SandboxProblem:
                    row.update(execution_status="blocked", failure_type="sandbox_cleanup_failure")
                    save_results(rows + [row])
                    raise
                row.update(vars(evaluation))
                if evaluation.tests_passed is not None:
                    row["pass_rate"] = evaluation.tests_passed / row["tests_total"]
        rows.append(row)
        save_results(rows)
        print(f"Task {row['task_id']} {row['execution_status']} ({row['failure_type'] or 'all checks passed'})")

    pipeline_failures = {
        "api_failure", "code_extraction_failure", "sandbox_initialization_failure",
        "sandbox_protocol_error",
    }
    return int(any(row["failure_type"] in pipeline_failures for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("probe", "sandbox", "run", "usage"))
    args = parser.parse_args()
    for logger in ("openai", "httpx", "httpcore"):
        logging.getLogger(logger).setLevel(logging.CRITICAL)
    config = yaml.safe_load((ROOT / "configs/p0.yaml").read_text(encoding="utf-8"))
    log = AttemptLog(RESULTS / "api_attempts.jsonl", config["api_attempt_limit"], phase="p0")
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if args.command == "usage":
                print(json.dumps(log.usage()))
                return 0
            if args.command == "probe":
                completion = DeepSeek(config, log, env_file=ROOT / ".env").complete(
                    "Reply with the word OK.", purpose="probe"
                )
                print(json.dumps({k: v for k, v in vars(completion).items() if k != "text"}))
                return 0
            if args.command == "sandbox":
                print(Sandbox(config["docker_image"], config["execution_timeout"]).prepare())
                return 0
            tasks = load_tasks()
            manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
            if manifest != build_manifest(tasks):
                raise ValueError("Split manifest differs from the pinned P0 split")
            return run_smoke(
                [tasks[task_id] for task_id in manifest["smoke_task_ids"]],
                config, log, env_file=ROOT / ".env",
            )
        except (APIProblem, SandboxProblem, OSError, ValueError) as error:
            parser.exit(1, f"P0 stopped: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
