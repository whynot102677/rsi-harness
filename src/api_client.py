"""DeepSeek Chat Completions with persistent request and cost limits."""

import fcntl
import hashlib
import json
import os
import time
from dataclasses import dataclass

from dotenv import dotenv_values
from openai import APIConnectionError, APIError, APIStatusError, OpenAI

MAX_P0_ATTEMPTS = 8
PHASE_LIMITS = {"p0": MAX_P0_ATTEMPTS, "p1": 30, "p2": 60}
FORMAL_PHASES = {"p25": 450, "p3": 450}
MAX_TOTAL_ATTEMPTS = 98
MAX_COST_MICROYUAN = 20_000_000
# DeepSeek Flash peak rates, with every input token charged as a cache miss.
INPUT_MICROYUAN = 2
OUTPUT_MICROYUAN = 8


class APIProblem(Exception):
    pass


class BudgetStop(APIProblem):
    pass


def prompt_hash(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()


class AttemptLog:
    def __init__(self, path, limit, *, phase=None):
        if not 1 <= limit <= (PHASE_LIMITS | FORMAL_PHASES)[phase or "p0"]:
            raise ValueError("Invalid phase API budget")
        self.path = path
        self.limit = limit
        self.phase = phase

    def _append(self, event):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            if event["event"] == "attempt":
                stream.seek(0)
                used = sum(json.loads(line)["event"] == "attempt" for line in stream)
                if used >= self.limit:
                    raise BudgetStop(f"{(self.phase or 'p0').upper()} API budget exhausted ({self.limit} attempts)")
                event["attempt"] = used + 1
            stream.write(json.dumps(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return event["attempt"]

    def reserve(self, purpose, task_id, *, prompt="", max_tokens=0):
        # UTF-8 length plus framing allowance bounds the estimated input size.
        reservation = (len(prompt.encode()) + 128) * INPUT_MICROYUAN + max_tokens * OUTPUT_MICROYUAN
        event = {
            "event": "attempt", "purpose": purpose, "task_id": task_id,
            "prompt_sha256": prompt_hash(prompt), "reserved_cost_microyuan": reservation,
        }
        if self.phase is None:
            return self._append(event)
        root = self.path.parent.parent
        root.mkdir(parents=True, exist_ok=True)
        with (root / "api_budget.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            formal = self.phase in FORMAL_PHASES
            usage = phase_usage(root, formal=formal)
            if sum(item["attempts"] for item in usage.values()) >= (450 if formal else MAX_TOTAL_ATTEMPTS):
                raise BudgetStop("Total API request budget exhausted")
            if sum(item["cost_guard_microyuan"] for item in usage.values()) + reservation >= MAX_COST_MICROYUAN:
                raise BudgetStop("Next API request would reach the 20 CNY cost limit")
            return self._append(event)

    def outcome(self, attempt, **fields):
        self._append({"event": "outcome", "attempt": attempt, **fields})

    def events(self):
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_SH)
            return [json.loads(line) for line in stream]

    def usage(self):
        events = self.events()
        measured = [
            event for event in events
            if event.get("input_tokens") is not None and event.get("output_tokens") is not None
        ]
        return {
            "attempts": sum(event["event"] == "attempt" for event in events),
            "attempts_with_usage": len(measured),
            "input_tokens": sum(event["input_tokens"] for event in measured),
            "output_tokens": sum(event["output_tokens"] for event in measured),
        }

    def replay(self, purpose, task_id, prompt):
        events = self.events()
        attempts = {
            event["attempt"] for event in events
            if event["event"] == "attempt" and event["purpose"] == purpose
            and event["task_id"] == task_id and event.get("prompt_sha256") == prompt_hash(prompt)
        }
        for event in reversed(events):
            if event["event"] == "outcome" and event["attempt"] in attempts and event["status"] == "ok":
                if "response_text" in event:
                    return Completion(
                        event["attempt"], event["response_text"], event["model"],
                        event["input_tokens"], event["output_tokens"], event["api_time"],
                    )
        return None


def phase_usage(root, *, formal=False):
    report = {}
    for phase, limit in (FORMAL_PHASES if formal else PHASE_LIMITS).items():
        events = AttemptLog(root / phase / "api_attempts.jsonl", limit, phase=phase).events()
        attempts = [event for event in events if event["event"] == "attempt"]
        measured = {
            event["attempt"]: event for event in events
            if event.get("input_tokens") is not None and event.get("output_tokens") is not None
        }
        inputs = sum(event["input_tokens"] for event in measured.values())
        outputs = sum(event["output_tokens"] for event in measured.values())
        estimate = inputs * INPUT_MICROYUAN + outputs * OUTPUT_MICROYUAN
        breakdown = {}
        for attempt in attempts:
            purpose = attempt["purpose"]
            if purpose.endswith("/proposal"):
                group = "rule_proposal"
            elif phase == "p2" or (phase == "p25" and "/validation/" in purpose):
                group = "candidate_validation"
            elif phase in ("p1", "p3"):
                group = "task_execution"
            elif phase == "p25":
                group = "experience_execution"
            else:
                group = purpose
            usage = breakdown.setdefault(group, {"attempts": 0, "input_tokens": 0, "output_tokens": 0})
            usage["attempts"] += 1
            outcome = measured.get(attempt["attempt"])
            if outcome is not None:
                usage["input_tokens"] += outcome["input_tokens"]
                usage["output_tokens"] += outcome["output_tokens"]
        for usage in breakdown.values():
            usage["estimated_peak_cny"] = (
                usage["input_tokens"] * INPUT_MICROYUAN + usage["output_tokens"] * OUTPUT_MICROYUAN
            ) / 1_000_000
        report[phase] = {
            "attempts": len(attempts), "attempts_with_usage": len(measured),
            "input_tokens": inputs, "output_tokens": outputs,
            "estimated_peak_cny": estimate / 1_000_000,
            "cost_guard_microyuan": estimate + sum(
                event.get("reserved_cost_microyuan", 0) for event in attempts if event["attempt"] not in measured
            ),
            "by_activity": breakdown,
        }
    return report


@dataclass
class Completion:
    attempt: int
    text: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    api_time: float


class DeepSeek:
    def __init__(self, config, log, *, env_file=None, expected_model=None):
        self.key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not self.key and env_file is not None and env_file.is_file():
            values = dotenv_values(env_file, interpolate=False)
            self.key = (values.get("DEEPSEEK_API_KEY") or "").strip()
        if not self.key:
            raise APIProblem("Set DEEPSEEK_API_KEY in the running shell or private project .env")
        if config["base_url"] != "https://api.deepseek.com":
            raise APIProblem("P0 only sends credentials to https://api.deepseek.com")
        self.config = config
        self.log = log
        self.expected_model = expected_model
        self.client = OpenAI(
            api_key=self.key,
            base_url=config["base_url"],
            timeout=config["request_timeout"],
            max_retries=0,
        )

    def complete(self, prompt, *, purpose, task_id=None, max_attempts=None):
        retries = self.config["retry_limit"]
        if max_attempts is not None:
            retries = min(retries, max_attempts - 1)
        for retry in range(retries + 1):
            attempt = self.log.reserve(purpose, task_id, prompt=prompt, max_tokens=self.config["max_tokens"])
            started = time.monotonic()
            try:
                response = self.client.chat.completions.with_raw_response.create(
                    model=self.config["model"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.config["max_tokens"],
                    extra_body={"thinking": {"type": "disabled"}},
                )
                completion = response.parse()
            except (APIError, ValueError) as error:
                http_status = error.status_code if isinstance(error, APIStatusError) else None
                kind = type(error).__name__
                self.log.outcome(
                    attempt, status="error", http_status=http_status, error_type=kind,
                    api_time=round(time.monotonic() - started, 3),
                )
                transient = isinstance(error, APIConnectionError) or (
                    http_status is not None and (http_status in (408, 429) or http_status >= 500)
                )
                if transient and retry < retries:
                    time.sleep(1)
                    continue
                detail = f"HTTP {http_status}" if http_status else kind
                raise APIProblem(f"DeepSeek request failed ({detail})") from None

            elapsed = round(time.monotonic() - started, 3)
            usage = completion.usage
            input_tokens = usage.prompt_tokens if usage else None
            output_tokens = usage.completion_tokens if usage else None
            message = completion.choices[0].message.content if completion.choices else None
            model = (completion.model or "").replace(self.key, "[REDACTED]")
            valid = bool(message and message.strip() and model and usage)
            model_changed = self.expected_model is not None and model != self.expected_model
            sanitized = message.replace(self.key, "[REDACTED]") if message else None
            response_fields = {"response_text": sanitized} if self.log.phase in ("p1", "p2", "p25", "p3") else {}
            self.log.outcome(
                attempt, status="ok" if valid and not model_changed else "invalid_response",
                http_status=response.status_code,
                model=model, input_tokens=input_tokens, output_tokens=output_tokens, api_time=elapsed,
                **response_fields,
            )
            if not valid:
                raise APIProblem("DeepSeek response lacked content, model ID or token usage")
            if model_changed:
                raise APIProblem("DeepSeek response model differs from the P0 verified model")
            return Completion(
                attempt, sanitized, model, input_tokens, output_tokens, elapsed,
            )
