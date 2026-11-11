"""LLM clients: the real Anthropic client and a scripted fake for testing."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None  # None when no prices are configured
    wait_seconds: float = 0.0  # time spent waiting out usage limits (not counted as model time)
    model_id: str | None = None  # the exact model that answered, when known


class UsageLimitError(RuntimeError):
    """Raised when the subscription usage limit is hit and waiting did not help."""


class ClaudeCodeClient:
    """Uses the Claude Code CLI (`claude -p`) with your normal Claude subscription login.

    No API key and no extra payment: calls count against your plan's usage limits.
    - model: an alias ("haiku", "sonnet", "opus") or a full model ID
    - All tools are switched off and customizations (CLAUDE.md, memory, plugins, MCP) are not
      loaded (--safe-mode), so the model only sees our system prompt and message.
    - Cost reported is Claude Code's estimate of what the same tokens would cost on the API
      ("API-equivalent cost"). Nothing is billed.
    - If a usage limit is hit, it waits and retries (wait_minutes, up to max_wait_hours).
    """

    def __init__(self, model: str, claude_path: str | None = None, effort: str | None = None,
                 timeout_s: int = 1800, wait_minutes: float = 15, max_wait_hours: float = 8):
        import shutil
        self.model = model
        self.exe = claude_path or shutil.which("claude")
        if not self.exe:
            raise RuntimeError(
                "Could not find the `claude` command. Install Claude Code and log in once with `claude`."
            )
        self.effort = effort
        self.timeout = timeout_s
        self.wait_s = wait_minutes * 60
        self.max_wait_s = max_wait_hours * 3600

    def _call(self, system: str, user: str) -> dict:
        import json
        import subprocess
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:  # empty folder: nothing for it to read
            sp = Path(tmp) / "system_prompt.txt"
            sp.write_text(system, encoding="utf-8")
            cmd = [
                self.exe, "-p",
                "--model", self.model,
                "--output-format", "json",
                "--system-prompt-file", str(sp),
                "--disallowedTools", "*",
                "--strict-mcp-config",
                "--safe-mode",
                "--no-session-persistence",
            ]
            if self.effort:
                cmd += ["--effort", self.effort]
            proc = subprocess.run(
                cmd, input=user, capture_output=True, text=True, encoding="utf-8",
                cwd=tmp, timeout=self.timeout,
            )
        out = (proc.stdout or "").strip()
        try:
            data = json.loads(out.splitlines()[-1] if out else "{}")
        except json.JSONDecodeError:
            data = {}
        if not data:
            data = {"is_error": True, "result": out + "\n" + (proc.stderr or "")}
        data["_returncode"] = proc.returncode
        return data

    @staticmethod
    def _is_limit(data: dict) -> bool:
        text = str(data.get("result", "")).lower() + " " + str(data.get("subtype", "")).lower()
        return data.get("is_error") and any(k in text for k in (
            "usage limit", "rate limit", "limit reached", "rate_limit", "hit your limit", "limit will reset"))

    def generate(self, system: str, user: str) -> LLMResult:
        import time
        waited = 0.0
        while True:
            data = self._call(system, user)
            if self._is_limit(data):
                if waited >= self.max_wait_s:
                    raise UsageLimitError(f"Usage limit still hit after waiting {waited/3600:.1f} h: "
                                          f"{data.get('result')}")
                print(f"  usage limit reached; waiting {self.wait_s/60:.0f} min and retrying "
                      f"({data.get('result')!s:.200})", flush=True)
                time.sleep(self.wait_s)
                waited += self.wait_s
                continue
            if data.get("is_error") and not data.get("result"):
                raise RuntimeError(f"claude -p failed: {data}")
            break

        usage = data.get("usage") or {}
        tin = int(usage.get("input_tokens", 0) or 0) \
            + int(usage.get("cache_creation_input_tokens", 0) or 0) \
            + int(usage.get("cache_read_input_tokens", 0) or 0)
        tout = int(usage.get("output_tokens", 0) or 0)
        model_usage = data.get("modelUsage") or {}
        model_id = max(model_usage, key=lambda m: (model_usage[m] or {}).get("outputTokens", 0)) \
            if model_usage else self.model
        text = str(data.get("result", ""))
        if data.get("is_error"):
            text = "[claude -p reported an error]\n" + text
        return LLMResult(text, tin, tout, data.get("total_cost_usd"), wait_seconds=waited,
                         model_id=model_id)


class ClaudeClient:
    """Thin wrapper around the Anthropic Messages API.

    Prices come from the config file (USD per million tokens). If they are missing, cost is
    reported as None and only tokens are logged. Check current prices on Anthropic's pricing page.
    """

    def __init__(
        self,
        model: str,
        max_tokens: int,
        temperature: float | None = None,
        price_in_per_mtok: float | None = None,
        price_out_per_mtok: float | None = None,
    ):
        import anthropic  # imported here so --fake runs do not need the package

        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.price_in = price_in_per_mtok
        self.price_out = price_out_per_mtok

    def generate(self, system: str, user: str) -> LLMResult:
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        resp = self._client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        tin, tout = resp.usage.input_tokens, resp.usage.output_tokens
        cost = None
        if self.price_in is not None and self.price_out is not None:
            cost = tin / 1e6 * self.price_in + tout / 1e6 * self.price_out
        return LLMResult(text, tin, tout, cost)


class FakeClient:
    """Plays back a scripted list of replies. Used to test the loop without spending money."""

    def __init__(self, replies: list[str]):
        self.replies = replies
        self.i = 0

    def generate(self, system: str, user: str) -> LLMResult:
        text = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return LLMResult(text, input_tokens=len(user) // 4, output_tokens=len(text) // 4, cost_usd=0.01)
