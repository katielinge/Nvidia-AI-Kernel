"""LLM clients: the real Anthropic client and a scripted fake for testing."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None  # None when no prices are configured


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
