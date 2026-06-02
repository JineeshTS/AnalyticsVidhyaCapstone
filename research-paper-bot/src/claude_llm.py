"""
Claude CLI chat model -- the LLM backend for this project.

Instead of calling a paid API, this adapter shells out to the locally-installed
`claude` CLI in headless mode (`claude -p --output-format json`). That keeps the
whole RAG pipeline running on a Claude Max subscription with **zero per-call API
spend**, while still presenting a standard LangChain chat-model interface so the
rest of the codebase (rag.py, crag.py, evaluate.py, app.py) is unchanged.

Why a custom model instead of ChatOpenAI:
  - the capstone is being run on a VPS whose only LLM credential is the Claude
    CLI (no OpenAI key);
  - generation, the CRAG relevance grader, the query rewriter and the eval judge
    all route through here, so one adapter swaps the entire LLM layer.

Design notes:
  - The prompt is sent on **stdin** (not argv) so large RAG contexts and special
    characters are passed safely.
  - System messages become `--system-prompt`; a short guard is appended so the
    model answers directly and never tries to use tools.
  - `with_structured_output()` is implemented via JSON-mode prompting + Pydantic
    validation, because the CLI has no native tool/function-calling surface.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Type

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

import config

# A guard appended to every system prompt so the CLI behaves like a plain
# text-completion endpoint rather than an autonomous coding agent. It is kept
# task-neutral on purpose: it must NOT bias the model toward "answering" (which
# would break non-Q&A tasks like the follow-up-question rewriter), only forbid
# tool use and session side-effects.
_TOOL_GUARD = (
    "Follow the instructions above exactly and respond with the result only. "
    "Do not use any tools, do not search the web, do not read or write files, "
    "and do not ask follow-up questions."
)


class ClaudeCLIError(RuntimeError):
    """Raised when the `claude` CLI call fails or returns an error result."""


def _split_messages(messages: List[BaseMessage]) -> tuple[str, str]:
    """Split LangChain messages into (system_prompt, user_prompt).

    System messages are concatenated into the --system-prompt. Everything else
    is rendered into a single stdin prompt; a multi-turn list is flattened into
    a "Human:/Assistant:" transcript so conversation context survives.
    """
    system_parts: List[str] = []
    convo: List[BaseMessage] = []
    for m in messages:
        if isinstance(m, SystemMessage) or m.type == "system":
            system_parts.append(str(m.content))
        else:
            convo.append(m)

    system_prompt = "\n\n".join(p for p in system_parts if p).strip()

    if len(convo) == 1:
        user_prompt = str(convo[0].content)
    else:
        lines = []
        for m in convo:
            speaker = "Assistant" if m.type == "ai" else "Human"
            lines.append(f"{speaker}: {m.content}")
        user_prompt = "\n".join(lines)

    return system_prompt, user_prompt


def _extract_json(text: str) -> Dict[str, Any]:
    """Pull the first JSON object out of a model response (handles ``` fences)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ClaudeCLIError(f"No JSON object found in model output:\n{text}")
    return json.loads(candidate[start : end + 1])


class ClaudeCLIChat(BaseChatModel):
    """LangChain chat model backed by the local `claude` CLI (headless)."""

    model: str = config.CLAUDE_MODEL
    timeout: int = config.CLAUDE_TIMEOUT
    claude_bin: str = config.CLAUDE_BIN
    temperature: float = 0.0  # accepted for API parity; CLI uses its own default

    @property
    def _llm_type(self) -> str:
        return "claude-cli"

    def _call_cli(self, system_prompt: str, user_prompt: str) -> str:
        binary = shutil.which(self.claude_bin) or self.claude_bin
        sys_prompt = (system_prompt + "\n\n" + _TOOL_GUARD).strip() if system_prompt else _TOOL_GUARD
        cmd = [
            binary,
            "-p",
            "--output-format", "json",
            "--model", self.model,
            "--no-session-persistence",
            "--system-prompt", sys_prompt,
        ]
        try:
            proc = subprocess.run(
                cmd,
                input=user_prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise ClaudeCLIError(f"claude CLI timed out after {self.timeout}s") from e

        if proc.returncode != 0:
            raise ClaudeCLIError(
                f"claude CLI exited {proc.returncode}: {proc.stderr.strip()[:500]}"
            )
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise ClaudeCLIError(
                f"claude CLI returned non-JSON output:\n{proc.stdout[:500]}"
            ) from e

        if payload.get("is_error"):
            raise ClaudeCLIError(f"claude CLI error result: {payload.get('result')}")
        return str(payload.get("result", "")).strip()

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        system_prompt, user_prompt = _split_messages(messages)
        text = self._call_cli(system_prompt, user_prompt)
        message = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def with_structured_output(
        self, schema: Type[BaseModel], **kwargs: Any
    ) -> Runnable:
        """Return a runnable that produces a validated `schema` instance.

        The CLI has no tool-calling surface, so structured output is done with
        JSON-mode prompting: we describe the required JSON shape, then parse and
        validate the response with Pydantic.
        """
        json_schema = schema.model_json_schema()
        props = json_schema.get("properties", {})
        field_help = "\n".join(
            f'  - "{name}": {spec.get("description", spec.get("type", ""))}'
            for name, spec in props.items()
        )
        example = {name: spec.get("type", "value") for name, spec in props.items()}
        fmt_instruction = (
            "Respond with ONLY a single JSON object, no prose, no code fences. "
            f"It must have exactly these fields:\n{field_help}\n"
            f'Example shape: {json.dumps(example)}'
        )

        def _invoke(prompt_value: Any) -> BaseModel:
            messages = list(prompt_value.to_messages())
            messages.append(SystemMessage(content=fmt_instruction))
            system_prompt, user_prompt = _split_messages(messages)
            text = self._call_cli(system_prompt, user_prompt)
            data = _extract_json(text)
            return schema(**data)

        return RunnableLambda(_invoke)


if __name__ == "__main__":
    llm = ClaudeCLIChat()
    print("smoke test ->", llm.invoke("Reply with exactly: OK").content)
