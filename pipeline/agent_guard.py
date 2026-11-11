# -*- coding: utf-8 -*-
"""
Agent Guardrails and Language Safety.
Enforces:
  1. Maximum 6 tool calls per agent request.
  2. Maximum 20-second timeout per agent request.
  3. Language guard: forbids definitive medical diagnoses (AML, ALL, MDS, etc.)
     in agent outputs and directs to physician evaluation.
"""

import time
import re
from typing import Any, Callable, Dict, List, Optional
from app.db import log_agent_event


FORBIDDEN_DIAGNOSTIC_PATTERNS = [
    r"\b(?:diagnosed with|has|suffers from|confirms)\s+(?:AML|ALL|acute myeloid leukemia|acute lymphoblastic leukemia|MDS|myelodysplastic|leukemia|lymphoma)\b",
    r"\bpatient has (?:AML|ALL|leukemia)\b",
    r"\bpositive for (?:acute leukemia|AML|ALL)\b",
    r"\bdefinitive diagnosis:?\s*(?:AML|ALL|leukemia)\b",
]

DIAGNOSIS_DISCLAIMER = (
    " [Clinical note: Cytomorphology AI provides descriptive cell analysis only. "
    "Definitive clinical diagnosis requires expert physician evaluation, flow cytometry, "
    "and cytogenetic/molecular correlation.]"
)


def sanitize_agent_language(text: str) -> str:
    """
    Ensure agent language does not state or assert an unverified medical diagnosis.
    Replaces definitive diagnostic claims with descriptive findings and clinical disclaimer.
    """
    if not text:
        return text

    sanitized = text
    matched = False
    for pattern in FORBIDDEN_DIAGNOSTIC_PATTERNS:
        if re.search(pattern, sanitized, re.IGNORECASE):
            matched = True
            sanitized = re.sub(
                pattern,
                "findings suggestive of increased blast/atypical cell proportion requiring physician correlation",
                sanitized,
                flags=re.IGNORECASE,
            )

    if matched and DIAGNOSIS_DISCLAIMER not in sanitized:
        sanitized += DIAGNOSIS_DISCLAIMER

    return sanitized


class AgentExecutionGuard:
    """
    Tracks and limits tool executions per agent request.
    Enforces maximum 6 tool calls and 20-second wall-clock timeout.
    """

    def __init__(self, agent_name: str, case_id: str = "", max_calls: int = 6, timeout_seconds: float = 20.0):
        self.agent_name = agent_name
        self.case_id = case_id
        self.max_calls = max_calls
        self.timeout_seconds = timeout_seconds
        self.start_time = time.time()
        self.call_count = 0
        self.history: List[Dict[str, Any]] = []

    def can_call(self) -> bool:
        if self.call_count >= self.max_calls:
            return False
        if (time.time() - self.start_time) > self.timeout_seconds:
            return False
        return True

    def record_step(self, tool_name: str, input_summary: str, output_summary: str, status: str = "success"):
        self.call_count += 1
        elapsed = round(time.time() - self.start_time, 2)
        step_entry = {
            "agent": self.agent_name,
            "tool": tool_name,
            "input": input_summary,
            "output": output_summary,
            "status": status,
            "step": self.call_count,
            "elapsed": elapsed,
        }
        self.history.append(step_entry)
        log_agent_event(
            case_id=self.case_id,
            agent=self.agent_name,
            tool=tool_name,
            input_summary=input_summary,
            output_summary=output_summary,
            status=status,
        )

    def execute_tool(self, tool_name: str, func: Callable, *args, **kwargs) -> Any:
        if not self.can_call():
            reason = "Call limit exceeded (max 6)" if self.call_count >= self.max_calls else "Timeout exceeded (20s)"
            self.record_step(tool_name, str(args)[:100], reason, status="aborted")
            raise RuntimeError(f"Agent tool execution blocked: {reason}")

        in_summary = f"{tool_name}({', '.join([str(a) for a in args] + [f'{k}={v}' for k,v in kwargs.items()])})"[:200]
        try:
            result = func(*args, **kwargs)
            out_summary = str(result)[:200]
            self.record_step(tool_name, in_summary, out_summary, status="success")
            return result
        except Exception as e:
            self.record_step(tool_name, in_summary, f"Error: {e}"[:200], status="error")
            raise

