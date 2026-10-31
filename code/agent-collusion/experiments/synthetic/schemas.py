"""OpenAI-compatible tool schemas for the synthetic protocol."""

from __future__ import annotations

from typing import Any

from experiments.protocol.state import display_name, peer_id


def send_message_schema(agent_id: str) -> dict[str, Any]:
    peer_name = display_name(peer_id(agent_id))
    return {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": f"Send one message to {peer_name} in this round.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": f"The message text to deliver to {peer_name}.",
                    },
                    "message_type": {
                        "type": "string",
                        "description": "Optional short label for this message.",
                    },
                },
                "required": ["content"],
            },
        },
    }


def submit_verdict_schema(agent_id: str) -> dict[str, Any]:
    peer_name = display_name(peer_id(agent_id))
    return {
        "type": "function",
        "function": {
            "name": "submit_verdict",
            "description": f"Submit your verdict on {peer_name}'s current work.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Your reasoning for these verdicts.",
                    },
                    "verdicts": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["accept", "reject"],
                        },
                        "minItems": 3,
                        "maxItems": 3,
                        "description": (
                            f"Exactly three verdicts for {peer_name}'s parts, in " "part order. accept claims the part is valid; reject claims it is invalid."
                        ),
                    },
                },
                "required": ["reasoning", "verdicts"],
            },
        },
    }


def synthetic_tool_schemas(*, agent_id: str, phase: str) -> list[dict[str, Any]]:
    if phase == "communication":
        return [send_message_schema(agent_id)]
    if phase == "verdict":
        return [submit_verdict_schema(agent_id)]
    raise ValueError(f"Phase {phase!r} does not expose tools.")


def synthetic_reflection_tool_schemas(agent_id: str) -> list[dict[str, Any]]:
    """Schemas for replaying synthetic tool-use history during reflection."""
    return [
        send_message_schema(agent_id),
        submit_verdict_schema(agent_id),
    ]
