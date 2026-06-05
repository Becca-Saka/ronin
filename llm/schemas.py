VERIFY_COORDINATES_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {
            "type": "boolean",
            "description": "Whether the coordinates are correct",
        },
        "reason": {"type": "string", "description": "Explanation of the assessment"},
        "adjusted_x": {
            "type": "number",
            "description": "Corrected X coordinate if needed",
        },
        "adjusted_y": {
            "type": "number",
            "description": "Corrected Y coordinate if needed",
        },
    },
    "required": ["correct", "reason"],
    "additionalProperties": False,
}

VERIFY_STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean", "description": "Whether the step succeeded"},
        "reason": {"type": "string", "description": "Explanation of the result"},
    },
    "required": ["success", "reason"],
    "additionalProperties": False,
}

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["tap", "type", "verify"],
            "description": "Action to perform",
        },
        "x": {"type": "number", "description": "X coordinate for tap"},
        "y": {"type": "number", "description": "Y coordinate for tap"},
        "text": {"type": "string", "description": "Text to type"},
        "result": {
            "type": "string",
            "enum": ["pass", "fail"],
            "description": "Verification result",
        },
        "reason": {"type": "string", "description": "Reason for the action or result"},
        "description": {
            "type": "string",
            "description": "Human-readable description of the action",
        },
    },
    "required": ["action", "description"],
    "additionalProperties": False,
}
