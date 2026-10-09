import argparse
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import litellm
from dotenv import load_dotenv

load_dotenv()

SYSTEM_MESSAGE = """You are an AI agent controlling a mobile app on Android or iOS to achieve a test goal.
Your job is to execute instructions by interacting with the app one action at a time using the provided tools.

### Goal / Instructions
- Sign in to the app with email 'test@email.com' and password 'password123'
- Click on the first notification item in the list
- On the detail screen, type 'Awesome notification!' in the reply field
- Toggle the 'Pin notification' switch to true
- Click the 'Submit' button

### Current UI Elements Detected (YOLO/OCR)
- button | text: 'Sign In' | center: (540, 1820)
- text_field | text: 'Email' | center: (540, 960)
- text_field | text: 'Password' | center: (540, 1120)
- switch | text: 'Remember me' | center: (880, 1280)
- text | text: 'Forgot password?' | center: (540, 1420)
- image | center: (540, 420)

### Keyboard State
HIDDEN — the soft keyboard is not visible. All detected elements are app UI.

### Action History (with system-observed outcomes)
Step 1: tap(540, 960) — Success — email field focused, keyboard appeared
Step 2: type('test@email.com') at (540, 960) — Success — email entered
Step 3: key_event(4) — Success — keyboard dismissed
Step 4: tap(540, 1120) — Success — password field focused

### Session Trail — App Knowledge From This Session
- Login screen uses standard email/password layout with Remember me toggle
- Submit button only appears after both fields are filled
- Tapping outside a field does not dismiss keyboard on Android; use back key (4)

### Allowed Action Types
Each action must use exactly one tool call. Available interactions:
- tap: tap at pixel coordinates
- type: enter text into a field (must tap to focus first)
- clear: clear a text field at coordinates
- double_tap, long_press, drag_and_drop, swipe: gesture interactions
- key_event: system keys (4=back, 3=home, 66=enter, 67=backspace)
- capability: system-level mocks (location, network, battery, permissions, clipboard, deeplink, app state)
- done: signal goal complete or execution must stop

CRITICAL CONSTRAINTS:
1. Use tap for clicking and type for entering text. Never use click or send_keys.
2. For type, you MUST provide both text AND coordinates of the target field.
3. Before typing into a field, first tap to focus it.
4. The soft keyboard blocks taps behind it. Dismiss with key_event 4 before tapping lower-screen controls.
5. Do NOT repeat an action that succeeded unless the system outcome shows it failed.
6. Use center coordinates from the Current UI Elements list. Do not guess from memory.
7. When the keyboard is visible, only use input elements in the upper portion of the screen.
8. Set external_navigation true if the action leaves the app (browser, payment, system dialog).

### Response Instructions
1. Set current_screen to the descriptive name of the screen you see.
2. Set evaluation_previous_goal: assess whether the previous action succeeded, failed, or is uncertain.
3. Set observation: describe only what is currently visible.
4. Set reasoning: explain why the chosen action is the best next step.
5. Choose exactly ONE tool call per turn. Use done when the goal is complete.
6. When the user greets you casually with no active task, respond briefly in plain text — do not call tools."""

USER_MESSAGE = "hi"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "tap",
            "description": "Tap at screen coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "X coordinate in pixels"},
                    "y": {"type": "integer", "description": "Y coordinate in pixels"},
                    "reason": {"type": "string", "description": "Why this tap advances the goal"},
                },
                "required": ["x", "y", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type",
            "description": "Type text into a field at the given coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to enter"},
                    "x": {"type": "integer", "description": "X coordinate of the target field"},
                    "y": {"type": "integer", "description": "Y coordinate of the target field"},
                    "reason": {"type": "string", "description": "Why this input advances the goal"},
                },
                "required": ["text", "x", "y", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clear",
            "description": "Clear a text field at the given coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["x", "y", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "double_tap",
            "description": "Double-tap at screen coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["x", "y", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "long_press",
            "description": "Long press at screen coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "duration": {"type": "number", "description": "Duration in seconds"},
                    "reason": {"type": "string"},
                },
                "required": ["x", "y", "duration", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "drag_and_drop",
            "description": "Drag from source coordinates to destination coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_x": {"type": "integer"},
                    "source_y": {"type": "integer"},
                    "destination_x": {"type": "integer"},
                    "destination_y": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["source_x", "source_y", "destination_x", "destination_y", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "swipe",
            "description": "Swipe from start to end coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_x": {"type": "integer"},
                    "start_y": {"type": "integer"},
                    "end_x": {"type": "integer"},
                    "end_y": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["start_x", "start_y", "end_x", "end_y", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "key_event",
            "description": "Trigger a system key event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "integer",
                        "description": "Keycode: 4=back, 3=home, 66=enter, 67=backspace",
                    },
                    "reason": {"type": "string"},
                },
                "required": ["key", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_location",
            "description": "Mock device GPS location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["latitude", "longitude", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_network",
            "description": "Mock device network connectivity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["offline", "online", "wifi_only", "data_only", "airplane"],
                    },
                    "reason": {"type": "string"},
                },
                "required": ["mode", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_permission",
            "description": "Grant or revoke an app permission.",
            "parameters": {
                "type": "object",
                "properties": {
                    "permission": {
                        "type": "string",
                        "enum": ["camera", "location", "notifications", "photos", "contacts"],
                    },
                    "grant": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["permission", "grant", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_battery",
            "description": "Mock device battery level and charging state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {"type": "integer", "description": "Battery percentage 0-100"},
                    "charging": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["level", "charging", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_clipboard",
            "description": "Set the device clipboard contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["text", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_deeplink",
            "description": "Open a deeplink URL in the app or system handler.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["url", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_app_state",
            "description": "Force-stop the app or clear its data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["clear_data", "force_stop"]},
                    "reason": {"type": "string"},
                },
                "required": ["action", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Signal that the goal is complete or execution must stop.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Why execution is stopping"},
                },
                "required": ["reason"],
            },
        },
    },
]

GEMINI_MODEL = "gemini/gemini-2.5-flash"
OPENROUTER_MODEL = "openrouter/google/gemini-2.5-flash"


def format_response(message) -> str:
    parts = []
    if message.content:
        parts.append(message.content.strip())
    tool_calls = getattr(message, "tool_calls", None) or []
    for call in tool_calls:
        fn = call.function
        parts.append(f"[tool] {fn.name}({fn.arguments})")
    return " | ".join(parts) if parts else "(empty)"


def call_provider(name: str, model: str) -> dict:
    start = time.perf_counter()
    try:
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": USER_MESSAGE},
            ],
            tools=TOOLS,
            max_tokens=128,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        message = response.choices[0].message
        return {
            "name": name,
            "model": model,
            "latency_ms": latency_ms,
            "content": format_response(message),
            "error": None,
        }
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "name": name,
            "model": model,
            "latency_ms": latency_ms,
            "content": None,
            "error": str(exc),
        }


JOBS = [
    ("Gemini (direct)", GEMINI_MODEL),
    ("OpenRouter", OPENROUTER_MODEL),
]


def run_round() -> list[dict]:
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            pool.submit(call_provider, name, model): name for name, model in JOBS
        }
        for future in as_completed(futures):
            results.append(future.result())
    return results


def print_round(results: list[dict]):
    ordered = sorted(results, key=lambda r: r["latency_ms"])
    for rank, result in enumerate(ordered, start=1):
        status = "ERROR" if result["error"] else "OK"
        print(f"{rank}. {result['name']} — {result['latency_ms']:.0f} ms [{status}]")
        if result["error"]:
            print(f"   {result['error']}")
        elif result["content"]:
            print(f"   {result['content']!r}")

    ok = [r for r in results if not r["error"]]
    if len(ok) == 2:
        winner = min(ok, key=lambda r: r["latency_ms"])
        loser = max(ok, key=lambda r: r["latency_ms"])
        delta = loser["latency_ms"] - winner["latency_ms"]
        print(
            f"\nFirst to respond: {winner['name']} "
            f"({delta:.0f} ms faster than {loser['name']})"
        )


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * p / 100
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    weight = idx - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def print_stats(name: str, latencies: list[float]):
    print(f"\n{name}")
    print(f"  runs:   {len(latencies)}")
    print(f"  mean:   {statistics.mean(latencies):.0f} ms")
    print(f"  median: {statistics.median(latencies):.0f} ms")
    print(f"  min:    {min(latencies):.0f} ms")
    print(f"  max:    {max(latencies):.0f} ms")
    print(f"  p95:    {percentile(latencies, 95):.0f} ms")
    if len(latencies) > 1:
        print(f"  stdev:  {statistics.stdev(latencies):.0f} ms")


def benchmark(runs: int, delay_s: float):
    print(f"Benchmark: {runs} rounds, {delay_s}s delay between rounds")
    print(f"System: {len(SYSTEM_MESSAGE)} chars, {len(TOOLS)} tools")
    print(f"User: {USER_MESSAGE!r}\n")

    by_name: dict[str, list[float]] = {name: [] for name, _ in JOBS}
    wins: dict[str, int] = {name: 0 for name, _ in JOBS}
    errors: dict[str, int] = {name: 0 for name, _ in JOBS}

    for i in range(1, runs + 1):
        print(f"--- Round {i}/{runs} ---")
        results = run_round()
        print_round(results)
        print()

        ok = [r for r in results if not r["error"]]
        for result in results:
            if result["error"]:
                errors[result["name"]] += 1
            else:
                by_name[result["name"]].append(result["latency_ms"])

        if len(ok) == 2:
            winner = min(ok, key=lambda r: r["latency_ms"])
            wins[winner["name"]] += 1

        if i < runs and delay_s > 0:
            time.sleep(delay_s)

    print("=" * 40)
    print("SUMMARY")
    print("=" * 40)

    for name, _ in JOBS:
        if by_name[name]:
            print_stats(name, by_name[name])
        else:
            print(f"\n{name}: no successful runs")

    total_races = sum(wins.values())
    if total_races:
        print("\nHead-to-head wins")
        for name, count in wins.items():
            pct = 100 * count / total_races
            print(f"  {name}: {count}/{total_races} ({pct:.0f}%)")

    if any(errors.values()):
        print("\nErrors")
        for name, count in errors.items():
            if count:
                print(f"  {name}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Compare Gemini direct vs OpenRouter latency")
    parser.add_argument("--benchmark", action="store_true", help="Run multiple rounds and print stats")
    parser.add_argument("--runs", type=int, default=10, help="Benchmark rounds (default: 10)")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between benchmark rounds")
    args = parser.parse_args()

    missing = []
    if not os.getenv("GEMINI_API_KEY"):
        missing.append("GEMINI_API_KEY")
    if not os.getenv("OPENROUTER_API_KEY"):
        missing.append("OPENROUTER_API_KEY")
    if missing:
        raise EnvironmentError(f"Missing env vars: {', '.join(missing)}")

    if args.benchmark:
        benchmark(args.runs, args.delay)
        return

    print(f"System: {len(SYSTEM_MESSAGE)} chars, {len(TOOLS)} tools")
    print(f"User: {USER_MESSAGE!r}\n")
    print_round(run_round())


if __name__ == "__main__":
    main()
