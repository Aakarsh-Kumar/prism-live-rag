"""Exercise real dashboard controls through Chrome DevTools; save DOM and screenshots.

Requires a local Chrome debugging port and websocket-client (verification only).
No browser dependency is added to the judge runtime.
"""
from __future__ import annotations
import argparse
import base64
import json
import time
import urllib.request
from pathlib import Path
import websocket


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18081")
    parser.add_argument("--debug-port", type=int, default=9222)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--mode", choices=("deterministic", "provider"), default="deterministic")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(f"http://127.0.0.1:{args.debug_port}/json/new?{args.url}", method="PUT")
    with urllib.request.urlopen(request) as response:
        target = json.load(response)
    ws = websocket.create_connection(target["webSocketDebuggerUrl"], suppress_origin=True, timeout=30)
    sequence = 0

    def cdp(method, params=None):
        nonlocal sequence
        sequence += 1
        ws.send(json.dumps({"id": sequence, "method": method, "params": params or {}}))
        while True:
            message = json.loads(ws.recv())
            if message.get("id") == sequence:
                if "error" in message:
                    raise RuntimeError(message["error"])
                return message.get("result", {})

    def evaluate(expression):
        result = cdp("Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True})
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"])
        return result.get("result", {}).get("value")

    cdp("Emulation.setDeviceMetricsOverride", {"width": 1440, "height": 1100, "deviceScaleFactor": 1, "mobile": False})
    deadline = time.monotonic() + 90
    while not evaluate("document.querySelector('#scenario')?.options.length > 10"):
        if time.monotonic() > deadline:
            raise RuntimeError("Dashboard scenarios did not become ready")
        time.sleep(1)
    with urllib.request.urlopen(args.url + "/api/scenarios") as response:
        metadata = json.load(response)
    candidates = [row for row in metadata["scenarios"] if row["is_evaluated_test"]]
    if args.mode == "provider" and not metadata["provider_available"]:
        raise RuntimeError("Provider mode unavailable; no silent mode substitution")
    selected, seen = [], set()
    for row in candidates:
        key = (row["domain"], row["case_class"], row["category"])
        if key not in seen:
            selected.append(row)
            seen.add(key)
    selected += [row for row in candidates if row not in selected]
    records = []
    for index, row in enumerate(selected[:args.limit], 1):
        evaluate(f"document.querySelector('#scenario').value={json.dumps(row['id'])}; document.querySelector('#scenario').dispatchEvent(new Event('change')); document.querySelector('#mode').value={json.dumps(args.mode)}; document.querySelector('#speed').value='4'; document.querySelector('#run').click();")
        deadline = time.monotonic() + 600
        while True:
            status = evaluate("document.querySelector('#run-status').textContent")
            if status == "Scenario completed":
                break
            if "error" in status.lower() or time.monotonic() > deadline:
                raise RuntimeError(f"{row['id']}: {status}")
            time.sleep(1)
        record = evaluate("JSON.parse(JSON.stringify({text:document.body.innerText, fields:Object.fromEntries([...document.querySelectorAll('[id]')].map(e=>[e.id,e.innerText])), events:document.querySelector('#raw-log').textContent.split('\\n').filter(Boolean).map(JSON.parse)}))")
        record.update(scenario_id=row["id"], query=row["description"], category=row["category"], case_class=row["case_class"])
        screenshot = cdp("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True})
        (args.output / f"{index:02d}.png").write_bytes(base64.b64decode(screenshot["data"]))
        records.append(record)
        (args.output / "results.json").write_text(json.dumps({"server": metadata, "runs": records}, indent=2))
        print(f"{index}/{args.limit} {row['domain']} {row['case_class']} {row['description'][:60]}", flush=True)
    ws.close()


if __name__ == "__main__":
    main()
