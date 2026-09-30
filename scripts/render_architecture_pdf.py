"""Print the architecture brief using an already-running local Chrome debugger."""
from __future__ import annotations
import argparse
import base64
import html
import json
from pathlib import Path
import re
import urllib.request
import websocket


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("submission/system-architecture-brief.pdf"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    blocks = []
    for block in (root / "docs/system-architecture-brief.md").read_text().split("\n\n"):
        text = html.escape(block.strip())
        if text.startswith("# "):
            blocks.append("<h1>" + text[2:] + "</h1>")
        elif text.startswith("## "):
            blocks.append("<h2>" + text[3:] + "</h2>")
        elif re.match(r"\d+\. ", text):
            blocks.append("<ol>" + "".join("<li>" + re.sub(r"^\d+\. ", "", line) + "</li>" for line in text.splitlines()) + "</ol>")
        elif text.startswith("- "):
            blocks.append("<ul>" + "".join("<li>" + line.removeprefix("- ") + "</li>" for line in text.splitlines()) + "</ul>")
        else:
            blocks.append("<p>" + text.replace("\n", " ") + "</p>")
    markup = "<!doctype html><meta charset=utf-8><style>body{font:11pt/1.45 Arial;color:#172820}h1{font-size:22pt}h2{font-size:14pt;break-after:avoid}li{margin-bottom:8pt}p{orphans:3;widows:3}</style>" + "".join(blocks)
    page = Path("/tmp/prism-architecture-print-20260930.html")
    page.write_text(markup)
    request = urllib.request.Request("http://127.0.0.1:9222/json/new?" + page.as_uri(), method="PUT")
    with urllib.request.urlopen(request) as response:
        target = json.load(response)
    ws = websocket.create_connection(target["webSocketDebuggerUrl"], suppress_origin=True, timeout=30)
    ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate", "params": {
        "expression": "new Promise(resolve=>{const check=()=>document.body?.innerText.length>500?resolve(true):setTimeout(check,50);check();})",
        "awaitPromise": True, "returnByValue": True}}))
    while json.loads(ws.recv()).get("id") != 1:
        pass
    ws.send(json.dumps({"id": 2, "method": "Page.printToPDF", "params": {"printBackground": True, "paperWidth": 8.27, "paperHeight": 11.69, "marginTop": .65, "marginBottom": .65, "marginLeft": .65, "marginRight": .65}}))
    while True:
        result = json.loads(ws.recv())
        if result.get("id") == 2:
            if "error" in result:
                raise RuntimeError(result["error"])
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(base64.b64decode(result["result"]["data"]))
            break
    ws.close()
    print(args.output)


if __name__ == "__main__":
    main()
