"""A tiny stand-in for llama-server, used by the tests. Understands the few flags Reel passes and the /health
and /v1/chat/completions endpoints (streaming and not)."""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
alias = args[args.index("--alias") + 1] if "--alias" in args else "model"
print("fake llama-server args:", " ".join(args), flush=True)
if "--crash" in args:
    sys.exit(3)
started = time.time()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._json(200 if time.time() - started > 0.5 else 503, {"status": "ok"})
        else:
            self._json(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path != "/v1/chat/completions":
            return self._json(404, {})
        if body.get("model") != alias:
            return self._json(400, {"error": {"message": "wrong alias", "type": "invalid_request_error"}})
        text = "Hello from " + alias + ": " + body["messages"][-1]["content"]
        if not body.get("stream"):
            return self._json(200, {"choices": [{"message": {"role": "assistant", "content": text}}]})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunks = [{"choices": [{"delta": {"reasoning_content": "thinking"}}]}]
        chunks += [{"choices": [{"delta": {"content": w + " "}}]} for w in text.split()]
        chunks.append({"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": len(text.split())},
                       "timings": {"predicted_per_second": 42.0}})
        for c in chunks:
            self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
