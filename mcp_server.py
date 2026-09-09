import sys
import json
from client import FlashAttentionKernel

kernel = FlashAttentionKernel(block_size=4)

def handle_call(name, arguments):
    if name == "forward":
        q = arguments["q"]
        k = arguments["k"]
        v = arguments["v"]
        b = arguments.get("block_size", 4)
        k_inst = FlashAttentionKernel(block_size=b)
        out = k_inst.forward(q, k, v)
        return {"output": out}
    return {"error": f"Unknown tool: {name}"}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            res = handle_call(req.get("name"), req.get("arguments", {}))
            print(json.dumps({"id": req.get("id"), "result": res}))
            sys.stdout.flush()
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            sys.stdout.flush()

if __name__ == "__main__":
    main()
