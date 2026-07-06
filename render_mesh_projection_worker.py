import contextlib
import json
import sys

from render_mesh_projection_cpu import project_mesh_from_base_dir


def main() -> None:
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("cmd") == "quit":
                print(json.dumps({"ok": True}), flush=True)
                return
            if request.get("cmd") != "project":
                raise ValueError(f"Unknown command: {request.get('cmd')}")
            base_dir = request["base_dir"]
            workers = int(request.get("workers", 1))
            with contextlib.redirect_stdout(sys.stderr):
                paths = project_mesh_from_base_dir(base_dir, workers=workers)
            print(json.dumps({"ok": True, "paths": paths}), flush=True)
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
