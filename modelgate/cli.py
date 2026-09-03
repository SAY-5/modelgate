"""Command line entry point: `modelgate serve`, `modelgate train`, and `modelgate eval`."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="modelgate")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the serving API")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--log-level", default="info")

    train = sub.add_parser("train", help="train model versions and write artifacts")
    train.add_argument("--out", default="artifacts")
    train.add_argument("--seed", type=int, default=7)

    ev = sub.add_parser("eval", help="replay a request log against model versions")
    ev.add_argument("--log", required=True)
    ev.add_argument("--versions", nargs="+", required=True)
    ev.add_argument("--artifacts", default="artifacts")
    ev.add_argument("--truth", choices=("auto", "reference", "none"), default="auto")
    ev.add_argument("--threshold", type=float, default=2.0)
    ev.add_argument("--json", default=None)

    args = parser.parse_args(argv)
    if args.command == "eval":
        from modelgate.eval import main as eval_main

        eval_argv = ["--log", args.log, "--versions", *args.versions]
        eval_argv += ["--artifacts", args.artifacts, "--truth", args.truth]
        eval_argv += ["--threshold", str(args.threshold)]
        if args.json:
            eval_argv += ["--json", args.json]
        return eval_main(eval_argv)
    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "modelgate.serving.app:app", host=args.host, port=args.port, log_level=args.log_level
        )
        return 0
    if args.command == "train":
        from modelgate.model.train import main as train_main

        train_main(["--out", args.out, "--seed", str(args.seed)])
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
