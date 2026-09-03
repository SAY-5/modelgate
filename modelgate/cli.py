"""Command line entry point: `modelgate serve` and `modelgate train`."""

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

    args = parser.parse_args(argv)
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
