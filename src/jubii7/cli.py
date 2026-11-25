"""Command line: `jubii7 puzzle | check | train | predict | serve`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path.cwd()


def _puzzle(_: argparse.Namespace) -> int:
    from jubii7.puzzle.solver import solve

    solutions = solve()
    print(f"Solutions found: {len(solutions)} (1 = the answer is unique)")
    for people in solutions:
        for p in people:
            print(f"  {p.name:6} {p.emirate:10} {p.drink:9} {p.job}")
        chef = next(p.name for p in people if p.job == "chef")
        dev = next(p for p in people if p.name == "Dev")
        print(f"\nChef: {chef}. Dev lives in: {dev.emirate}.")
    return 0 if len(solutions) == 1 else 1


def _check(args: argparse.Namespace) -> int:
    from jubii7.invoice import default_checker, load_invoice

    report = default_checker().run(load_invoice(args.invoice))
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 2 if report.has_errors and args.strict else 0


def _train(args: argparse.Namespace) -> int:
    from jubii7.classifier.train import train

    card = train(ROOT, with_curves=not args.fast)
    print(json.dumps({k: card[k] for k in ("model", "runs", "holdout")}, indent=2))
    return 0


def _predict(args: argparse.Namespace) -> int:
    from jubii7.api.app import default_classifier

    print(json.dumps(default_classifier(ROOT).predict(args.text).to_dict(), ensure_ascii=False, indent=2))
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("jubii7.api.app:app", host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jubii7", description=__doc__)
    sub = parser.add_subparsers(required=True)

    sub.add_parser("puzzle", help="solve the warm-up puzzle").set_defaults(func=_puzzle)

    check = sub.add_parser("check", help="profile and check an invoice JSON")
    check.add_argument("invoice", nargs="?", default="data/invoice_WB-2026-11-0457.json")
    check.add_argument("--strict", action="store_true", help="exit code 2 if any ERROR is found")
    check.set_defaults(func=_check)

    train = sub.add_parser("train", help="train, compare, evaluate and save the classifier")
    train.add_argument("--fast", action="store_true", help="skip the diagnostic curves")
    train.set_defaults(func=_train)

    predict = sub.add_parser("predict", help="classify one invoice line")
    predict.add_argument("text")
    predict.set_defaults(func=_predict)

    serve = sub.add_parser("serve", help="run the API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
