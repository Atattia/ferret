"""Explicit offline-index maintenance and opt-in model downloads."""
import argparse
import json
from core.maintenance import PRESETS, install_model, install_ocr, rebuild, config_lock, atomic_json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Ferret model and index maintenance")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download and validate a pinned ONNX model")
    download.add_argument("preset", choices=PRESETS)
    download.add_argument("destination")
    ocr = commands.add_parser("download-ocr", help="Install Arabic and English Tesseract data in a local folder")
    ocr.add_argument("destination", nargs="?", default="models/tessdata")
    build = commands.add_parser("rebuild", help="Resume a separate index build; active index is preserved")
    build.add_argument("--config", default="config/settings.json")
    build.add_argument("--model", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--reranker")
    build.add_argument("--activate", action="store_true")
    rollback = commands.add_parser("rollback", help="Restore the previous model/index configuration")
    rollback.add_argument("--config", default="config/settings.json")
    args = parser.parse_args()
    try:
        if args.command == "download":
            print(install_model(args.preset, args.destination))
        elif args.command == "download-ocr":
            print(install_ocr(args.destination))
        elif args.command == "rebuild":
            print(json.dumps(rebuild(args.config, args.model, args.output, args.activate, args.reranker), indent=2))
        else:
            with config_lock(args.config):
                previous = json.loads(Path(args.config + ".previous").read_text())
                if not Path(previous["db_path"]).expanduser().is_file():
                    raise ValueError("Previous database is missing")
                atomic_json(args.config, previous)
            print("Previous configuration restored. Restart Ferret.")
    except Exception as exc:
        parser.exit(1, f"ferret: {exc}\n")


if __name__ == "__main__":
    main()
