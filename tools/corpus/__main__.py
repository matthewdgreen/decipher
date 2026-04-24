from __future__ import annotations

import argparse
from pathlib import Path

from .build_model import build_model
from .sources.gutenberg import download_books, download_catalog
from .verify_model import verify_model


def _default_output(language: str) -> Path:
    return Path("models") / f"ngram5_{language}.bin"


def _default_corpus_dir(language: str) -> Path:
    return Path("corpus_data") / language


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tools.corpus")
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download")
    download.add_argument("language", choices=["en"])
    download.add_argument("--output-dir", type=Path, default=None)
    download.add_argument("--max-books", type=int, default=100)

    build = sub.add_parser("build")
    build.add_argument("language", choices=["en", "de", "fr", "it", "la"])
    build.add_argument("--corpus-dir", type=Path, required=True)
    build.add_argument("--output", type=Path, default=None)

    verify = sub.add_parser("verify")
    verify.add_argument("model_path", type=Path)

    run = sub.add_parser("run")
    run.add_argument("language", choices=["en"])
    run.add_argument("--output", type=Path, default=None)
    run.add_argument("--corpus-dir", type=Path, default=None)
    run.add_argument("--max-books", type=int, default=100)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "download":
        output_dir = args.output_dir or _default_corpus_dir(args.language)
        catalog = download_catalog(output_dir)
        paths = download_books(
            catalog_path=catalog,
            output_dir=output_dir,
            language=args.language,
            max_books=args.max_books,
        )
        print(f"Downloaded {len(paths)} books to {output_dir}")
        return 0

    if args.command == "build":
        output = args.output or _default_output(args.language)
        stats = build_model(
            language=args.language,
            corpus_dir=args.corpus_dir,
            output_path=output,
        )
        print(f"Built {output} from {stats.raw_files} files")
        print(f"Metadata: {stats.metadata_path}")
        return 0

    if args.command == "verify":
        result = verify_model(args.model_path)
        print(f"Verified {args.model_path}")
        print(f"Unknown log prob: {result['unknown_log_prob']}")
        return 0

    if args.command == "run":
        corpus_dir = args.corpus_dir or _default_corpus_dir(args.language)
        output = args.output or _default_output(args.language)
        catalog = download_catalog(corpus_dir)
        download_books(
            catalog_path=catalog,
            output_dir=corpus_dir,
            language=args.language,
            max_books=args.max_books,
        )
        stats = build_model(
            language=args.language,
            corpus_dir=corpus_dir,
            output_path=output,
        )
        verify_model(output)
        print(f"Built and verified {output}")
        print(f"Metadata: {stats.metadata_path}")
        return 0

    parser.error(f"unsupported command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
