from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build_model import build_model
from .sources.anc import download_masc_texts, download_oanc_texts
from .sources.bnc import import_bnc_texts
from .sources.common import load_manifest
from .sources.gutenberg import download_gutenberg_books
from .verify_model import verify_model


SUPPORTED_LANGUAGES = ["en", "de", "fr", "it", "la"]
DEFAULT_SOURCES_BY_LANGUAGE = {
    "en": ["gutenberg"],
    "de": ["gutenberg"],
    "fr": ["gutenberg"],
    "it": ["gutenberg"],
    "la": ["gutenberg"],
}
SOURCE_CHOICES_BY_LANGUAGE = {
    "en": ["gutenberg", "oanc", "masc", "bnc"],
    "de": ["gutenberg"],
    "fr": ["gutenberg"],
    "it": ["gutenberg"],
    "la": ["gutenberg"],
}


def _default_output(language: str) -> Path:
    return Path("models") / f"ngram5_{language}.bin"


def _default_corpus_dir(language: str) -> Path:
    return Path("corpus_data") / language


def _download_sources(
    *,
    language: str,
    corpus_dir: Path,
    sources: list[str],
    max_books: int,
    bnc_source_dir: Path | None = None,
) -> list[dict]:
    entries: list[dict] = []
    for source in sources:
        if source == "gutenberg":
            entries.append(
                download_gutenberg_books(
                    corpus_dir=corpus_dir,
                    language=language,
                    max_books=max_books,
                )
            )
        elif source == "oanc":
            entries.append(download_oanc_texts(corpus_dir=corpus_dir, language=language))
        elif source == "masc":
            entries.append(download_masc_texts(corpus_dir=corpus_dir, language=language))
        elif source == "bnc":
            if bnc_source_dir is None:
                raise ValueError("--bnc-source-dir is required when --source bnc is selected")
            entries.append(
                import_bnc_texts(
                    corpus_dir=corpus_dir,
                    source_dir=bnc_source_dir,
                    language=language,
                )
            )
        else:
            raise ValueError(f"Unsupported source {source!r}")
    return entries


def _build_sources_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source",
        dest="sources",
        action="append",
        help=(
            "Corpus source(s) to include. Repeat to mix sources. "
            "English: gutenberg, oanc, masc, bnc. Other languages: gutenberg."
        ),
    )
    parser.add_argument(
        "--bnc-source-dir",
        type=Path,
        default=None,
        help="Path to a licensed local BNC checkout when using --source bnc.",
    )


def _resolve_sources(language: str, sources: list[str] | None) -> list[str]:
    lang = (language or "en").strip().lower()
    requested = sources or DEFAULT_SOURCES_BY_LANGUAGE[lang]
    allowed = set(SOURCE_CHOICES_BY_LANGUAGE[lang])
    invalid = [source for source in requested if source not in allowed]
    if invalid:
        raise ValueError(
            f"Unsupported source(s) for language={lang!r}: {', '.join(invalid)}. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )
    return requested


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.corpus",
        description=(
            "Build Zenith-compatible 5-gram models from local and downloaded corpora. "
            "English supports Gutenberg, OANC, MASC, and licensed local BNC imports; "
            "de/fr/it/la currently support Gutenberg."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download")
    download.add_argument("language", choices=SUPPORTED_LANGUAGES)
    download.add_argument("--output-dir", type=Path, default=None)
    download.add_argument("--max-books", type=int, default=100)
    _build_sources_arg(download)

    build = sub.add_parser("build")
    build.add_argument("language", choices=SUPPORTED_LANGUAGES)
    build.add_argument("--corpus-dir", type=Path, required=True)
    build.add_argument("--output", type=Path, default=None)
    _build_v2_args(build)

    verify = sub.add_parser("verify")
    verify.add_argument("model_path", type=Path)

    run = sub.add_parser("run")
    run.add_argument("language", choices=SUPPORTED_LANGUAGES)
    run.add_argument("--output", type=Path, default=None)
    run.add_argument("--corpus-dir", type=Path, default=None)
    run.add_argument("--max-books", type=int, default=100)
    _build_sources_arg(run)
    _build_v2_args(run)

    return parser


def _build_v2_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--alphabet",
        type=str,
        default=None,
        help=(
            "Emit a zenith_binary_v2 model over this ordered symbol alphabet "
            "instead of the default 26-letter v1 model. Symbol i maps to "
            "alphabet[i]; normalization keeps only lowercased characters in the "
            "alphabet."
        ),
    )
    parser.add_argument(
        "--order",
        type=int,
        default=None,
        help="N-gram order for a --alphabet (v2) build. Defaults to 5.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help=(
            "Optional variant slug written to the model sidecar (e.g. "
            "'historical_1600_1899'). Used by the model registry to select "
            "among per-language models."
        ),
    )
    parser.add_argument(
        "--display-label",
        type=str,
        default=None,
        help="Optional human-readable label written to the model sidecar.",
    )


def _validate_v2_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate the v2 (--alphabet/--order) option combination for build/run."""
    alphabet = getattr(args, "alphabet", None)
    order = getattr(args, "order", None)
    if order is not None and alphabet is None:
        parser.error("--order only applies to v2 builds; pass --alphabet as well")
    if alphabet is not None and args.output is None:
        parser.error(
            "--alphabet (zenith_binary_v2) requires an explicit --output so a v2 "
            "build cannot clobber the tracked v1 models under models/"
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "download":
        output_dir = args.output_dir or _default_corpus_dir(args.language)
        sources = _resolve_sources(args.language, args.sources)
        entries = _download_sources(
            language=args.language,
            corpus_dir=output_dir,
            sources=sources,
            max_books=args.max_books,
            bnc_source_dir=args.bnc_source_dir,
        )
        print(f"Downloaded {len(entries)} source(s) to {output_dir}")
        for entry in entries:
            print(f"  - {entry['name']}: {entry['text_files']} text files")
        return 0

    if args.command == "build":
        _validate_v2_args(parser, args)
        output = args.output or _default_output(args.language)
        manifest = load_manifest(args.corpus_dir)
        stats = build_model(
            language=args.language,
            corpus_dir=args.corpus_dir,
            output_path=output,
            sources=manifest.get("sources") or None,
            alphabet=args.alphabet,
            order=args.order,
            variant=args.variant,
            display_label=args.display_label,
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
        _validate_v2_args(parser, args)
        corpus_dir = args.corpus_dir or _default_corpus_dir(args.language)
        output = args.output or _default_output(args.language)
        sources = _resolve_sources(args.language, args.sources)
        entries = _download_sources(
            language=args.language,
            corpus_dir=corpus_dir,
            sources=sources,
            max_books=args.max_books,
            bnc_source_dir=args.bnc_source_dir,
        )
        manifest = load_manifest(corpus_dir)
        stats = build_model(
            language=args.language,
            corpus_dir=corpus_dir,
            output_path=output,
            sources=manifest.get("sources") or None,
            alphabet=args.alphabet,
            order=args.order,
            variant=args.variant,
            display_label=args.display_label,
        )
        verify_model(output)
        print(f"Built and verified {output}")
        print(f"Metadata: {stats.metadata_path}")
        print(f"Sources: {json.dumps([entry['name'] for entry in entries])}")
        return 0

    parser.error(f"unsupported command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
