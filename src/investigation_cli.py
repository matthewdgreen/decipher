"""``decipher investigation`` — the structured investigation CLI (milestone I-1).

A second thin transport skin over the transport-neutral
:class:`~investigation_service.service.InvestigationService`, mirroring the MCP
server (spec ``docs/specs/investigation_cli_spec.md`` §6, sub-spec
``docs/specs/investigation_cli_i1_impl_spec.md``). This module owns only
argv↔dict mapping and process exit-code classification; it adds ZERO schema or
validation logic — the shared service validates every argument object against
the operation manifest.

I-1 scope: the NINE read-class verbs (auto-registered from the operation
manifest by ``cli_verb``), the canonical ``--input-json`` / ``--input-file``
input path, and the ``call OPERATION`` escape hatch restricted to read-class
operations. Mutations, per-invocation leases, ``--revision``, the exhaustive
exit matrix, and verify/external flags all land in later milestones.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from investigation_service import manifest
from investigation_service.service import InvestigationService, LeasePolicy
from mcp_server.registry import InvestigationRegistry, default_registry_dir


# --------------------------------------------------------------------------- #
# Post-parse CLI input errors (unparseable JSON, unreadable file, input-mode   #
# conflicts, the `call` class/name checks). All map to exit 2 via the shared   #
# result table; they never reach the service.                                  #
# --------------------------------------------------------------------------- #
class _CliInputError(Exception):
    """A transport-layer input error carrying its own JSON `reason`/`detail`."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


# Reasons the CLI itself emits (never produced by the service). Kept alongside
# the service's `invalid_arguments` in the one shared exit table below.
_CLI_INPUT_REASONS = frozenset(
    {"invalid_cli_arguments", "operation_not_yet_available", "unknown_operation"}
)


def result_to_exit_code(result: dict) -> int:
    """The single shared status→exit-code table (extended in I-2).

    I-1 classes:
      - non-error result (`status` != "error")                     -> 0
      - `invalid_arguments` (service) / CLI input reasons          -> 2
      - `internal_error`                                           -> 5
      - any other domain error result                             -> 1

    Exit codes 3 (blocked) and 4 (revision conflict) are reserved for I-2's
    mutating verbs; no I-1 read produces them, so they are not classified here.
    """
    if result.get("status") != "error":
        return 0
    reason = result.get("reason")
    if reason == "invalid_arguments" or reason in _CLI_INPUT_REASONS:
        return 2
    if reason == "internal_error":
        return 5
    return 1


# --------------------------------------------------------------------------- #
# Manifest-driven verb registration                                            #
# --------------------------------------------------------------------------- #
_SCALAR_TYPES = {"string": str, "integer": int, "number": float}


def _kebab(name: str) -> str:
    return name.replace("_", "-")


def _prop_kind(prop_schema: dict) -> str:
    """Classify a schema property into an argparse strategy.

    Returns one of: ``"scalar"`` (string/integer/number, incl. enums),
    ``"bool"`` (boolean), or ``"json"`` (array/object/union — routed through a
    ``--<name>-json`` flag parsed with ``json.loads``).
    """
    ptype = prop_schema.get("type")
    if isinstance(ptype, list):
        # Union type -> lossless JSON flag.
        return "json"
    if ptype == "boolean":
        return "bool"
    if ptype in _SCALAR_TYPES or (ptype is None and "enum" in prop_schema):
        return "scalar"
    if ptype in ("array", "object"):
        return "json"
    # Anything unrecognized routes losslessly through JSON rather than guessing.
    return "json"


def _add_operation_arguments(vp: argparse.ArgumentParser, op: manifest.OperationSpec) -> None:
    """Register one verb's argparse surface derived from its schema.

    Nothing here is argparse-``required``: the positional ID is optional
    (``nargs="?"``) and every flag defaults to ``None`` so the object can be
    supplied via ``--input-json`` / ``--input-file`` instead. Required-ness is
    enforced by ``InvestigationService.dispatch`` (schema validation), keeping
    the CLI free of schema logic.

    Records ``_operation_args`` on the parser defaults: a list of
    ``(prop_name, dest, kind)`` tuples the runner walks to build the object.
    """
    schema = op.schema
    props: dict = schema.get("properties", {})
    required = set(schema.get("required") or [])
    meta: list[tuple[str, str, str]] = []

    # investigation_id becomes the positional ID when the schema requires it.
    if "investigation_id" in props and "investigation_id" in required:
        vp.add_argument("investigation_id", nargs="?", metavar="ID",
                        help="Investigation id (from `investigation list`).")
        meta.append(("investigation_id", "investigation_id", "scalar"))

    for prop, pschema in props.items():
        if prop == "investigation_id":
            continue
        kind = _prop_kind(pschema)
        if kind == "json":
            flag = f"--{_kebab(prop)}-json"
            dest = f"arg_{prop}"
            vp.add_argument(flag, dest=dest, default=None,
                            help=f"JSON value for `{prop}` (array/object).")
            meta.append((prop, dest, "json"))
        elif kind == "bool":
            flag = f"--{_kebab(prop)}"
            dest = f"arg_{prop}"
            vp.add_argument(flag, dest=dest, default=None,
                            action=argparse.BooleanOptionalAction,
                            help=f"Boolean `{prop}` (omit to leave unset).")
            meta.append((prop, dest, "bool"))
        else:  # scalar
            flag = f"--{_kebab(prop)}"
            dest = f"arg_{prop}"
            argtype = _SCALAR_TYPES.get(pschema.get("type"), str)
            kwargs: dict = {"dest": dest, "default": None, "type": argtype}
            enum = pschema.get("enum")
            if enum:
                kwargs["choices"] = enum
            vp.add_argument(flag, help=f"Value for `{prop}`.", **kwargs)
            meta.append((prop, dest, "scalar"))

    vp.add_argument("--input-json", dest="input_json", default=None, metavar="JSON",
                    help="Whole argument object as one JSON object.")
    vp.add_argument("--input-file", dest="input_file", default=None, metavar="PATH",
                    help="Read the argument object from a JSON file ('-' = stdin).")
    vp.set_defaults(_operation=op.name, _is_call=False, _operation_args=meta)


def add_investigation_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register `decipher investigation` and its verb tree from the manifest.

    One friendly verb per read-class :class:`OperationSpec` (auto-registered by
    ``cli_verb`` — no hand-written verb list), plus the reserved ``call``
    escape hatch. Global transport options precede the verb.
    """
    inv = subparsers.add_parser(
        "investigation",
        help="Structured investigation CLI (JSON in/out over the shared service)",
        description=(
            "Machine-readable access to the investigation state machine. Every "
            "invocation loads current code and prints exactly one JSON object."
        ),
    )
    inv.add_argument(
        "--registry-dir", dest="registry_dir", default=None, metavar="DIR",
        help="Investigation registry directory "
             "(default: $DECIPHER_MCP_REGISTRY or ~/.config/decipher/investigations).",
    )
    verbs = inv.add_subparsers(dest="investigation_verb", metavar="VERB", required=True)

    for op in manifest.OPERATIONS:
        if op.operation_class != "read":
            continue
        vp = verbs.add_parser(op.cli_verb, help=op.description.split(". ")[0])
        _add_operation_arguments(vp, op)

    # Reserved transport verb: dispatch any operation by canonical name via JSON.
    cp = verbs.add_parser(
        "call",
        help="Dispatch a manifest operation by canonical name (JSON input only).",
    )
    cp.add_argument("operation", metavar="OPERATION",
                    help="Canonical MCP operation name.")
    cp.add_argument("--input-json", dest="input_json", default=None, metavar="JSON",
                    help="Argument object as one JSON object.")
    cp.add_argument("--input-file", dest="input_file", default=None, metavar="PATH",
                    help="Read the argument object from a JSON file ('-' = stdin).")
    cp.set_defaults(_operation=None, _is_call=True)


# --------------------------------------------------------------------------- #
# Input intake                                                                 #
# --------------------------------------------------------------------------- #
def _parse_json_object(text: str, source: str) -> dict:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _CliInputError(
            "invalid_cli_arguments", f"{source} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(obj, dict):
        raise _CliInputError(
            "invalid_cli_arguments", f"{source} must be a JSON object, got {type(obj).__name__}"
        )
    return obj


def _read_input_file(path: str) -> str:
    if path == "-":
        try:
            return sys.stdin.read()
        except (OSError, UnicodeDecodeError) as exc:
            raise _CliInputError(
                "invalid_cli_arguments", f"could not read stdin: {exc}"
            ) from exc
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise _CliInputError(
            "invalid_cli_arguments", f"could not read --input-file {path!r}: {exc}"
        ) from exc


def _build_arguments(args: argparse.Namespace) -> tuple[str, dict]:
    """Resolve the (operation_name, argument_object) pair for a friendly verb.

    Enforces the input-mode contract (sub-spec §2): at most one of
    {friendly flags, --input-json, --input-file}, with the sole exception that
    the positional ID may combine with a JSON object that does not itself carry
    ``investigation_id``. All violations raise :class:`_CliInputError` (exit 2).
    """
    operation = args._operation
    meta: list[tuple[str, str, str]] = args._operation_args

    json_text = args.input_json
    file_path = args.input_file
    json_present = json_text is not None
    file_present = file_path is not None

    # Which friendly flags were explicitly provided?
    id_present = False
    non_id_present = False
    for prop, dest, _kind in meta:
        if getattr(args, dest, None) is None:
            continue
        if prop == "investigation_id":
            id_present = True
        else:
            non_id_present = True

    if json_present and file_present:
        raise _CliInputError(
            "invalid_cli_arguments",
            "supply only one of --input-json / --input-file, not both",
        )
    if non_id_present and (json_present or file_present):
        raise _CliInputError(
            "invalid_cli_arguments",
            "friendly flags cannot be combined with --input-json/--input-file",
        )

    if json_present or file_present:
        raw = json_text if json_present else _read_input_file(file_path)
        source = "--input-json" if json_present else "--input-file"
        obj = _parse_json_object(raw, source)
        if id_present:
            if "investigation_id" in obj:
                raise _CliInputError(
                    "invalid_cli_arguments",
                    "investigation_id given as both the positional ID and in the "
                    "JSON object",
                )
            obj["investigation_id"] = getattr(args, "investigation_id")
        return operation, obj

    # Friendly-flag mode: build the object from the provided flags.
    obj = {}
    for prop, dest, kind in meta:
        val = getattr(args, dest, None)
        if val is None:
            continue
        if kind == "json":
            obj[prop] = _parse_json_object_value(val, f"--{_kebab(prop)}-json")
        else:
            obj[prop] = val
    return operation, obj


def _parse_json_object_value(text: str, source: str):
    """Parse a ``--<name>-json`` value (any JSON value, not only objects)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise _CliInputError(
            "invalid_cli_arguments", f"{source} is not valid JSON: {exc}"
        ) from exc


def _build_call_arguments(args: argparse.Namespace) -> tuple[str, dict]:
    """Resolve (operation_name, argument_object) for the `call` escape hatch.

    Enforces the read-only + known-operation guard BEFORE any registry/service
    construction (sub-spec §2), so a non-read `call` never touches the registry.
    """
    operation = args.operation
    op = manifest.operation_for(operation)
    if op is None:
        raise _CliInputError(
            "unknown_operation",
            f"no operation named {operation!r} in the manifest",
        )
    if op.operation_class != "read":
        raise _CliInputError(
            "operation_not_yet_available",
            "mutating operations arrive with milestone I-2 "
            "(invocation-held lease semantics)",
        )

    json_present = args.input_json is not None
    file_present = args.input_file is not None
    if json_present and file_present:
        raise _CliInputError(
            "invalid_cli_arguments",
            "supply only one of --input-json / --input-file, not both",
        )
    if json_present:
        obj = _parse_json_object(args.input_json, "--input-json")
    elif file_present:
        obj = _parse_json_object(_read_input_file(args.input_file), "--input-file")
    else:
        obj = {}
    return operation, obj


# --------------------------------------------------------------------------- #
# Service construction + dispatch                                              #
# --------------------------------------------------------------------------- #
def _resolve_registry_dir(args: argparse.Namespace) -> Path:
    """Explicit flag → $DECIPHER_MCP_REGISTRY → config default (reuse helper)."""
    if getattr(args, "registry_dir", None):
        return Path(args.registry_dir).expanduser()
    return default_registry_dir()


def _dispatch(args: argparse.Namespace) -> dict:
    """Resolve arguments, construct the service lazily, and dispatch.

    The `call` class/name guard runs before service construction. Reads never
    acquire the writer lease and never verify (no provider configured).
    """
    if getattr(args, "_is_call", False):
        operation, obj = _build_call_arguments(args)
    else:
        operation, obj = _build_arguments(args)

    registry = InvestigationRegistry(_resolve_registry_dir(args))
    service = InvestigationService(
        registry=registry,
        client_name="cli",
        lease_policy=LeasePolicy.SESSION_HELD,
    )
    return service.dispatch(operation, obj)


def run_investigation_command(args: argparse.Namespace) -> int:
    """Run one `decipher investigation` invocation; return the process exit code.

    Prints exactly one JSON object (+ ``\\n``) to stdout — byte-identical to the
    MCP server's tool-result body for reads — and classifies the result via the
    shared :func:`result_to_exit_code` table. Diagnostics go to stderr only.
    """
    try:
        result = _dispatch(args)
    except _CliInputError as exc:
        result = {"status": "error", "reason": exc.reason, "detail": exc.detail}
    except Exception:  # noqa: BLE001 - any unexpected crash is an internal error
        if os.environ.get("DECIPHER_CLI_DEBUG") == "1":
            traceback.print_exc(file=sys.stderr)
        result = {"status": "error", "reason": "internal_error"}

    sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result_to_exit_code(result)
