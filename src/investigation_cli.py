"""``decipher investigation`` — the structured investigation CLI (milestone I-1).

A second thin transport skin over the transport-neutral
:class:`~investigation_service.service.InvestigationService`, mirroring the MCP
server (spec ``docs/specs/investigation_cli_spec.md`` §6, sub-spec
``docs/specs/investigation_cli_i1_impl_spec.md``). This module owns only
argv↔dict mapping and process exit-code classification; it adds ZERO schema or
validation logic — the shared service validates every argument object against
the operation manifest.

I-2 scope: the create + mutate verbs join the I-1 reads (all auto-registered
from the manifest by ``cli_verb``), ``--revision`` aliases ``expected_revision``,
``start`` gains ``--ciphertext``/``--ciphertext-file`` (``-`` = stdin RAW
ciphertext) source rules, dispatch runs under the ``INVOCATION_HELD`` lease
policy (acquire→commit→release per invocation), and the shared exit table gains
3 (blocked) and 4 (conflict). Three ops keep a typed
``operation_not_yet_available`` error: ``experiment_submit``/
``experiment_collect`` (I-3) and ``request_independent_verification`` (I-5).
Verify/external flags land in I-5.
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

# Manifest ops registered as friendly verbs but NOT yet dispatchable in I-2:
# their canonical name maps to the typed `operation_not_yet_available` error
# (exit 2) naming the milestone that lands them. The verbs still PARSE and
# `call` still knows the names; only dispatch is short-circuited BEFORE any
# registry/service construction (so the registry is untouched — sub-spec §0/§4).
_EXCLUDED_OPS: dict[str, str] = {
    "experiment_submit": (
        "experiments on the CLI (the two-commit lifecycle, --wait/--detach, and "
        "crash reconciliation) arrive with milestone I-3"
    ),
    "experiment_collect": (
        "experiments on the CLI (the two-commit lifecycle, --wait/--detach, and "
        "crash reconciliation) arrive with milestone I-3"
    ),
    "request_independent_verification": (
        "verification and every external-call path (--verify-provider / "
        "--allow-external) arrive with milestone I-5"
    ),
}


def result_to_exit_code(result: dict) -> int:
    """The single shared status→exit-code table, exhaustive over service classes.

    Service status classes and their exit codes (spec §3.1 output/exit contract):
      - non-error/blocked/conflict result (`status` absent or "ok"/…)  -> 0
      - `status == "blocked"` (investigation_terminal, writer_lease_held,
        gate refusals such as attestation_required)                    -> 3
      - `status == "conflict"` (revision_mismatch)                     -> 4
      - `status == "error"` with:
          * `invalid_arguments` (schema) / `invalid_investigation_id` /
            CLI input reasons (parse/mode/unknown/not-yet-available)   -> 2
          * `internal_error`                                           -> 5
          * any other domain/lookup/unavailable reason                 -> 1

    The `reason` field is always the machine-readable identifier; the exit code
    only classifies. Totality: an unknown reason UNDER `status == "error"` falls
    through to 1 (reason preserved); any non-error/blocked/conflict status —
    including domain statuses like "active"/"unsolved" in read bodies — exits 0.
    """
    status = result.get("status")
    if status == "blocked":
        return 3
    if status == "conflict":
        return 4
    if status != "error":
        return 0
    reason = result.get("reason")
    if (
        reason == "invalid_arguments"
        or reason == "invalid_investigation_id"
        or reason in _CLI_INPUT_REASONS
    ):
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
        if prop == "expected_revision":
            # `--revision R` is the friendly alias for expected_revision on every
            # mutating verb (spec §3.1.4 / sub-spec §3.2). Required-ness stays
            # service-enforced: a missing revision is the service's
            # invalid_arguments (exit 2), never an argparse error.
            dest = "arg_expected_revision"
            vp.add_argument(
                "--revision", dest=dest, default=None, type=int, metavar="R",
                help="Expected revision from `investigation status` "
                     "(alias for expected_revision; a mismatch is a conflict).",
            )
            meta.append(("expected_revision", dest, "scalar"))
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

    # One friendly verb per manifest operation (read + create + mutate). The
    # three not-yet-dispatchable ops (experiment_submit/experiment_collect ->
    # I-3, request_independent_verification -> I-5) are still registered; only
    # their dispatch is short-circuited to the typed error (see _EXCLUDED_OPS).
    for op in manifest.OPERATIONS:
        vp = verbs.add_parser(op.cli_verb, help=op.description.split(". ")[0])
        _add_operation_arguments(vp, op)
        if op.name == "investigation_start":
            # `start` gains the transport-only ciphertext-file alias (spec §3.1.5
            # / sub-spec §3.3): reads RAW ciphertext bytes (UTF-8), NOT a JSON
            # object, and fills the schema's `ciphertext` property. Mutually
            # exclusive with --ciphertext and JSON-object input.
            vp.add_argument(
                "--ciphertext-file", dest="ciphertext_file", default=None,
                metavar="PATH",
                help="Read RAW ciphertext (UTF-8) from a file ('-' = stdin); "
                     "fills the ciphertext property (XOR --ciphertext / JSON input).",
            )

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


def _read_ciphertext_file(path: str) -> str:
    """Read RAW ciphertext (UTF-8) for `start --ciphertext-file` ('-' = stdin).

    NOT a JSON object — the whole file content becomes the `ciphertext` value.
    File-read/UTF-8 failures are typed CLI input errors (exit 2); ciphertext
    size/format failures remain domain results from the service (sub-spec §3.3).
    """
    if path == "-":
        try:
            return sys.stdin.read()
        except (OSError, UnicodeDecodeError) as exc:
            raise _CliInputError(
                "invalid_cli_arguments", f"could not read ciphertext from stdin: {exc}"
            ) from exc
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise _CliInputError(
            "invalid_cli_arguments",
            f"could not read --ciphertext-file {path!r}: {exc}",
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

    # start-only ciphertext source alias (sub-spec §3.3): --ciphertext-file is
    # one of three mutually exclusive ciphertext sources (--ciphertext XOR
    # --ciphertext-file XOR JSON-object input). It fills the `ciphertext`
    # property; ordinary flags (--language/--label/...) may still accompany it.
    ciphertext_file = getattr(args, "ciphertext_file", None)
    if ciphertext_file is not None:
        if json_present or file_present:
            raise _CliInputError(
                "invalid_cli_arguments",
                "--ciphertext-file cannot be combined with --input-json/--input-file",
            )
        if getattr(args, "arg_ciphertext", None) is not None:
            raise _CliInputError(
                "invalid_cli_arguments",
                "supply only one of --ciphertext / --ciphertext-file",
            )
        obj = {}
        for prop, dest, kind in meta:
            val = getattr(args, dest, None)
            if val is None:
                continue
            if kind == "json":
                obj[prop] = _parse_json_object_value(val, f"--{_kebab(prop)}-json")
            else:
                obj[prop] = val
        obj["ciphertext"] = _read_ciphertext_file(ciphertext_file)
        return operation, obj

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

    Enforces only the known-operation guard here; the not-yet-available
    exclusions (§3.1) are applied centrally in :func:`run_investigation_command`
    BEFORE any registry/service construction, so `call` on an excluded op
    returns the same typed error and never touches the registry. In I-2 `call`
    dispatches any operation EXCEPT the three exclusions.
    """
    operation = args.operation
    op = manifest.operation_for(operation)
    if op is None:
        raise _CliInputError(
            "unknown_operation",
            f"no operation named {operation!r} in the manifest",
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


def run_investigation_command(args: argparse.Namespace) -> int:
    """Run one `decipher investigation` invocation; return the process exit code.

    Prints exactly one JSON object (+ ``\\n``) to stdout — byte-identical to the
    MCP server's tool-result body — and classifies the result via the shared
    :func:`result_to_exit_code` table. Diagnostics go to stderr only.

    Lease lifecycle (I-2): the service runs under ``INVOCATION_HELD`` — a
    mutating dispatch acquires the writer lease and releases it in the service's
    own ``finally`` after the commit. The CLI-level ``finally`` here calls
    ``service.shutdown()`` as best-effort finalization so a SIGINT mid-verb still
    releases any lease before the conventional signal exit (KeyboardInterrupt is
    not caught, so it propagates to the conventional 130 exit after cleanup).
    """
    service = None
    try:
        if getattr(args, "_is_call", False):
            operation, obj = _build_call_arguments(args)
        else:
            operation, obj = _build_arguments(args)
        # Not-yet-dispatchable ops (§3.1) short-circuit BEFORE any registry or
        # service construction, so the registry stays untouched.
        if operation in _EXCLUDED_OPS:
            raise _CliInputError("operation_not_yet_available", _EXCLUDED_OPS[operation])
        registry = InvestigationRegistry(_resolve_registry_dir(args))
        service = InvestigationService(
            registry=registry,
            client_name="cli",
            lease_policy=LeasePolicy.INVOCATION_HELD,
        )
        result = service.dispatch(operation, obj)
    except _CliInputError as exc:
        result = {"status": "error", "reason": exc.reason, "detail": exc.detail}
    except Exception:  # noqa: BLE001 - any unexpected crash is an internal error
        if os.environ.get("DECIPHER_CLI_DEBUG") == "1":
            traceback.print_exc(file=sys.stderr)
        result = {"status": "error", "reason": "internal_error"}
    finally:
        if service is not None:
            try:
                service.shutdown()
            except Exception:  # noqa: BLE001 - finalization is best-effort
                pass

    sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
    return result_to_exit_code(result)
