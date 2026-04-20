#!/usr/bin/env python3
"""Wrapper for Zenith's local GraphQL/WebSocket API.

The Zenith release ships as a Spring Boot UI/server jar rather than a batch
CLI. This wrapper starts the server on a temporary localhost port, subscribes
to solution updates over the GraphQL WebSocket protocol, starts a solve via the
HTTP GraphQL mutation, then writes the best/final plaintext it observes.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import socket
import struct
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


SUBSCRIPTION = """
subscription($requestId: ID!) {
  solutionUpdates(requestId: $requestId) {
    type
    epochData { epochsCompleted epochsTotal }
    solutionData { plaintext scores }
  }
}
"""

MUTATION = """
mutation($input: SolutionRequest!) {
  solveSolution(input: $input)
}
"""

CONFIG_QUERY = "{ configuration { epochs } }"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Zenith through its local GraphQL API.")
    parser.add_argument("--java", default="java")
    parser.add_argument("--jar", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--sampler-iterations", type=int, default=5000)
    parser.add_argument("--temperature-min", type=float, default=0.006)
    parser.add_argument("--temperature-max", type=float, default=0.012)
    parser.add_argument("--fitness-function", default="NgramAndIndexOfCoincidence")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--max-heap", default="2G")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()

    input_file = Path(args.input).resolve()
    output_file = Path(args.output).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    jar = Path(args.jar).resolve()
    port = args.port or _free_port()
    request_id = f"decipher-{uuid.uuid4().hex[:12]}"
    log_file = output_file.parent / "zenith_server.log"
    server = None

    try:
        with log_file.open("w", encoding="utf-8") as log:
            server = subprocess.Popen(
                [
                    args.java,
                    f"-Xmx{args.max_heap}",
                    "-jar",
                    str(jar),
                    f"--server.port={port}",
                ],
                cwd=jar.parent,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )

            endpoint = f"http://127.0.0.1:{port}/graphql"
            _wait_for_server(endpoint, server, timeout=90)

            ws = _GraphQLWebSocket("127.0.0.1", port, "/graphql")
            ws.connect()
            ws.send({"type": "connection_init"})
            _wait_for_type(ws, "connection_ack", timeout=20)
            ws.send({
                "id": "1",
                "type": "subscribe",
                "payload": {
                    "query": SUBSCRIPTION,
                    "variables": {"requestId": request_id},
                },
            })

            ciphertext = list(input_file.read_text(encoding="utf-8").strip())
            variables = {
                "input": {
                    "requestId": request_id,
                    "rows": 1,
                    "columns": len(ciphertext),
                    "ciphertext": ciphertext,
                    "epochs": args.epochs,
                    "fitnessFunction": {"name": args.fitness_function},
                    "simulatedAnnealingConfiguration": {
                        "samplerIterations": args.sampler_iterations,
                        "annealingTemperatureMin": args.temperature_min,
                        "annealingTemperatureMax": args.temperature_max,
                    },
                }
            }
            _graphql_post(endpoint, MUTATION, variables)

            plaintext = _collect_solution(ws, timeout=args.timeout_seconds)
            ws.close()

        if plaintext:
            normalized = f"SOLUTION: {plaintext}\n"
            output_file.write_text(normalized, encoding="utf-8")
            print(normalized, end="")
            return 0

        msg = "No Zenith solution update was received before timeout.\n"
        output_file.write_text(msg, encoding="utf-8")
        print(msg, end="")
        return 1
    finally:
        if server and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()


def _wait_for_server(endpoint: str, proc: subprocess.Popen, timeout: int) -> None:
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Zenith exited during startup with code {proc.returncode}")
        try:
            _graphql_post(endpoint, CONFIG_QUERY, {})
            return
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            time.sleep(0.5)
    raise TimeoutError(f"Zenith did not become ready within {timeout}s: {last_error}")


def _graphql_post(endpoint: str, query: str, variables: dict) -> dict:
    body = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    request = Request(
        endpoint,
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310 - localhost wrapper
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("errors"):
        raise RuntimeError(json.dumps(payload["errors"], ensure_ascii=False))
    return payload


def _collect_solution(ws: "_GraphQLWebSocket", timeout: int) -> str:
    deadline = time.time() + timeout
    best = ""
    while time.time() < deadline:
        msg = ws.recv(timeout=max(0.5, min(5.0, deadline - time.time())))
        if not msg:
            continue
        msg_type = msg.get("type")
        if msg_type == "next":
            update = (((msg.get("payload") or {}).get("data") or {}).get("solutionUpdates") or {})
            solution = update.get("solutionData") or {}
            plaintext = _clean_plaintext(solution.get("plaintext", ""))
            if plaintext:
                best = plaintext
            if update.get("type") == "SOLUTION" and plaintext:
                return plaintext
        elif msg_type == "complete":
            return best
        elif msg_type == "error":
            raise RuntimeError(json.dumps(msg, ensure_ascii=False))
    return best


def _wait_for_type(ws: "_GraphQLWebSocket", expected: str, timeout: int) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = ws.recv(timeout=max(0.5, min(5.0, deadline - time.time())))
        if msg.get("type") == expected:
            return
        if msg.get("type") == "error":
            raise RuntimeError(json.dumps(msg, ensure_ascii=False))
    raise TimeoutError(f"Timed out waiting for WebSocket message {expected!r}")


def _clean_plaintext(text: str) -> str:
    return "".join(ch for ch in text.upper() if "A" <= ch <= "Z")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _GraphQLWebSocket:
    def __init__(self, host: str, port: int, path: str) -> None:
        self.host = host
        self.port = port
        self.path = path
        self.sock: socket.socket | None = None

    def connect(self) -> None:
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        sock = socket.create_connection((self.host, self.port), timeout=20)
        request = (
            f"GET {self.path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "Sec-WebSocket-Protocol: graphql-transport-ws\r\n"
            "\r\n"
        )
        sock.sendall(request.encode("ascii"))
        response = _read_http_response(sock)
        if " 101 " not in response.split("\r\n", 1)[0]:
            raise RuntimeError(f"WebSocket upgrade failed: {response[:300]}")
        self.sock = sock

    def send(self, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self._send_frame(data)

    def recv(self, timeout: float) -> dict:
        if self.sock is None:
            raise RuntimeError("WebSocket is not connected")
        self.sock.settimeout(timeout)
        try:
            opcode, data = self._recv_frame()
        except socket.timeout:
            return {}
        if opcode == 8:
            return {"type": "complete"}
        if opcode == 9:
            self._send_frame(data, opcode=10)
            return {}
        if opcode != 1:
            return {}
        return json.loads(data.decode("utf-8"))

    def close(self) -> None:
        if self.sock is not None:
            try:
                self._send_frame(b"", opcode=8)
            finally:
                self.sock.close()
                self.sock = None

    def _send_frame(self, data: bytes, opcode: int = 1) -> None:
        if self.sock is None:
            raise RuntimeError("WebSocket is not connected")
        header = bytearray([0x80 | opcode])
        length = len(data)
        if length < 126:
            header.append(0x80 | length)
        elif length <= 0xFFFF:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", length))
        mask = random.randbytes(4) if hasattr(random, "randbytes") else os.urandom(4)
        masked = bytes(byte ^ mask[i % 4] for i, byte in enumerate(data))
        self.sock.sendall(bytes(header) + mask + masked)

    def _recv_frame(self) -> tuple[int, bytes]:
        if self.sock is None:
            raise RuntimeError("WebSocket is not connected")
        first = _recv_exact(self.sock, 2)
        opcode = first[0] & 0x0F
        masked = bool(first[1] & 0x80)
        length = first[1] & 0x7F
        if length == 126:
            length = struct.unpack("!H", _recv_exact(self.sock, 2))[0]
        elif length == 127:
            length = struct.unpack("!Q", _recv_exact(self.sock, 8))[0]
        mask = _recv_exact(self.sock, 4) if masked else b""
        data = _recv_exact(self.sock, length)
        if masked:
            data = bytes(byte ^ mask[i % 4] for i, byte in enumerate(data))
        return opcode, data


def _read_http_response(sock: socket.socket) -> str:
    chunks: list[bytes] = []
    while b"\r\n\r\n" not in b"".join(chunks):
        chunks.append(sock.recv(4096))
        if not chunks[-1]:
            break
    return b"".join(chunks).decode("iso-8859-1", errors="replace")


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks: list[bytes] = []
    remaining = n
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise URLError("WebSocket closed unexpectedly")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


if __name__ == "__main__":
    raise SystemExit(main())
