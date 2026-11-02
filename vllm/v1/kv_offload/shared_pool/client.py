# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Blocking client for the shared KV pool server."""

import os
import socket

from vllm.v1.kv_offload.shared_pool import protocol as P

# TODO(kvpool protocol contract v2): these belong in protocol.py, which the
# server side owns; integration moves them there and drops this block.
PROTOCOL_VERSION = 2
OP_MERGE_TAG = "merge_tag"  # (src_tag, dst_tag)
OP_STATE = "state"  # no attach needed

# How long an engine waits for the server to answer ATTACH or INFO. The server
# defers both until its arena is ready (created, populated, spill restored).
ATTACH_TIMEOUT_SEC = float(os.environ.get("VLLM_KVPOOL_ATTACH_TIMEOUT_SEC", "900"))

# Every way a pool can be out of reach: no socket, refused, closed, timed out,
# or a malformed reply.
UNREACHABLE = (OSError, EOFError, ValueError, P.ProtocolError)


class PoolServerError(RuntimeError):
    """The server rejected a request (a protocol or state error)."""


class PoolClient:
    def __init__(self, socket_path: str, timeout: float = 120.0) -> None:
        self.socket_path = socket_path
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(timeout)
        try:
            self._sock.connect(socket_path)
        except OSError:
            self._sock.close()
            raise

    def settimeout(self, timeout: float) -> None:
        self._sock.settimeout(timeout)

    def call(self, op: str, args: object = None) -> object:
        P.send_message(self._sock, (op, args))
        reply = P.recv_message(self._sock)
        ok, value = reply  # type: ignore[misc]
        if not ok:
            raise PoolServerError(f"{op}: {value}")
        return value

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


def pool_info(
    socket_path: str, args: dict, timeout: float = ATTACH_TIMEOUT_SEC
) -> dict:
    """One-shot query of a pool's arena. The server defers the reply until the
    arena is ready, hence the long default timeout."""
    client = PoolClient(socket_path, timeout=timeout)
    try:
        return client.call(P.OP_INFO, args)  # type: ignore[return-value]
    finally:
        client.close()
