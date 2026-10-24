# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Blocking client for the shared KV pool server."""

import socket

from vllm.v1.kv_offload.shared_pool import protocol as P


class PoolServerError(RuntimeError):
    """The server rejected a request (a protocol or state error)."""


class PoolClient:
    def __init__(self, socket_path: str, timeout: float = 120.0) -> None:
        self.socket_path = socket_path
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(timeout)
        self._sock.connect(socket_path)

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


def pool_info(socket_path: str, args: dict, timeout: float = 600.0) -> dict:
    """One-shot query of a pool's region (created on first use). The long
    timeout covers the server pre-faulting a new region."""
    client = PoolClient(socket_path, timeout=timeout)
    try:
        return client.call(P.OP_INFO, args)  # type: ignore[return-value]
    finally:
        client.close()
