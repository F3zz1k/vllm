# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire format between the shared KV pool server and its clients.

Standard library only: the server runs outside the vLLM environment.

Each message is a 4-byte big-endian length followed by a ``marshal`` payload.
A request is ``(op, args)`` and a reply is ``(ok, value)``, where ``value``
is the result when ``ok`` is true and an error string otherwise. Payloads
hold only bytes, str, int, bool, None, lists, tuples and dicts, which
``marshal`` round-trips exactly. The socket is local and its permissions
restrict it to the owner, so no further framing or authentication is used.
"""

import marshal
import socket
import struct

PROTOCOL_VERSION = 1

_LEN = struct.Struct(">I")
_MAX_MESSAGE = 256 << 20

# Lookup results, mirrored from vllm.v1.kv_offload.base.LookupResult.
MISS = 0
HIT = 1
HIT_PENDING = 2
RETRY = 3

# Operations.
OP_ATTACH = "attach"
OP_INFO = "info"
OP_LOOKUP = "lookup"
OP_LOOKUP_PREFIX = "lookup_prefix"  # stops after the first MISS
OP_END_SESSION = "end_session"
OP_PREPARE_LOAD = "prepare_load"
OP_COMPLETE_LOAD = "complete_load"
OP_PREPARE_STORE = "prepare_store"
# (keys, tag): exclusive-mode store of one conversation ("run"); added 2026-10-23.
OP_PREPARE_STORE_TAGGED = "prepare_store_tagged"
OP_COMPLETE_STORE = "complete_store"
OP_TOUCH = "touch"
OP_RELEASE_ALL = "release_all"
OP_STATS = "stats"
# Operator calls (no attach): {"pool": name, ...}.
OP_PIN = "pin"
OP_PIN_RECENT = "pin_recent"
OP_UNPIN = "unpin"


class ProtocolError(Exception):
    pass


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        part = sock.recv(n - len(buf))
        if not part:
            raise ConnectionError("shared KV pool peer closed the connection")
        buf += part
    return bytes(buf)


def send_message(sock: socket.socket, obj: object) -> None:
    payload = marshal.dumps(obj)
    if len(payload) > _MAX_MESSAGE:
        raise ProtocolError(f"message of {len(payload)} bytes exceeds the limit")
    sock.sendall(_LEN.pack(len(payload)) + payload)


def recv_message(sock: socket.socket) -> object:
    (length,) = _LEN.unpack(_recv_exact(sock, _LEN.size))
    if length > _MAX_MESSAGE:
        raise ProtocolError(f"message of {length} bytes exceeds the limit")
    return marshal.loads(_recv_exact(sock, length))


def encode(obj: object) -> bytes:
    payload = marshal.dumps(obj)
    return _LEN.pack(len(payload)) + payload


def decode_frames(buf: bytearray) -> list[object]:
    """Pop every complete frame from ``buf`` and return the decoded objects."""
    out = []
    while len(buf) >= _LEN.size:
        (length,) = _LEN.unpack_from(buf, 0)
        if length > _MAX_MESSAGE:
            raise ProtocolError(f"message of {length} bytes exceeds the limit")
        end = _LEN.size + length
        if len(buf) < end:
            break
        out.append(marshal.loads(bytes(buf[_LEN.size : end])))
        del buf[:end]
    return out
