# SPDX-License-Identifier: Apache-2.0
"""B70-PLE-RAM: the INT8 row-scale PLE table native in pinned host RAM (B70_PLE_INT8=1).

The host side (``ple_ram``, ``ple_nvme``) loads standalone; the class-level
tests import the tree's ``b70_ple_ram`` (CPU torch, no XPU) and are skipped
when vLLM does not import. B70_PLE_INT8_TEST_PATH (+ PLE_TABLE_PATH) adds a
row-exact and rel-L2 check against the real table.
"""
import importlib.util
import json
import os
import struct
import sys
import tempfile

import numpy as np
import pytest
import torch

HERE = os.path.dirname(__file__)
NVIDIA = os.path.join(HERE, "../../vllm/models/qwen4_exp/nvidia")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(NVIDIA, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ple_nvme = _load("ple_nvme")
ple_ram = _load("ple_ram")

DIM = 160
ROW = DIM + 4
QFN_ROWS = 320001536  # INT8-ple-rowscale-76427c71: U8 [320001536, 164]


def _vllm_ram():
    try:
        from vllm.models.qwen4_exp.nvidia import b70_ple_nvme, b70_ple_ram
    except Exception as exc:  # pragma: no cover - CPU venv without the tree's deps
        pytest.skip(f"vLLM tree does not import here: {exc}")
    return b70_ple_nvme, b70_ple_ram


def _quantize(x: torch.Tensor) -> np.ndarray:
    """build_int8_ple.py's formula: s = absmax/127 (fp32), q = round(x/s) clamp +-127."""
    x = x.to(torch.float32)
    s = x.abs().amax(1, keepdim=True) / 127.0
    q = (x / s.clamp_min(1e-30)).round().clamp_(-127, 127).to(torch.int8)
    packed = torch.cat([q.view(torch.uint8), s.contiguous().view(torch.uint8)], dim=1)
    return packed.numpy()


def _write(tmp, data: np.ndarray, pad: int = 37) -> str:
    hdr = json.dumps({"table": {"dtype": "U8", "shape": list(data.shape),
                                "data_offsets": [0, data.nbytes]},
                      "__metadata__": {"format": "lumnus-ple-int8-rowscale/v1"}}).encode()
    hdr += b" " * (pad + (-(len(hdr) + pad) % 8))  # data start off page alignment
    path = os.path.join(tmp, "t.safetensors")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)) + hdr + data.tobytes())
    return path


def _random_table(rows, seed=11):
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 256, size=(rows, ROW), dtype=np.uint8)
    data[:, DIM:] = np.abs(rng.standard_normal((rows, 1), dtype=np.float32)).view(np.uint8)
    return data


def _ranges(rows, tp):
    per = rows // tp
    return [(r * per, (r + 1) * per if r < tp - 1 else rows) for r in range(tp)]


# 1. owned-range math -------------------------------------------------------

@pytest.mark.parametrize("tp", [1, 2, 4])
def test_owned_ranges_and_slabs_qfn(tp):
    """The vLLM vocab shard of the QFN table for TP 1/2/4, and the slab plan over it."""
    try:
        from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"vLLM does not import: {exc}")
    per = QFN_ROWS // tp
    assert per * tp == QFN_ROWS
    total_pinned = 0
    for rank in range(tp):
        idx = VocabParallelEmbedding._get_indices(QFN_ROWS, QFN_ROWS, QFN_ROWS, QFN_ROWS, rank, tp)
        start, end = idx.org_vocab_start_index, idx.org_vocab_end_index
        assert (start, end) == (rank * per, (rank + 1) * per)
        slabs = ple_ram.plan_slabs(end - start, ROW)
        assert sum(slabs) == end - start
        for r in slabs[:-1]:  # every slab fills its power of two to within one row
            assert r * ROW <= ple_ram.MAX_SLAB_BYTES
            assert ple_ram.pow2_ceil(r * ROW) - r * ROW < ROW
        pinned = ple_ram.pinned_bytes(slabs, ROW)
        assert 0 <= pinned - (end - start) * ROW < ple_ram.TAIL_BYTES + len(slabs) * ROW
        total_pinned += pinned
    # 48.88 GiB in all, whatever the TP (vs 64 GiB for one power-of-two slab per TP4 rank).
    assert abs(total_pinned - QFN_ROWS * ROW) < tp * (ple_ram.TAIL_BYTES + 16 * ROW)
    if tp == 4:
        assert per == 80000384 and per * ROW == 13120062976
        assert ple_ram.plan_slabs(per, ROW)[:2] == [(1 << 33) // ROW, (1 << 32) // ROW]


def test_plan_slabs_edges():
    assert ple_ram.plan_slabs(0, ROW) == []
    assert ple_ram.plan_slabs(1, ROW) == [1]
    assert ple_ram.plan_slabs(6000, ROW, 1 << 16, 4096) == [399, 399, 399, 399, 399, 399, 399,
                                                          399, 399, 399, 399, 399, 399, 399,
                                                          399, 15]
    with pytest.raises(ValueError):
        ple_ram.plan_slabs(10, ROW, max_bytes=100)


# 2. load + gather, row exact ---------------------------------------------------

@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("slab,tail", [(1 << 33, 1 << 20), (1 << 16, 4096), (1 << 13, 512)])
def test_load_and_gather_row_exact(tp, slab, tail):
    rows = 6000
    with tempfile.TemporaryDirectory() as tmp:
        data = _random_table(rows)
        path = _write(tmp, data)
        start, shape, dt = ple_nvme.safetensors_tensor_location(path, "table")
        assert (shape, dt) == ([rows, ROW], "U8")
        rng = np.random.default_rng(3)
        stage_sum = np.zeros((64, 4, ROW), np.int64)
        for tp_start, tp_end in _ranges(rows, tp):
            t = ple_ram.PleRamTable(tp_start, tp_end, ROW, max_slab_bytes=slab, tail_bytes=tail)
            info = t.load(path, start, threads=3, chunk_bytes=2 * ple_nvme.PAGE)
            assert info["direct"] and info["rows"] == tp_end - tp_start
            assert np.array_equal(np.concatenate(t.slabs), data[tp_start:tp_end])
            # Direct reads of the file agree byte for byte (the boot sample check).
            store = ple_nvme.PleNvmeRowStore(path, start, ROW, rows, io_threads=2, reader="py")
            own = np.unique(rng.integers(tp_start, tp_end, 300))
            assert np.array_equal(t.rows(own), store.read_rows(own))
            store.close()
            t.scan_scales(DIM)
            with pytest.raises(IndexError):
                t.rows(np.array([tp_end]))
        # One step over all ranks: each rank fills only its rows; the byte sum (ETP
        # all-reduce, one owner per row) is the full gather.
        ids = rng.integers(0, rows, size=(64, 4))
        heads = np.arange(4)
        for tp_start, tp_end in _ranges(rows, tp):
            t = ple_ram.PleRamTable(tp_start, tp_end, ROW, max_slab_bytes=slab, tail_bytes=tail)
            t.load(path, start, threads=2)
            stage = np.zeros((64, 4, ROW), np.uint8)
            n = t.gather_into(stage, ids, heads)
            own = (ids >= tp_start) & (ids < tp_end)
            assert n == int(own.sum())
            assert np.array_equal(stage[own], data[ids[own]])
            assert not stage[~own].any()
            stage_sum += stage
        assert np.array_equal(stage_sum.astype(np.uint8), data[ids])


def test_gather_into_own_head_subset():
    """Only the rank's heads are gathered, into their own columns of the stage."""
    rows = 4096
    with tempfile.TemporaryDirectory() as tmp:
        data = _random_table(rows, seed=5)
        path = _write(tmp, data)
        start, _, _ = ple_nvme.safetensors_tensor_location(path, "table")
        t = ple_ram.PleRamTable(1024, 3072, ROW, max_slab_bytes=1 << 15, tail_bytes=2048)
        t.load(path, start)
        heads = np.array([1, 2])
        ids = np.random.default_rng(1).integers(0, rows, size=(40, 4))
        stage = np.zeros((48, 4, ROW), np.uint8)
        t.gather_into(stage, ids[:, heads], heads)
        for c, h in enumerate(heads):
            own = (ids[:, h] >= 1024) & (ids[:, h] < 3072)
            assert np.array_equal(stage[:40, h][own], data[ids[:, h][own]])
            assert not stage[:40, h][~own].any()
        assert not stage[:, [0, 3]].any() and not stage[40:].any()


def test_scan_scales_refuses_bad_scale():
    data = _random_table(3000)
    for bad in (np.float32(-1.0), np.float32(np.nan), np.float32(np.inf)):
        d = data.copy()
        d[2222, DIM:] = np.array([bad], np.float32).view(np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, d)
            start, _, _ = ple_nvme.safetensors_tensor_location(path, "table")
            t = ple_ram.PleRamTable(1000, 3000, ROW, max_slab_bytes=1 << 15, tail_bytes=2048)
            t.load(path, start)
            with pytest.raises(ValueError, match="row 2222"):
                t.scan_scales(DIM, step=97)


# 3. dequant vs the BF16 table ------------------------------------------------

def test_dequant_matches_reference_and_bf16():
    b70_ple_nvme, b70_ple_ram = _vllm_ram()
    rows = 5000
    g = torch.Generator().manual_seed(0)
    bf16 = (torch.randn(rows, DIM, generator=g) * 0.02).to(torch.bfloat16)
    bf16[17] = 0  # zero row: scale 0, dequant 0
    packed = _quantize(bf16)
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, packed)
        start, _, _ = ple_nvme.safetensors_tensor_location(path, "table")
        ids = np.random.default_rng(9).integers(0, rows, size=(256, 3))
        heads = np.arange(3)
        stage_sum = np.zeros((256, 3, ROW), np.int64)
        for tp_start, tp_end in _ranges(rows, 4):
            t = ple_ram.PleRamTable(tp_start, tp_end, ROW, max_slab_bytes=1 << 16, tail_bytes=4096)
            t.load(path, start)
            stage = np.zeros((256, 3, ROW), np.uint8)
            t.gather_into(stage, ids, heads)
            stage_sum += stage
        # The consumer's ETP reduce is an int8 byte sum (exact, one owner per row).
        reduced = torch.from_numpy(stage_sum.astype(np.uint8))
        got = b70_ple_nvme.int8_dequantize(reduced, DIM, torch.bfloat16)
        # Row-exact vs the file: q * s in fp32, one rounding.
        rows_bytes = torch.from_numpy(packed[ids])
        q = rows_bytes[..., :DIM].view(torch.int8).to(torch.float32)
        s = rows_bytes[..., DIM:].contiguous().view(torch.float32)
        assert torch.equal(got, (q * s).to(torch.bfloat16))
        assert not got[(ids == 17)].any()
        # rel-L2 vs the BF16 table (the 76427c71 table measures 0.0066).
        want = bf16[torch.from_numpy(ids)].to(torch.float32)
        rel = float(torch.linalg.vector_norm(got.float() - want) / torch.linalg.vector_norm(want))
        assert rel < 0.01, rel


def test_bf16_crosscheck_on_loaded_rows(monkeypatch):
    b70_ple_nvme, b70_ple_ram = _vllm_ram()
    rows = 6000
    g = torch.Generator().manual_seed(1)
    bf16 = (torch.randn(rows, DIM, generator=g) * 0.03).to(torch.bfloat16)
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, _quantize(bf16))
        ref = os.path.join(tmp, "ple_table.pt")
        torch.save({"table": bf16}, ref)
        start, _, _ = ple_nvme.safetensors_tensor_location(path, "table")
        t = ple_ram.PleRamTable(1500, 3000, ROW, max_slab_bytes=1 << 16, tail_bytes=4096)
        t.load(path, start)
        monkeypatch.setenv("PLE_TABLE_PATH", ref)
        read = lambda r: b70_ple_ram._crosscheck_rows(t, r, path)  # noqa: E731
        rel = b70_ple_nvme.bf16_crosscheck(read, DIM, rows, path, "test")
        assert rel is not None and rel < 0.01
        # A row-shifted table (wrong file or order) is refused.
        shifted = os.path.join(tmp, "shift.safetensors")
        os.rename(_write(tmp, np.roll(_quantize(bf16), 1, axis=0)), shifted)
        t2 = ple_ram.PleRamTable(1500, 3000, ROW, max_slab_bytes=1 << 16, tail_bytes=4096)
        t2.load(shifted, start)
        read2 = lambda r: b70_ple_ram._crosscheck_rows(t2, r, shifted)  # noqa: E731
        with pytest.raises(ValueError, match="disagrees"):
            b70_ple_nvme.bf16_crosscheck(read2, DIM, rows, shifted, "test")
        monkeypatch.setenv("B70_PLE_INT8_CROSSCHECK", "0")
        assert b70_ple_nvme.bf16_crosscheck(read2, DIM, rows, shifted, "test") is None


# 4. the class: stage fill, flags --------------------------------------------

def test_fill_stage_uses_ram_rows():
    _, b70_ple_ram = _vllm_ram()
    rows = 4096
    with tempfile.TemporaryDirectory() as tmp:
        data = _random_table(rows, seed=2)
        path = _write(tmp, data)
        start, _, _ = ple_nvme.safetensors_tensor_location(path, "table")
        emb = object.__new__(b70_ple_ram.Qwen4ExpPLEInt8RAMEmbedding)
        emb._table = ple_ram.PleRamTable(2048, 4096, ROW, max_slab_bytes=1 << 15, tail_bytes=2048)
        emb._table.load(path, start)
        emb._own_heads = np.array([2, 3])
        emb._stage_np = np.full((32, 4, ROW), 7, np.uint8)
        emb._dirty = 32
        ids = np.random.default_rng(4).integers(0, rows, size=(20, 4))
        emb._fill_stage(ids, 20, 24)
        for h in range(4):
            own = (ids[:, h] >= 2048) & (h >= 2)
            assert np.array_equal(emb._stage_np[:20, h][own], data[ids[:, h][own]])
            assert not emb._stage_np[:20, h][~own].any()
        assert not emb._stage_np[20:].any()  # padding and the previous step's rows cleared
        assert emb._dirty == 20
        emb._fill_stage(None, 0, 8)
        assert not emb._stage_np.any() and emb._dirty == 0


@pytest.mark.parametrize("ram,nvme,path,ok", [
    ("1", "0", "x", True), ("1", None, "x", True), ("0", "1", "x", False),
    (None, None, None, False), ("1", "1", "x", "conflict"), ("1", "0", None, "path")])
def test_flags(monkeypatch, ram, nvme, path, ok):
    _, b70_ple_ram = _vllm_ram()
    for name, value in (("B70_PLE_INT8", ram), ("B70_PLE_INT8_NVME", nvme),
                        ("B70_PLE_INT8_PATH", path)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    if ok == "conflict":
        with pytest.raises(ValueError, match="exclusive"):
            b70_ple_ram.enabled()
        with pytest.raises(ValueError, match="exclusive"):  # also when built directly
            b70_ple_ram.Qwen4ExpPLEInt8RAMEmbedding(1, 1)
    elif ok == "path":
        with pytest.raises(ValueError, match="B70_PLE_INT8_PATH"):
            b70_ple_ram.enabled()
    else:
        assert b70_ple_ram.enabled() is ok


def test_selection_order_in_ngram_embedding():
    """The RAM gate runs first, so both flags set raise before the NVMe class is picked."""
    with open(os.path.join(NVIDIA, "ngram_embedding.py")) as f:
        src = f.read()
    ram = src.index("if _b70_ple_int8_ram():")
    nvme = src.index("elif _b70_ple_int8_nvme():")
    assert ram < nvme
    assert "embedding_cls = Qwen4ExpPLEInt8RAMEmbedding" in src[ram:nvme]


# 5. the real table (optional) -------------------------------------------------

@pytest.mark.skipif(not os.path.exists(os.environ.get("B70_PLE_INT8_TEST_PATH", "/nonexistent")),
                    reason="B70_PLE_INT8_TEST_PATH (the real INT8 table) not present")
def test_real_table_slice():
    path = os.environ["B70_PLE_INT8_TEST_PATH"]
    start, shape, dt = ple_nvme.safetensors_tensor_location(path, "table")
    assert dt == "U8" and shape[1] == ROW
    lo = shape[0] // 4 + 12345
    t = ple_ram.PleRamTable(lo, lo + 200_000, ROW, max_slab_bytes=1 << 22)
    t.load(path, start, threads=4)
    t.scan_scales(DIM)
    ids = np.random.default_rng(0).integers(lo, lo + 200_000, size=2000)
    store = ple_nvme.PleNvmeRowStore(path, start, ROW, shape[0], io_threads=8, reader="py")
    assert np.array_equal(t.rows(ids), store.read_rows(ids))
    store.close()
    ref_path = os.environ.get("PLE_TABLE_PATH")
    if ref_path:
        ref = torch.load(ref_path, mmap=True, weights_only=True)
        ref = ref["table"] if isinstance(ref, dict) else ref
        b = torch.from_numpy(t.rows(ids))
        got = b[:, :DIM].view(torch.int8).float() * b[:, DIM:].contiguous().view(torch.float32)
        want = ref[torch.from_numpy(ids)].float()
        rel = float(torch.linalg.vector_norm(got - want) / torch.linalg.vector_norm(want))
        assert rel < 0.02, rel
