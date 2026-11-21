# SPDX-License-Identifier: Apache-2.0
"""B70-SM: CPU tests (TRITON_INTERPRET=1) for the top-k-first sampler.

1. support/probabilities of topk_support vs a float64 sort-based reference, with
   bf16-quantised logits (real ties), mixed temperature / top-k / top-p, greedy and
   top-k=1 rows; and vs the stock apply_top_k_top_p_pytorch masks;
2. fast_sample: chi-square against the reference distribution, greedy rows;
3. fast_reject vs the stock rejection_sample on identical inputs: token-for-token
   agreement, per-token acceptance probability, and the marginal law of the first
   emitted token (must equal the target distribution).
"""

import os

os.environ.setdefault("TRITON_INTERPRET", "1")
# vllm.triton_utils keeps Triton on with zero active drivers only in this case
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402

from vllm.triton_utils import tl, tldevice  # noqa: E402

# libdevice.log1p has no interpreter lowering (the stock _resample_kernel and this
# module both call it); stand in log(1 + x) so both run, identically, on CPU.
tldevice.log1p = lambda x: tl.log(1.0 + x)

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch  # noqa: E402
from vllm.v1.worker.gpu.sample import b70_sampler_fast as sm  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (  # noqa: E402
    rejection_sample,
)


class _T:
    def __init__(self, t):
        self.gpu = t
        self.np = t.numpy()


class States:
    def __init__(self, temp, top_k, top_p, seeds, min_p=None):
        self.temperature = _T(torch.tensor(temp, dtype=torch.float32))
        self.top_k = _T(torch.tensor(top_k, dtype=torch.int32))
        self.top_p = _T(torch.tensor(top_p, dtype=torch.float32))
        self.seeds = _T(torch.tensor(seeds, dtype=torch.int64))
        self.min_p = _T(torch.zeros(len(temp)) if min_p is None else torch.tensor(min_p))


def ref_support(logits, temp, k, p):
    """Float64-ordered reference: scaled logits fp32 (same division), order by
    (value desc, id asc), top-k, softmax over the k, top-p exclusive cumsum."""
    n, v = logits.shape
    out = []
    for r in range(n):
        x = logits[r].float()
        if temp[r] != 0:
            x = x / torch.tensor(temp[r], dtype=torch.float32)
        kk = 1 if temp[r] == 0 else min(int(k[r]), v)
        order = sorted(range(v), key=lambda i: (-float(x[i]), i))[:kk]
        vals = np.array([float(x[i]) for i in order], dtype=np.float64)
        pr = np.exp(vals - vals.max())
        pr /= pr.sum()
        cs = np.cumsum(pr) - pr
        keep = (cs < p[r]) | (p[r] >= 1.0)
        keep[0] = True
        kept = [order[j] for j in range(kk) if keep[j]]
        kv = vals[keep]
        lp = kv - (kv.max() + np.log(np.exp(kv - kv.max()).sum()))
        out.append((kept, lp))
    return out


def _quantised(n, v, seed, scale=3.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, v, generator=g) * scale).to(torch.bfloat16)


V = 3000
BLK = 512


def _support(logits, st, kmax):
    eim = torch.arange(logits.shape[0], dtype=torch.int32)
    return sm.topk_support(logits, eim, st, kmax, block=BLK)


def test_support_exact_with_ties_and_mixed_params():
    n = 12
    logits = _quantised(n, V, 1)
    # force a tie at the k-th boundary on row 0 (k=20): equal values straddling
    srt = torch.sort(logits[0].float(), descending=True).values
    logits[0, torch.argsort(logits[0].float(), descending=True)[19:22]] = srt[19].to(
        torch.bfloat16
    )
    temp = [0.7, 0.7, 0.0, 1.0, 0.3, 0.7, 0.7, 0.7, 1.5, 0.7, 0.7, 0.7]
    k = [20, 20, 20, 20, 5, 1, 32, 20, 20, 20, 2, 20]
    p = [0.8, 0.95, 0.8, 1.0, 0.9, 0.8, 0.5, 0.01, 0.8, 1.0, 0.95, 0.999]
    st = States(temp, k, p, list(range(n)))
    ids, lm, lp = _support(logits, st, 32)
    ref = ref_support(logits, temp, k, p)
    bad = 0
    for r in range(n):
        got = [int(i) for i, l in zip(ids[r], lp[r]) if l > -float("inf")]
        kept, rlp = ref[r]
        if got != kept:
            bad += 1
            print("row", r, got, kept)
            continue
        assert np.allclose(
            [float(l) for l in lp[r] if l > -float("inf")], rlp, atol=2e-5
        )
    assert bad == 0
    # tie row: the lowest ids of the tied value are the ones kept
    tied = (logits[0].float() == srt[19]).nonzero().flatten().tolist()
    assert len(tied) >= 3


def test_support_matches_stock_pytorch_masks():
    """Against the stock sort-based apply_top_k_top_p_pytorch. The stock Triton
    (Qrita) kernel does not run in the CPU interpreter, so the sort path stands in.
    It keeps *all* ties at the k-th value while Qrita (and this module) keep exactly
    k, lowest ids first; so with top-p off the two must agree up to that tie class,
    and with top-p on they must agree wherever no tie touches the nucleus edge."""
    n = 10
    logits = _quantised(n, V, 2)
    k = [20, 20, 10, 20, 32, 20, 20, 20, 5, 20]
    p = [1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.95, 0.5, 0.9, 0.99]
    temp = [0.7] * n
    st = States(temp, k, p, list(range(n)))
    ids, lm, lp = _support(logits, st, 32)
    scaled = logits.float() / 0.7
    stock = apply_top_k_top_p_pytorch(scaled.clone(), torch.tensor(k), torch.tensor(p))
    for r in range(n):
        want = set(torch.nonzero(stock[r] > -float("inf")).flatten().tolist())
        got = {int(i) for i, l in zip(ids[r], lp[r]) if l > -float("inf")}
        kth = torch.topk(scaled[r], k[r]).values[-1]
        tie_class = set(torch.nonzero(scaled[r] == kth).flatten().tolist())
        if p[r] == 1.0:
            assert got <= want and (want - got) <= tie_class, (r, got ^ want)
            assert len(got) == k[r]
        elif got != want:
            # only acceptable when a tie sits on the k-th value or the nucleus edge
            vals = {float(scaled[r][t]) for t in got ^ want}
            assert any(v in {float(scaled[r][t]) for t in tie_class} or
                       sum(1 for t in range(V) if float(scaled[r][t]) == v) > 1
                       for v in vals), (r, got ^ want)

def test_fast_sample_chisquare_and_greedy():
    n = 1200
    base = _quantised(1, V, 3)
    logits = base.expand(n, V).contiguous()
    temp = [0.7] * n
    st = States(temp, [20] * n, [0.9] * n, list(range(n)))
    pos = torch.arange(n, dtype=torch.int64) + 5
    eim = torch.arange(n, dtype=torch.int32)
    # distinct seeds per row give independent draws
    st.seeds.gpu.copy_(torch.randint(0, 2**62, (n,), dtype=torch.int64))
    got = sm.fast_sample(logits, eim, st, pos, 20, block=BLK)
    (kept, lp), = ref_support(base, [0.7], [20], [0.9])
    probs = np.exp(lp)
    cnt = np.array([(got == t).sum().item() for t in kept], dtype=np.float64)
    assert cnt.sum() == n, "sampled outside the support"
    exp = probs * n
    ok = exp >= 5
    pooled_o = np.append(cnt[ok], cnt[~ok].sum())
    pooled_e = np.append(exp[ok], exp[~ok].sum())
    if pooled_e[-1] < 5:
        pooled_o, pooled_e = pooled_o[:-1], pooled_e[:-1]
    pooled_e = pooled_e * pooled_o.sum() / pooled_e.sum()
    chi = stats.chisquare(pooled_o, pooled_e)
    print("chi-square p =", chi.pvalue, "bins", len(pooled_o))
    assert chi.pvalue > 1e-3

    # greedy rows and top-k=1 rows
    g = _quantised(4, V, 4)
    st2 = States([0.0, 0.7, 0.0, 0.7], [20, 1, 1, 20], [0.8] * 4, [1, 2, 3, 4])
    eim4 = torch.arange(4, dtype=torch.int32)
    out = sm.fast_sample(g, eim4, st2, torch.arange(4), 20, block=BLK)
    want = [
        sorted(range(V), key=lambda i: (-float(g[r].float()[i]), i))[0]
        for r in range(4)
    ]
    assert out[0] == want[0] and out[1] == want[1] and out[2] == want[2]


def _spec_case(r, steps, v, seed, temp=0.7, k=20, p=0.9, kd=16):
    """r requests x (steps+1) rows. Draft = dense fp32 [r, steps, v] with kd finite
    candidates; draft tokens sampled from q."""
    g = torch.Generator().manual_seed(seed)
    tl_ = (torch.randn(r, steps + 1, v, generator=g) * 2.5).to(torch.bfloat16)
    cand = torch.stack(
        [
            torch.stack(
                [torch.randperm(v, generator=g)[:kd] for _ in range(steps)]
            )
            for _ in range(r)
        ]
    )  # [r, steps, kd]
    dscore = torch.randn(r, steps, kd, generator=g) * 2.0
    # make the draft overlap the target's head so acceptance is non-trivial
    for a in range(r):
        for s_ in range(steps):
            top = torch.topk(tl_[a, s_].float(), 6).indices
            cand[a, s_, :6] = top
            # q close to the target head (temperature-scaled logits + noise)
            dscore[a, s_, :6] = tl_[a, s_].float()[top] + torch.randn(6, generator=g) * 0.4
    draft = torch.full((r, steps, v), -float("inf"))
    draft.scatter_(2, cand, dscore)
    q = torch.softmax(dscore / temp, dim=-1)
    pick = torch.multinomial(q.reshape(-1, kd), 1, generator=g).reshape(r, steps)
    dtok = cand.gather(2, pick.unsqueeze(-1)).squeeze(-1)  # [r, steps]
    return tl_, draft, cand, dtok


def _run_both(tl_, draft, cand, dtok, temp, k, p, seeds, steps, v):
    r = tl_.shape[0]
    n = r * (steps + 1)
    logits = tl_.reshape(n, v)
    cu = torch.arange(0, n + 1, steps + 1, dtype=torch.int32)
    idx_mapping = torch.arange(r, dtype=torch.int32)
    eim = idx_mapping.repeat_interleave(steps + 1)
    local = torch.arange(steps + 1, dtype=torch.int32).repeat(r)
    draft_sampled = torch.full((n,), -1, dtype=torch.int64)
    draft_sampled.view(r, steps + 1)[:, 1:] = dtok
    pos = (torch.arange(r).unsqueeze(1) * 1000 + torch.arange(steps + 1)).reshape(-1)
    st = States([temp] * r, [k] * r, [p] * r, seeds)
    # stock: temperature + top-k/top-p on an fp32 copy, then the stock kernels
    proc = logits.float() / temp
    proc = apply_top_k_top_p_pytorch(
        proc, torch.full((n,), k), torch.full((n,), p)
    )
    s_stock, n_stock = rejection_sample(
        proc,
        draft,
        draft_sampled,
        cu,
        pos,
        idx_mapping,
        eim,
        local,
        st.temperature.gpu,
        st.seeds.gpu,
        steps,
    )
    s_fast, n_fast = sm.fast_reject(
        logits,
        draft,
        cand,
        draft_sampled,
        pos,
        cu,
        idx_mapping,
        eim,
        st,
        steps,
        k,
        block=BLK,
    )
    return s_stock, n_stock, s_fast, n_fast


def test_reject_matches_stock_tokenwise():
    r, steps, v = 24, 7, 2048
    tl_, draft, cand, dtok = _spec_case(r, steps, v, 11)
    seeds = list(range(100, 100 + r))
    s_s, n_s, s_f, n_f = _run_both(tl_, draft, cand, dtok, 0.7, 20, 0.9, seeds, steps, v)
    assert n_s.tolist() == n_f.tolist(), (n_s.tolist(), n_f.tolist())
    for a in range(r):
        m = int(n_s[a])
        assert s_s[a, :m].tolist() == s_f[a, :m].tolist(), a
    print("accepted lengths", n_s.tolist())


def test_reject_greedy_and_mixed_rows():
    r, steps, v = 6, 7, 2048
    tl_, draft, cand, dtok = _spec_case(r, steps, v, 12)
    # greedy rows accept iff draft == argmax: seed the draft with the target argmax
    # on a prefix for rows 0 and 1
    for a in (0, 1):
        for s_ in range(3):
            dtok[a, s_] = tl_[a, s_].float().argmax()
            cand[a, s_, 0] = dtok[a, s_]
            draft[a, s_] = -float("inf")
            draft[a, s_, cand[a, s_]] = torch.randn(cand.shape[-1])
    n = r * (steps + 1)
    temps = [0.0, 0.0, 0.7, 0.7, 0.0, 0.7]
    ks = [20, 1, 20, 5, 20, 32]
    ps = [0.8] * r
    logits = tl_.reshape(n, v)
    cu = torch.arange(0, n + 1, steps + 1, dtype=torch.int32)
    idx_mapping = torch.arange(r, dtype=torch.int32)
    eim = idx_mapping.repeat_interleave(steps + 1)
    local = torch.arange(steps + 1, dtype=torch.int32).repeat(r)
    ds = torch.full((n,), -1, dtype=torch.int64)
    ds.view(r, steps + 1)[:, 1:] = dtok
    pos = (torch.arange(r).unsqueeze(1) * 1000 + torch.arange(steps + 1)).reshape(-1)
    st = States(temps, ks, ps, list(range(r)))
    proc = logits.float().clone()
    for a in range(r):
        rows = slice(a * (steps + 1), (a + 1) * (steps + 1))
        if temps[a] != 0:
            proc[rows] = proc[rows] / temps[a]
        proc[rows] = apply_top_k_top_p_pytorch(
            proc[rows], torch.full((steps + 1,), ks[a]), torch.full((steps + 1,), ps[a])
        )
    s_s, n_s = rejection_sample(
        proc, draft, ds, cu, pos, idx_mapping, eim, local,
        st.temperature.gpu, st.seeds.gpu, steps,
    )
    s_f, n_f = sm.fast_reject(
        logits, draft, cand, ds, pos, cu, idx_mapping, eim, st, steps, 32, block=BLK
    )
    assert n_s.tolist() == n_f.tolist()
    for a in range(r):
        m = int(n_s[a])
        assert s_s[a, :m].tolist() == s_f[a, :m].tolist(), a
    assert n_s[0] >= 4 and n_s[1] >= 4  # greedy rows accepted the planted prefix


def test_first_token_marginal_equals_target_and_acceptance():
    """One draft token per request, many requests with a shared (p, q): the first
    emitted token is distributed as p, and P(accept d) = min(1, p(d)/q(d))."""
    r, steps, v = 600, 1, 1024
    g = torch.Generator().manual_seed(5)
    row = (torch.randn(1, v, generator=g) * 2.5).to(torch.bfloat16)
    tl_ = row.reshape(1, 1, v).expand(r, 2, v).contiguous()
    kd = 16
    cand1 = torch.cat([torch.topk(row[0].float(), 8).indices, torch.randperm(v, generator=g)[:8]])
    cand1 = torch.unique(cand1)[:kd]
    kd = cand1.numel()
    dscore1 = torch.randn(kd, generator=g) * 1.5
    temp, k, p = 0.7, 20, 0.9
    draft = torch.full((r, steps, v), -float("inf"))
    draft[:, 0, cand1] = dscore1
    cand = cand1.reshape(1, 1, kd).expand(r, 1, kd).contiguous()
    q = torch.softmax(dscore1 / temp, 0)
    dtok = cand1[torch.multinomial(q, r, replacement=True, generator=g)].reshape(r, 1)
    seeds = torch.randint(0, 2**62, (r,), generator=g).tolist()
    s_s, n_s, s_f, n_f = _run_both(tl_, draft, cand, dtok, temp, k, p, seeds, steps, v)

    (kept, lp), = ref_support(row, [temp], [k], [p])
    pt = dict(zip(kept, np.exp(lp)))
    qd = dict(zip(cand1.tolist(), q.numpy()))
    for name, s_, n_ in (("stock", s_s, n_s), ("fast", s_f, n_f)):
        first = s_[:, 0].numpy()
        # acceptance rate per distinct draft token
        acc = (n_ == 2).numpy()
        for d in set(dtok[:, 0].tolist()):
            sel = dtok[:, 0].numpy() == d
            if sel.sum() < 60:
                continue
            want = min(1.0, pt.get(d, 0.0) / qd[d])
            got = acc[sel].mean()
            se = np.sqrt(max(want * (1 - want), 1e-4) / sel.sum())
            assert abs(got - want) < 5 * se + 0.02, (name, d, got, want)
        cnt = np.array([(first == t).sum() for t in kept], dtype=np.float64)
        assert cnt.sum() == r, (name, "token outside support")
        exp = np.array([pt[t] for t in kept]) * r
        ok = exp >= 5
        po = np.append(cnt[ok], cnt[~ok].sum())
        pe = np.append(exp[ok], exp[~ok].sum())
        if pe[-1] < 5:
            po, pe = po[:-1], pe[:-1]
        pe = pe * po.sum() / pe.sum()
        c = stats.chisquare(po, pe)
        print(name, "first-token chi-square p =", c.pvalue, "bins", len(po))
        assert c.pvalue > 1e-3, name
    # same decisions as stock, draw for draw
    same = (n_s == n_f).float().mean().item()
    tokmatch = float((s_s[:, 0] == s_f[:, 0]).float().mean())
    print("decision agreement", same, "first-token agreement", tokmatch)
    assert same > 0.995 and tokmatch > 0.99


def test_gate_and_fallbacks_in_rejection_sampler(monkeypatch):
    """The wiring in RejectionSampler._b70_try_fast: off by default, taken when the
    batch is inside the envelope, declined (None -> stock path) outside it."""
    from types import SimpleNamespace

    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    r, steps, v = 2, 7, 1024
    tl_, draft, cand, dtok = _spec_case(r, steps, v, 21)
    n = r * (steps + 1)
    st = States([0.7, 0.0], [20, 20], [0.9, 0.9], [1, 2])
    num_none = np.full(r, -1, dtype=np.int32)

    class FakeSampler:
        use_fp64_gumbel = False
        return_sampling_mask = False
        sampling_states = st

        def apply_sampling_params(self, logits, *a, **kw):
            assert kw["b70_fast"] is True
            return logits

    rs = object.__new__(RejectionSampler)
    rs.sampler = FakeSampler()
    rs.use_block_verification = False
    rs.synthetic_conditional_rates = None
    rs.watermark_key = None
    rs.enable_adaptive_verification = False
    rs.num_speculative_steps = steps
    st.num_logprobs = num_none  # sampling_states.max_num_logprobs reads this
    st.max_num_logprobs = lambda idx: int(num_none[idx].max())
    ib = SimpleNamespace(
        idx_mapping_np=np.arange(r),
        idx_mapping=torch.arange(r, dtype=torch.int32),
        expanded_idx_mapping=torch.arange(r, dtype=torch.int32).repeat_interleave(steps + 1),
        expanded_local_pos=torch.arange(steps + 1, dtype=torch.int32).repeat(r),
        seq_lens_cpu_upper_bound=torch.zeros(r, dtype=torch.int32),
        cu_num_logits=torch.arange(0, n + 1, steps + 1, dtype=torch.int32),
    )
    ds = torch.full((n,), -1, dtype=torch.int64)
    ds.view(r, steps + 1)[:, 1:] = dtok
    pos = torch.arange(n, dtype=torch.int64)
    logits = tl_.reshape(n, v)

    def call(**over):
        args = dict(draft_logits=draft, draft_candidate_ids=cand, invalid_drafts=None)
        args.update(over)
        return rs._b70_try_fast(
            logits, ib, args["draft_logits"], args["draft_candidate_ids"],
            args["invalid_drafts"], ds, pos,
        )

    monkeypatch.delenv("B70_SAMPLER_FAST", raising=False)
    assert not sm.enabled()
    monkeypatch.setenv("B70_SAMPLER_FAST", "1")
    assert sm.enabled()
    out = call()
    assert out is not None and out[1].shape == (r,)
    assert call(draft_candidate_ids=None) is None  # drafter without cached candidates
    assert call(invalid_drafts=torch.zeros(n, dtype=torch.bool)) is None  # grammar
    st.min_p.gpu[0] = 0.05
    st.min_p.np[0] = 0.05
    assert call() is None  # min_p
    st.min_p.np[0] = 0.0
    st.top_k.np[0] = v  # top-k disabled on a sampled request
    assert call() is None
    st.top_k.np[0] = 20
    st.top_k.np[1] = v  # top-k disabled on a greedy request is fine
    assert call() is not None
    st.num_logprobs = np.array([5, -1], dtype=np.int32)
    num_none[:] = [5, -1]
    assert call() is None  # logprobs requested


def test_warmup_runs():
    sm.warmup(torch.device("cpu"), 2048, 7, 16)
