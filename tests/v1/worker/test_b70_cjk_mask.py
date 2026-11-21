# SPDX-License-Identifier: Apache-2.0
"""B70-CJK: CPU tests (TRITON_INTERPRET=1) for the serving-side Han mask.

1. mask builder on the served QFN tokenizer (B70_CJK_TOKENIZER=<tokenizer.json>):
   counts, kana and Cyrillic stay emittable, Han reachable only through Ext-A;
2. the per-request gate: flag parsing, system/user scan, explicit overrides,
   ChatCompletionRequest.to_sampling_params with the env on and off;
3. the device state: kernel == torch reference, flagged rows -inf on masked ids,
   unflagged rows bit-identical, every speculative verification row masked;
4. rejection sampling (stock and B70-SM fast) never emits a masked token when
   the draft proposes masked ids with high probability.
"""

import json
import os

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from vllm.triton_utils import tl, tldevice  # noqa: E402

tldevice.log1p = lambda x: tl.log(1.0 + x)

from vllm.v1.sample import b70_cjk  # noqa: E402
from vllm.v1.worker.gpu.sample import cjk_mask as cm  # noqa: E402

TOK = os.environ.get("B70_CJK_TOKENIZER")


class _Uva:
    """CPU stand-in for UvaBackedTensor (no pinned allocator on CPU)."""

    def __init__(self, size, dtype):
        self.cpu = torch.zeros(size, dtype=dtype)
        self.np = self.cpu.numpy()
        self.gpu = self.cpu

    def copy_to_uva(self, n=None):
        self.gpu = self.cpu.clone()
        return self.gpu


@pytest.fixture(scope="module")
def tokenizer():
    if not TOK or not os.path.exists(TOK):
        pytest.skip("B70_CJK_TOKENIZER not set")
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(tokenizer_file=TOK)


@pytest.fixture(scope="module")
def built(tokenizer):
    return b70_cjk.build_mask(tokenizer, 248_320)


# ------------------------------------------------------------------ 1. builder


def test_counts_on_qfn_tokenizer(built):
    mask, c = built
    assert c["vocab"] == 248_077  # 248,044 BPE + 33 added
    assert c["han"] == 55_328
    assert c["partial"] == 331  # 342 partial CJK-lead tokens - 11 kept 0xE3 ones
    assert c["masked"] == 55_659 == int(mask.sum())
    assert not mask[248_044:].any()  # padding ids


def test_e3_partials_kept_and_lead_rule():
    keep = [b"\xe3", b"\xe3\x80", b"\xe3\x81", b"\xe3\x82", b"\xe3\x83", b" \xe3\x82"]
    for b in keep:
        assert b70_cjk.classify(b) is None, b
    assert b70_cjk.classify(b"\xe3\x90") == "partial"  # Ext-A U+3400..
    assert b70_cjk.classify(b"\xe4") == "partial"
    assert b70_cjk.classify(b"\xe4\xb8") == "partial"
    assert b70_cjk.classify(b"\xe4\xb7") is None  # U+4DC0 hexagrams, not Han
    assert b70_cjk.classify(b"\xe9") == "partial"
    assert b70_cjk.classify(b"\xb8\xad") is None  # continuation only
    assert b70_cjk.classify("中".encode()) == "han"
    assert b70_cjk.classify("。".encode()) is None
    assert b70_cjk.classify("да".encode()) is None


def test_specials_never_masked(tokenizer, built):
    mask, _ = built
    for i in tokenizer.all_special_ids + list(tokenizer.get_added_vocab().values()):
        assert not mask[i]


def _ids(tok, s):
    return tok.encode(s, add_special_tokens=False)


def test_kana_cyrillic_latin_emittable(tokenizer, built):
    mask, _ = built
    kana = "".join(
        chr(c) for c in list(range(0x3041, 0x3097)) + list(range(0x30A1, 0x30FB))
    )
    for ch in kana:
        assert not mask[_ids(tokenizer, ch)].any(), hex(ord(ch))
    for s in [
        "Здравей, свят! Как си днес?",
        "Hello world",
        "ありがとうございます",
        "ё ъ ѝ",
    ]:
        assert not mask[_ids(tokenizer, s)].any(), s


def test_han_unreachable_except_ext_a(tokenizer, built):
    """Every URO Han code point's canonical encoding contains a masked token;
    the only escapes are Ext-A / compatibility ideographs without a full token,
    spelled from the lone 0xE3 / 0xEF byte tokens (kept for kana)."""
    mask, _ = built
    reach = []
    for lo, hi in b70_cjk.HAN_RANGES:
        for c in range(lo, hi + 1):
            if not mask[_ids(tokenizer, chr(c))].any():
                reach.append(c)
    assert all(0x3400 <= c <= 0x3FFF or 0xF900 <= c <= 0xFAFF for c in reach), [
        hex(c) for c in reach[:10]
    ]
    assert not any(0x4E00 <= c <= 0x9FFF for c in reach)
    print(f"Han code points reachable through unmasked byte tokens: {len(reach)}")


# ------------------------------------------------------------------- 2. gate


def test_parse_and_resolve():
    p = b70_cjk.parse_flag
    assert p(None) is None
    assert p(True) is True and p(False) is False
    assert p(0) is False and p(1) is True
    assert p("off") is False and p("TRUE") is True and p(" no ") is False
    with pytest.raises(ValueError):
        p("maybe")
    with pytest.raises(ValueError):
        p(2)
    assert b70_cjk.resolve(None, False) is True
    assert b70_cjk.resolve(None, True) is False
    assert b70_cjk.resolve(True, True) is True
    assert b70_cjk.resolve(0, False) is False


def test_messages_scan_roles_and_parts():
    m = b70_cjk.messages_have_han
    assert not m([{"role": "user", "content": "Здравей"}])
    assert m([{"role": "user", "content": "請用中文回答"}])
    assert m([{"role": "system", "content": "答"}, {"role": "user", "content": "x"}])
    assert m([{"role": "developer", "content": "漢"}])
    # a Han leak in an earlier assistant turn must not switch the mask off
    assert not m(
        [
            {"role": "user", "content": "Какво е това?"},
            {"role": "assistant", "content": "Това е 例如 пример"},
            {"role": "user", "content": "Продължи"},
        ]
    )
    assert m([{"role": "user", "content": [{"type": "text", "text": "看"}]}])
    assert not m(
        [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]
    )
    assert not m([{"role": "tool", "content": "中文"}])


def test_trade_off_glosses_in_bulgarian_answer_are_masked():
    """A Bulgarian conversation about Daoism and Buddhism: Bulgarian user turns, an answer
    that legitimately glosses 无, 真人, 格義, 化胡經 next to one true bleed
    ("терени,历史 на дебатите"). The auto-gate cannot tell the two apart: user
    turns carry no Han, so the mask is on and the glosses go too (the model then
    writes them in transliteration or drops them). The escape is explicit:
    cjk_mask=false for the conversation, or Han in a user turn (e.g. the user
    pastes 格義), which turns the mask off for that request."""
    convo = [
        {"role": "system", "content": "Отговаряй на български."},
        {"role": "user", "content": "Как даоизмът влияе на ранния китайски будизъм?"},
        {
            "role": "assistant",
            "content": "Чрез метода 格義 (geyi) и понятия като 无 и 真人; "
            "спорът около 化胡經 ... терени,历史 на дебатите",
        },
        {"role": "user", "content": "Разкажи повече за geyi."},
    ]
    assert b70_cjk.chat_extra_args({}, convo, None) == {"cjk_mask": True}
    assert b70_cjk.chat_extra_args({"cjk_mask": False}, convo, None) == {
        "cjk_mask": False
    }
    pasted = convo + [{"role": "user", "content": "А 格義 точно какво значи?"}]
    assert b70_cjk.chat_extra_args({}, pasted, None) == {"cjk_mask": False}


def test_chat_extra_args():
    f = b70_cjk.chat_extra_args
    u = [{"role": "user", "content": "Здравей"}]
    h = [{"role": "user", "content": "你好"}]
    assert f({}, u, None) == {"cjk_mask": True}
    assert f({}, h, None) == {"cjk_mask": False}
    assert f({"cjk_mask": 0}, u, None) == {"cjk_mask": False}
    assert f({"cjk_mask": "true"}, h, None) == {"cjk_mask": True}
    assert f({}, u, {"cjk_mask": "off"}) == {"cjk_mask": False}
    src = {"other": 1}
    out = f(src, u, None)
    assert src == {"other": 1} and out == {"other": 1, "cjk_mask": True}


def _req(**kw):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    body = {"model": "m", "messages": [{"role": "user", "content": "Здравей"}], **kw}
    return ChatCompletionRequest.model_validate_json(json.dumps(body))


def test_protocol_env_off_is_stock(monkeypatch):
    monkeypatch.delenv(b70_cjk.ENV, raising=False)
    sp = _req().to_sampling_params(64, {})
    assert sp.extra_args is None
    sp = _req(vllm_xargs={"cjk_mask": False}).to_sampling_params(64, {})
    assert sp.extra_args == {"cjk_mask": 0}  # passed through untouched


def test_protocol_env_on(monkeypatch):
    monkeypatch.setenv(b70_cjk.ENV, "1")
    assert _req().to_sampling_params(64, {}).extra_args == {"cjk_mask": True}
    zh = [{"role": "user", "content": "用中文解释"}]
    assert _req(messages=zh).to_sampling_params(64, {}).extra_args == {
        "cjk_mask": False
    }
    sp = _req(vllm_xargs={"cjk_mask": False}).to_sampling_params(64, {})
    assert sp.extra_args == {"cjk_mask": False}
    sp = _req(messages=zh, vllm_xargs={"cjk_mask": True}).to_sampling_params(64, {})
    assert sp.extra_args == {"cjk_mask": True}
    sp = _req(cjk_mask=False).to_sampling_params(64, {})
    assert sp.extra_args == {"cjk_mask": False}
    from vllm.exceptions import VLLMValidationError

    with pytest.raises(VLLMValidationError):
        _req(vllm_xargs={"cjk_mask": "sometimes"}).to_sampling_params(64, {})


def test_prompt_token_fallback(tokenizer):
    b70_cjk._PROMPT_MASK = None
    assert not b70_cjk.prompt_has_han_tokens(tokenizer, _ids(tokenizer, "Здравей"))
    assert not b70_cjk.prompt_has_han_tokens(tokenizer, _ids(tokenizer, "ありがとう。"))
    assert b70_cjk.prompt_has_han_tokens(tokenizer, _ids(tokenizer, "Здравей 中国"))
    assert not b70_cjk.prompt_has_han_tokens(tokenizer, None)


# ----------------------------------------------------------- 3. device state


class _SP:
    def __init__(self, extra):
        self.extra_args = extra


class _FakeTok:
    """Byte-level vocab: ids 0..9 ASCII, 10 '中', 11 lone 0xE4, 12 'д', 13 lone 0xE3."""

    def __init__(self):
        u = {b: c for c, b in b70_cjk._U2B.items()}
        toks = [chr(ord("a") + i) for i in range(10)]
        toks += ["".join(u[b] for b in "中".encode())]
        toks += [u[0xE4]]
        toks += ["".join(u[b] for b in "д".encode())]
        toks += [u[0xE3]]
        self._v = {t: i for i, t in enumerate(toks)}
        self.all_special_ids = []

    def get_vocab(self):
        return dict(self._v)

    def get_added_vocab(self):
        return {}


@pytest.fixture
def state(monkeypatch):
    monkeypatch.setattr(cm, "UvaBackedTensor", _Uva)
    return cm.CjkMaskState(None, 4, 16, torch.device("cpu"), tokenizer=_FakeTok())


def test_state_flags_and_mask(state):
    assert state.counts["masked"] == 2
    assert state.mask.nonzero().flatten().tolist() == [10, 11]
    assert state.add_request(0, _SP(None)) is True  # engine-direct: masked
    assert state.add_request(1, _SP({"cjk_mask": False})) is False
    assert state.add_request(2, _SP({"cjk_mask": True})) is True
    assert state.add_request(3, _SP({"cjk_mask": 0})) is False
    state.apply_staged_writes()
    assert state.flags.gpu.tolist() == [1, 0, 1, 0]
    assert state.any_flagged(np.array([1, 2]))
    assert not state.any_flagged(np.array([1, 3]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_kernel_matches_reference_and_rows(state, dtype):
    for i, f in enumerate([1, 0, 1, 0]):
        state.add_request(i, _SP({"cjk_mask": bool(f)}))
    state.apply_staged_writes()
    # spec decode: 3 requests x (7 drafts + bonus) rows, request slots 2, 1, 0
    idx = torch.tensor([2, 1, 0], dtype=torch.int32)
    eim = idx.repeat_interleave(8)
    g = torch.Generator().manual_seed(0)
    base = (torch.randn(24, 16, generator=g) * 3).to(dtype)
    ref = base.clone()
    cm.apply_cjk_mask_reference(ref, eim, state.flags.gpu, state.mask)
    ker = base.clone()
    cm.apply_cjk_mask(ker, eim, state.flags.gpu, state.mask_u8)
    assert torch.equal(ref, ker)
    flagged = state.flags.gpu[eim.long()] != 0
    assert flagged.sum() == 16  # slots 2 and 0, all 8 rows each
    assert torch.isneginf(ker[flagged][:, [10, 11]]).all()
    keep = [i for i in range(16) if i not in (10, 11)]
    assert torch.equal(ker[flagged][:, keep], base[flagged][:, keep])
    assert torch.equal(ker[~flagged], base[~flagged])  # bit-identical


def test_apply_skips_unflagged_batch_and_strided(state):
    state.add_request(0, _SP({"cjk_mask": False}))
    state.add_request(1, _SP({"cjk_mask": True}))
    state.apply_staged_writes()
    x = torch.randn(4, 16)
    y = x.clone()
    state.apply(y, torch.zeros(4, dtype=torch.int32), np.array([0]))
    assert torch.equal(x, y)
    # sharded path: logits[:, :vocab] of a wider (padded) row
    wide = torch.randn(2, 20)
    view = wide[:, :16]
    orig = wide.clone()
    cm.apply_cjk_mask(
        view, torch.tensor([1, 0], dtype=torch.int32), state.flags.gpu, state.mask_u8
    )
    assert torch.isneginf(wide[0, [10, 11]]).all()
    assert torch.equal(wide[0, 16:], orig[0, 16:]) and torch.equal(wide[1], orig[1])


# ------------------------------------------------------- 4. spec-decode verify


@pytest.mark.parametrize("verifier", ["stock", "fast"])
def test_rejection_never_emits_masked(verifier):
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
    from vllm.v1.worker.gpu.sample import b70_sampler_fast as sm
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

    class _T:
        def __init__(self, t):
            self.gpu, self.np = t, t.numpy()

    class _St:
        def __init__(self, r, temp, k, p):
            self.temperature = _T(torch.full((r,), temp))
            self.top_k = _T(torch.full((r,), k, dtype=torch.int32))
            self.top_p = _T(torch.full((r,), p))
            self.seeds = _T(torch.arange(r, dtype=torch.int64) * 7 + 1)
            self.min_p = _T(torch.zeros(r))

    r, steps, v, kd, k, temp, p = 6, 7, 512, 16, 20, 0.7, 0.95
    n = r * (steps + 1)
    vmask = torch.zeros(v, dtype=torch.bool)
    vmask[100:160] = True  # the "Han" ids
    g = torch.Generator().manual_seed(1)
    target = torch.randn(n, v, generator=g)
    target[:, 100:110] += 6.0  # target prefers masked ids before the mask
    cand = torch.stack(
        [torch.randperm(v, generator=g)[:kd] for _ in range(r * steps)]
    ).reshape(r, steps, kd)
    cand[..., :4] = torch.tensor([100, 101, 102, 103])
    dscore = torch.randn(r, steps, kd, generator=g)
    dscore[..., :4] += 8.0  # drafter proposes masked ids with high probability
    draft = torch.full((r, steps, v), -float("inf"))
    draft.scatter_(2, cand, dscore)
    q = torch.softmax(dscore / temp, dim=-1)
    pick = torch.multinomial(q.reshape(-1, kd), 1, generator=g).reshape(r, steps)
    dtok = cand.gather(2, pick.unsqueeze(-1)).squeeze(-1)
    assert vmask[dtok].float().mean() > 0.9

    cu = torch.arange(0, n + 1, steps + 1, dtype=torch.int32)
    idx = torch.arange(r, dtype=torch.int32)
    eim = idx.repeat_interleave(steps + 1)
    local = torch.arange(steps + 1, dtype=torch.int32).repeat(r)
    ds = torch.full((n,), -1, dtype=torch.int64)
    ds.view(r, steps + 1)[:, 1:] = dtok
    pos = (torch.arange(r).unsqueeze(1) * 1000 + torch.arange(steps + 1)).reshape(-1)
    st = _St(r, temp, k, p)
    flags = torch.ones(r, dtype=torch.int32)

    def stock(logits):
        proc = apply_top_k_top_p_pytorch(
            logits.float() / temp, torch.full((n,), k), torch.full((n,), p)
        )
        return rejection_sample(
            proc,
            draft,
            ds,
            cu,
            pos,
            idx,
            eim,
            local,
            st.temperature.gpu,
            st.seeds.gpu,
            steps,
        )

    def fast(logits):
        return sm.fast_reject(
            logits, draft, cand, ds, pos, cu, idx, eim, st, steps, k, block=128
        )

    def emitted(out):
        s, num = out
        return torch.cat([s[i, : int(num[i])] for i in range(r)])

    run = stock if verifier == "stock" else fast
    try:
        control = emitted(run(target))
    except Exception as e:  # some Triton interpreters cannot run the stock kernel
        # interpreter limits (nested jit without TRITON_INTERPRET at import;
        # scalar load masks in the B70-SM residual branch on some Triton builds)
        if any(m in repr(e) for m in ("outside of the scope", "mask has unsupported")):
            pytest.skip(f"Triton interpreter cannot run the {verifier} verifier")
        raise
    # control: without the mask the verifier emits masked ids
    assert vmask[control].any()
    masked = target.clone()
    cm.apply_cjk_mask(masked, eim, flags, vmask.view(torch.uint8))
    toks = emitted(run(masked))
    assert toks.numel() >= r
    assert not vmask[toks].any()


def test_fast_support_excludes_masked():
    """B70-SM verifies on the top-k support of the processed target logits; a
    draft outside it has t_lp = -inf and is never accepted, and the resample
    draws from the support. So masked ids must never enter the support."""
    from vllm.v1.worker.gpu.sample import b70_sampler_fast as sm

    class _T:
        def __init__(self, t):
            self.gpu, self.np = t, t.numpy()

    class _St:
        temperature = _T(torch.full((4,), 0.7))
        top_k = _T(torch.full((4,), 20, dtype=torch.int32))
        top_p = _T(torch.full((4,), 0.95))
        min_p = _T(torch.zeros(4))

    v = 1024
    vmask = torch.zeros(v, dtype=torch.bool)
    vmask[100:160] = True
    g = torch.Generator().manual_seed(3)
    logits = torch.randn(32, v, generator=g)
    logits[:, 100:130] += 8.0  # masked ids dominate before the mask
    eim = torch.arange(4, dtype=torch.int32).repeat_interleave(8)
    ids, lm, _ = sm.topk_support(logits.clone(), eim, _St, 20, block=256)
    assert vmask[ids.long().clamp(0, v - 1)][torch.isfinite(lm)].any()  # control
    masked = logits.clone()
    cm.apply_cjk_mask(
        masked, eim, torch.ones(4, dtype=torch.int32), vmask.view(torch.uint8)
    )
    ids, lm, _ = sm.topk_support(masked, eim, _St, 20, block=256)
    live = torch.isfinite(lm)
    assert live.sum(1).min() >= 1
    assert not vmask[ids.long()[live]].any()


def test_env_default_off(monkeypatch):
    monkeypatch.delenv(b70_cjk.ENV, raising=False)
    assert not b70_cjk.enabled()
    monkeypatch.setenv(b70_cjk.ENV, "1")
    assert b70_cjk.enabled()
