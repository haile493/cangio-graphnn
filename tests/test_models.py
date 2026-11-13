"""Smoke test: every model builds from this repo alone and returns a full-resolution logit map."""
import pytest
import torch

from src.models import build_model

BASELINES = ["hrsei_gnn", "venagnosis_pfgnn", "gsri_net", "cgmn_net"]


@pytest.mark.parametrize("arch", BASELINES)
def test_baseline_forward(arch):
    model = build_model(arch, in_channels=9, num_classes=1).eval()
    with torch.no_grad():
        out = model(torch.rand(1, 9, 64, 64))
    logits = out["seg_logits"] if hasattr(out, "keys") else out
    assert logits.shape == (1, 1, 64, 64)


def test_big_sam_v2_forward_without_sam():
    model = build_model("big_sam_v2", in_channels=9, num_classes=1, use_sam_prior=False).eval()
    with torch.no_grad():
        out = model(torch.rand(1, 9, 64, 64))
    logits = out["seg_logits"] if hasattr(out, "keys") else out
    assert logits.shape == (1, 1, 64, 64)
