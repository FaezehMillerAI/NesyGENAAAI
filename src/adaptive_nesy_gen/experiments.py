"""Named ablation configurations matching the paper experiment matrix."""

from __future__ import annotations

from dataclasses import replace

from .schema import PipelineConfig


def ablation_configs(base: PipelineConfig | None = None) -> dict[str, PipelineConfig]:
    base = base or PipelineConfig()
    return {
        "drafting_only": replace(
            base,
            enable_graph=False,
            enable_ltn=False,
            enable_gate=False,
            enable_revision=False,
        ),
        "rag_without_graph_ltn": replace(base, enable_graph=False, enable_ltn=False),
        "report_level_verification": replace(
            base, claim_level=False, always_verify=True, enable_revision=False
        ),
        "adaptive_audit_no_revision": replace(base, enable_revision=False),
        "full_adaptive": base,
        "always_on_claim_verification": replace(base, always_verify=True),
        "adaptive_without_ltn": replace(base, enable_ltn=False),
        "adaptive_without_gate": replace(base, enable_gate=False),
        "strict_fast_path": replace(base, tau_fast=0.90, min_support=3),
        "permissive_fast_path": replace(base, tau_fast=0.75, min_support=1),
    }
