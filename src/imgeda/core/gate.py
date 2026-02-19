"""Policy-as-code QA gating — evaluation logic."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from imgeda.core.duplicates import find_exact_duplicates
from imgeda.models.manifest import ImageRecord
from imgeda.models.policy import Policy


@dataclass(slots=True)
class CheckResult:
    name: str = ""
    threshold: float = 0.0
    observed: float = 0.0
    passed: bool = True
    sample_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class GateResult:
    passed: bool = True
    checks: list[CheckResult] = field(default_factory=list)
    total_images: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "total_images": self.total_images,
            "checks": [c.to_dict() for c in self.checks],
        }


def evaluate_policy(records: list[ImageRecord], policy: Policy) -> GateResult:
    """Evaluate a manifest against a policy. Returns structured gate result."""
    total = len(records)
    result = GateResult(total_images=total)

    if total == 0:
        result.checks.append(
            CheckResult(
                name="min_images_total",
                threshold=float(policy.min_images_total),
                observed=0.0,
                passed=False,
            )
        )
        result.passed = False
        return result

    valid = [r for r in records if not r.is_corrupt]

    # min_images_total
    result.checks.append(
        CheckResult(
            name="min_images_total",
            threshold=float(policy.min_images_total),
            observed=float(total),
            passed=total >= policy.min_images_total,
        )
    )

    # max_corrupt_pct
    corrupt = [r for r in records if r.is_corrupt]
    corrupt_pct = len(corrupt) / total * 100
    result.checks.append(
        CheckResult(
            name="max_corrupt_pct",
            threshold=policy.max_corrupt_pct,
            observed=round(corrupt_pct, 2),
            passed=corrupt_pct <= policy.max_corrupt_pct,
            sample_paths=[r.path for r in corrupt[:10]],
        )
    )

    # max_overexposed_pct
    overexposed = [r for r in records if r.is_overexposed]
    overexposed_pct = len(overexposed) / total * 100
    result.checks.append(
        CheckResult(
            name="max_overexposed_pct",
            threshold=policy.max_overexposed_pct,
            observed=round(overexposed_pct, 2),
            passed=overexposed_pct <= policy.max_overexposed_pct,
            sample_paths=[r.path for r in overexposed[:10]],
        )
    )

    # max_underexposed_pct (dark images)
    dark = [r for r in records if r.is_dark]
    dark_pct = len(dark) / total * 100
    result.checks.append(
        CheckResult(
            name="max_underexposed_pct",
            threshold=policy.max_underexposed_pct,
            observed=round(dark_pct, 2),
            passed=dark_pct <= policy.max_underexposed_pct,
            sample_paths=[r.path for r in dark[:10]],
        )
    )

    # max_duplicate_pct
    dup_groups = find_exact_duplicates(records)
    dup_count = sum(len(v) - 1 for v in dup_groups.values())
    dup_pct = dup_count / total * 100
    dup_paths: list[str] = []
    for group in dup_groups.values():
        dup_paths.extend(r.path for r in group[1:])
        if len(dup_paths) >= 10:
            break
    result.checks.append(
        CheckResult(
            name="max_duplicate_pct",
            threshold=policy.max_duplicate_pct,
            observed=round(dup_pct, 2),
            passed=dup_pct <= policy.max_duplicate_pct,
            sample_paths=dup_paths[:10],
        )
    )

    # max_blurry_pct
    if policy.max_blurry_pct < 100.0:
        blurry = [r for r in records if r.is_blurry]
        blurry_pct = len(blurry) / total * 100
        result.checks.append(
            CheckResult(
                name="max_blurry_pct",
                threshold=policy.max_blurry_pct,
                observed=round(blurry_pct, 2),
                passed=blurry_pct <= policy.max_blurry_pct,
                sample_paths=[r.path for r in blurry[:10]],
            )
        )

    # max_artifact_pct
    if policy.max_artifact_pct < 100.0:
        artifacts = [r for r in records if r.has_border_artifact]
        artifact_pct = len(artifacts) / total * 100
        result.checks.append(
            CheckResult(
                name="max_artifact_pct",
                threshold=policy.max_artifact_pct,
                observed=round(artifact_pct, 2),
                passed=artifact_pct <= policy.max_artifact_pct,
                sample_paths=[r.path for r in artifacts[:10]],
            )
        )

    # min_width
    if policy.min_width > 0:
        small = [r for r in valid if r.width < policy.min_width]
        result.checks.append(
            CheckResult(
                name="min_width",
                threshold=float(policy.min_width),
                observed=float(min((r.width for r in valid), default=0)),
                passed=len(small) == 0,
                sample_paths=[r.path for r in small[:10]],
            )
        )

    # min_height
    if policy.min_height > 0:
        short = [r for r in valid if r.height < policy.min_height]
        result.checks.append(
            CheckResult(
                name="min_height",
                threshold=float(policy.min_height),
                observed=float(min((r.height for r in valid), default=0)),
                passed=len(short) == 0,
                sample_paths=[r.path for r in short[:10]],
            )
        )

    # allowed_formats
    if policy.allowed_formats:
        allowed = {f.upper() for f in policy.allowed_formats}
        bad_fmt = [r for r in valid if r.format.upper() not in allowed]
        result.checks.append(
            CheckResult(
                name="allowed_formats",
                threshold=0.0,
                observed=float(len(bad_fmt)),
                passed=len(bad_fmt) == 0,
                sample_paths=[r.path for r in bad_fmt[:10]],
            )
        )

    # max_aspect_ratio
    if policy.max_aspect_ratio > 0:
        extreme = [r for r in valid if r.aspect_ratio > policy.max_aspect_ratio]
        result.checks.append(
            CheckResult(
                name="max_aspect_ratio",
                threshold=policy.max_aspect_ratio,
                observed=round(max((r.aspect_ratio for r in valid), default=0.0), 2),
                passed=len(extreme) == 0,
                sample_paths=[r.path for r in extreme[:10]],
            )
        )

    result.passed = all(c.passed for c in result.checks)
    return result


def load_policy(path: str) -> Policy:
    """Load a Policy from a YAML file."""
    import yaml  # type: ignore[import-untyped]

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        return Policy()

    kwargs = {k: v for k, v in data.items() if k in Policy.__dataclass_fields__}
    return Policy(**kwargs)
