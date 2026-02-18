"""imgeda gate — policy-as-code QA gating command."""

from __future__ import annotations

from typing import Optional

import orjson
import typer
from rich.console import Console

from imgeda.core.gate import evaluate_policy, load_policy
from imgeda.io.manifest_io import read_manifest

console = Console()


def gate(
    manifest: str = typer.Option(..., "-m", "--manifest", help="Path to manifest JSONL"),
    policy_path: str = typer.Option(..., "-p", "--policy", help="Path to policy YAML"),
    output: Optional[str] = typer.Option(None, "-o", "--out", help="Output JSON path"),
) -> None:
    """Evaluate a manifest against a quality policy.

    Exit code 0 = all checks pass, exit code 2 = one or more checks fail.
    """
    _, records = read_manifest(manifest)
    policy = load_policy(policy_path)
    result = evaluate_policy(records, policy)

    # Display results
    console.print(f"\n[bold]Quality Gate ({result.total_images:,} images)[/bold]\n")
    for check in result.checks:
        status = "[green]PASS[/green]" if check.passed else "[red]FAIL[/red]"
        console.print(f"  {status}  {check.name}: {check.observed} (threshold: {check.threshold})")
        if not check.passed and check.sample_paths:
            for p in check.sample_paths[:5]:
                console.print(f"         {p}")
            if len(check.sample_paths) > 5:
                console.print(f"         ... and {len(check.sample_paths) - 5} more")

    console.print()
    if result.passed:
        console.print("[bold green]Gate: PASSED[/bold green]")
    else:
        console.print("[bold red]Gate: FAILED[/bold red]")

    if output:
        data = orjson.dumps(result.to_dict(), option=orjson.OPT_INDENT_2)
        with open(output, "wb") as f:
            f.write(data)
        console.print(f"[green]Saved to {output}[/green]")

    if not result.passed:
        raise typer.Exit(2)
