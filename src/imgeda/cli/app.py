"""Root Typer app with global options."""

from __future__ import annotations

from typing import Optional

import typer

from imgeda.cli.check import check_app
from imgeda.cli.export import export_app
from imgeda.cli.plot import plot_app

app = typer.Typer(
    name="imgeda",
    help="High-performance image dataset exploratory data analysis CLI tool.",
    no_args_is_help=False,
    invoke_without_command=True,
)
app.add_typer(plot_app, name="plot", help="Generate plots from a manifest.")
app.add_typer(check_app, name="check", help="Check for issues in a manifest.")
app.add_typer(export_app, name="export", help="Export manifest to other formats.")


def _version_callback(value: bool) -> None:
    if value:
        from imgeda import __version__

        typer.echo(f"imgeda {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(
        None, "--version", "-V", callback=_version_callback, is_eager=True, help="Show version."
    ),
) -> None:
    """imgeda — Image Dataset EDA Tool.

    Run without arguments for interactive mode, or use a subcommand.
    """
    if ctx.invoked_subcommand is None:
        from imgeda.cli.interactive import run_interactive

        run_interactive()


# Import and register commands
from imgeda.cli.scan import scan  # noqa: E402
from imgeda.cli.report import report, info  # noqa: E402
from imgeda.cli.diff import diff  # noqa: E402
from imgeda.cli.gate import gate  # noqa: E402
from imgeda.cli.annotations import annotations_cmd  # noqa: E402
from imgeda.cli.embed import embed  # noqa: E402

app.command()(scan)
app.command()(info)
app.command()(report)
app.command()(diff)
app.command()(gate)
app.command(name="annotations")(annotations_cmd)
app.command()(embed)
