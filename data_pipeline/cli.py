"""CLI entry point for the HPE-AFF data pipeline."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import click
import structlog

log = structlog.get_logger()


def _data_root() -> Path:
    return Path(os.getenv("DATA_ROOT", "./data"))


def _state_path(data_root: Path) -> Path:
    return data_root / "pipeline_state.json"


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """HPE-AFF Data Engineering Pipeline CLI."""
    import structlog
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(),
    )


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--all", "run_all", is_flag=True, default=False, help="Run all stages in order")
@click.option(
    "--stage",
    type=click.Choice(["ingest", "order", "consolidate", "generate", "test"]),
    default=None,
    help="Run a single stage",
)
@click.option("--seed", default=42, show_default=True, help="Random seed")
def run(run_all: bool, stage: str | None, seed: int) -> None:
    """Run pipeline stages."""
    if not run_all and stage is None:
        raise click.UsageError("Specify --all or --stage <stage>")

    from data_pipeline import storage

    data_root = _data_root()
    state_path = _state_path(data_root)

    def _is_done(s: str) -> bool:
        return storage.is_stage_complete(s, state_path)

    def _mark_done(s: str, **kwargs: object) -> None:
        storage.mark_stage_complete(s, state_path, **kwargs)

    stages_to_run = (
        ["ingest", "order", "consolidate", "generate", "test"] if run_all else [stage]
    )

    records: list = []

    for stg in stages_to_run:
        if run_all and _is_done(stg):
            click.echo(f"[skip] {stg} already complete")
            # Load existing records for downstream stages
            if stg in ("ingest", "order") and not records:
                try:
                    import pandas as pd
                    parquet_path = data_root / "consolidated" / "master.parquet"
                    if parquet_path.exists():
                        from data_pipeline import loader
                        records = loader._load_all_records(data_root)
                except Exception:
                    pass
            continue

        click.echo(f"[run] {stg}...")
        t0 = time.monotonic()

        if stg == "ingest":
            from data_pipeline.ingest import funsd, xfund, vrdu, rvlcdip

            all_recs: list = []
            for mod, name in [(funsd, "funsd"), (xfund, "xfund"), (vrdu, "vrdu"), (rvlcdip, "rvlcdip")]:
                try:
                    recs = mod.ingest(data_root, seed)
                    all_recs.extend(recs)
                    click.echo(f"  {name}: {len(recs)} records")
                except Exception as exc:
                    click.echo(f"  {name}: FAILED — {exc}", err=True)

            records = all_recs
            _mark_done("ingest", records=len(records))

        elif stg == "order":
            if not records:
                click.echo("  No records from ingest — run ingest first", err=True)
                return
            from data_pipeline import order
            records = order.run(records, seed)
            _mark_done("order", records=len(records))

        elif stg == "consolidate":
            if not records:
                click.echo("  No records — run ingest+order first", err=True)
                return
            from data_pipeline import consolidate
            consolidate.run(records, data_root, seed)

        elif stg == "generate":
            from data_pipeline.generate import synthetic
            new_recs = synthetic.run(data_root, seed)
            click.echo(f"  Generated {len(new_recs)} records")
            # Append to consolidated dataset
            if new_recs and records:
                all_recs = records + new_recs
                from data_pipeline import consolidate
                consolidate.run(all_recs, data_root, seed)
            _mark_done("generate", new_records=len(new_recs))

        elif stg == "test":
            import subprocess
            result = subprocess.run(
                ["pytest", "data_pipeline/tests/", "-v", "--tb=short"],
                cwd=str(Path(__file__).parent.parent),
            )
            if result.returncode == 0:
                _mark_done("test")
            else:
                click.echo("Tests FAILED", err=True)
                raise SystemExit(1)

        elapsed = time.monotonic() - t0
        click.echo(f"  done in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@cli.command()
def status() -> None:
    """Show which pipeline stages are complete."""
    data_root = _data_root()
    state_path = _state_path(data_root)

    if not state_path.exists():
        click.echo("Pipeline not started — no pipeline_state.json found.")
        return

    with open(state_path) as fh:
        state = json.load(fh)

    stages = ["ingest", "order", "consolidate", "generate", "test"]
    for stg in stages:
        info = state.get(stg, {})
        s = info.get("status", "pending")
        ts = info.get("completed_at", "")
        icon = "✓" if s == "complete" else "○"
        click.echo(f"  {icon} {stg:<12} {s:<10} {ts}")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

@cli.command()
def report() -> None:
    """Print manifest summary."""
    data_root = _data_root()
    manifest_path = data_root / "consolidated" / "manifest.json"

    if not manifest_path.exists():
        click.echo("No manifest found. Run consolidate stage first.")
        return

    from data_pipeline import storage
    m = storage.read_manifest(manifest_path)

    click.echo(f"\nHPE-AFF Dataset Report")
    click.echo(f"Created: {m.get('created_at', 'unknown')}")
    click.echo(f"Seed:    {m.get('seed', 'unknown')}")
    click.echo(f"Total:   {m.get('total_documents', 0):,} documents\n")

    click.echo("By source:")
    for src, counts in m.get("by_source", {}).items():
        total = counts.get("total", 0)
        if total == 0:
            continue
        train = counts.get("train", 0)
        val = counts.get("val", 0)
        test = counts.get("test", 0)
        click.echo(f"  {src:<22} total={total:>5}  train={train:>5}  val={val:>4}  test={test:>4}")

    click.echo("\nBy quality tier:")
    for tier, count in m.get("by_quality_tier", {}).items():
        click.echo(f"  {tier:<22} {count:>5}")

    click.echo("\nBy split:")
    for split, count in m.get("by_split", {}).items():
        click.echo(f"  {split:<22} {count:>5}")

    vrdu_gt = m.get("vrdu_with_gt_payload", 0)
    click.echo(f"\nVRDU records with gt_payload: {vrdu_gt:,}")


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--split", required=True, type=click.Choice(["train", "val", "test"]))
@click.option("--output", required=True, type=click.Path(), help="Output directory")
def export(split: str, output: str) -> None:
    """Export a split to a target directory."""
    data_root = _data_root()
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    from data_pipeline import storage
    parquet_path = data_root / "consolidated" / "master.parquet"

    if not parquet_path.exists():
        click.echo("No dataset found. Run the pipeline first.", err=True)
        raise SystemExit(1)

    df = storage.read_parquet(parquet_path)
    split_df = df[df["split"] == split]

    # Write filtered parquet
    out_parquet = out_dir / f"{split}_master.parquet"
    split_df.to_parquet(out_parquet, index=False)

    # Copy field JSONs
    fields_src = data_root / "consolidated" / "fields"
    fields_dst = out_dir / "fields"
    fields_dst.mkdir(exist_ok=True)

    copied = 0
    for _, row in split_df.iterrows():
        src_file = fields_src / f"{row['source']}_{row['doc_id']}.json"
        src_file = Path(str(src_file).replace("/", "_").replace("\\", "_"))
        # Search by constructed name
        name = f"{row['source']}_{row['doc_id']}.json".replace("/", "_").replace("\\", "_")
        src_file = fields_src / name
        if src_file.exists():
            shutil.copy2(src_file, fields_dst / name)
            copied += 1

    click.echo(f"Exported {len(split_df)} records ({copied} field JSONs) to {out_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
