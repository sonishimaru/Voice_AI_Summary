"""CLI subcommands for measuring transcription accuracy. Registered on `app` at import."""

from __future__ import annotations

from pathlib import Path

import typer

from .cli import app

eval_app = typer.Typer(name="eval", help="Measure transcription accuracy on a fixed clip set.")
app.add_typer(eval_app, name="eval")


@eval_app.command("build")
def eval_build(
    day: str = typer.Option(..., "--day", help="Local date YYYY-MM-DD to take clips from."),
    out: Path = typer.Option(..., "--out", help="Where to write the eval set."),  # noqa: B008
    limit: int = typer.Option(40, "--limit", help="Keep at most this many clips."),
) -> None:
    """Seed an eval set from a day's transcript. Correct the text by hand afterwards."""
    from .config import load_config
    from .db import connect
    from .evaluate import build_clips, save_clips
    from .timeutil import local_day_bounds

    cfg = load_config()
    conn = connect(cfg.paths.db_path)
    start, end = local_day_bounds(day, cfg.summarize.timezone)
    clips = build_clips(conn, start, end)[:limit]
    save_clips(out, clips)
    seconds = sum(c.seconds for c in clips)
    typer.echo(f"wrote {len(clips)} clip(s), {seconds / 60:.1f} min of audio, to {out}")
    typer.echo("Correct the `text` of each clip by hand - until then the CER measures nothing.")


@eval_app.command("run")
def eval_run(
    clips_path: Path = typer.Option(..., "--set", help="Eval set written by `eval build`."),  # noqa: B008
    model: str = typer.Option(None, "--model", help="Override [asr] model for this run."),
    beam_size: int = typer.Option(None, "--beam-size", help="Override [asr] beam_size."),
    normalize: str = typer.Option(None, "--normalize", help="Override [asr] normalize."),
    hotwords: bool = typer.Option(
        None, "--hotwords/--no-hotwords", help="Override [asr] use_glossary_hotwords."
    ),
    show: int = typer.Option(0, "--show", help="Print this many of the worst clips."),
) -> None:
    """Re-transcribe the eval set with the current settings and report CER."""
    from .asr import get_backend
    from .config import load_config
    from .db import connect
    from .evaluate import aggregate, load_clips, run_eval

    cfg = load_config()
    if model:
        cfg.asr.model = model
    if beam_size:
        cfg.asr.beam_size = beam_size
    if normalize:
        cfg.asr.normalize = normalize
    if hotwords is not None:
        cfg.asr.use_glossary_hotwords = hotwords

    conn = connect(cfg.paths.db_path)
    clips = load_clips(clips_path)
    backend = get_backend(cfg)
    results = run_eval(conn, cfg, clips, backend)
    stats = aggregate(results)

    typer.echo(
        f"model={cfg.asr.model} beam={cfg.asr.beam_size} normalize={cfg.asr.normalize} "
        f"glossary_hotwords={cfg.asr.use_glossary_hotwords}"
    )
    typer.echo(
        f"clips={stats['clips']} audio={stats['seconds'] / 60:.1f}min "
        f"ref_chars={stats['ref_chars']} edits={stats['edits']} "
        f"empty={stats['empty']}  CER={stats['cer']:.3f}"
    )
    for r in sorted(results, key=lambda r: -r.cer)[:show]:
        typer.echo(f"  CER={r.cer:.2f} 参照: {r.clip.text}")
        typer.echo(f"           出力: {r.hypothesis}")
