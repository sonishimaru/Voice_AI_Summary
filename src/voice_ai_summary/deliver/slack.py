"""Send Slack messages via incoming webhooks."""

from __future__ import annotations

import re

import httpx


def send_slack(webhook_url: str, markdown: str, *, timeout: float = 15.0) -> None:
    """Post markdown to Slack via incoming webhook.

    Converts Markdown to Slack mrkdwn, splits into chunks if needed (Slack limit ~40k,
    split on paragraph boundaries at ≤3500 chars per chunk), and posts sequentially.

    Raises RuntimeError with status and body on non-2xx response.
    """
    chunks = _chunk_markdown(markdown, max_chars=3500)
    for chunk in chunks:
        text = _markdown_to_mrkdwn(chunk)
        payload = {"text": text}
        resp = httpx.post(webhook_url, json=payload, timeout=timeout)
        if not resp.is_success:
            raise RuntimeError(f"Slack webhook failed with {resp.status_code}: {resp.text}")  # noqa: E501


def _markdown_to_mrkdwn(markdown: str) -> str:
    """Convert Markdown to Slack mrkdwn.

    - `# title` / `## title` → `*title*` (bold)
    - `- bullet` → `- bullet` (unchanged)
    - `**text**` → `*text*` (bold)
    """
    # Convert # and ## headings to bold lines
    text = re.sub(r"^#{1,2}\s+(.+)$", r"*\1*", markdown, flags=re.MULTILINE)
    # Convert **text** to *text*
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)
    return text


def _chunk_markdown(markdown: str, max_chars: int = 3500) -> list[str]:
    """Split markdown into chunks on paragraph boundaries.

    Aim for max_chars per chunk. Paragraphs are separated by blank lines.
    """
    if len(markdown) <= max_chars:
        return [markdown]

    # Split into paragraphs (separated by 2+ newlines or just by double newline)
    paragraphs = re.split(r"\n\n+", markdown.strip())
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_size = 0

    for para in paragraphs:
        para_size = len(para) + 2  # +2 for the newlines we'll add between paragraphs
        if current_size + para_size > max_chars and current_chunk:
            # Start a new chunk
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_size = para_size
        else:
            current_chunk.append(para)
            current_size += para_size

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks
