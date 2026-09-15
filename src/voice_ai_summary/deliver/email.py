"""Send emails via SMTP."""

from __future__ import annotations

import smtplib
from collections.abc import Callable
from email.message import EmailMessage

from ..config import DeliverConfig


def send_email(
    cfg: DeliverConfig,
    password: str,
    subject: str,
    markdown: str,
    *,
    smtp_factory: Callable[..., smtplib.SMTP] | None = None,
) -> None:
    """Send plain-text email digest via SMTP.

    Args:
        cfg: DeliverConfig with smtp_host, smtp_port, smtp_user, email_from, email_to.
        password: SMTP password.
        subject: Email subject.
        markdown: Email body (plain text, UTF-8).
        smtp_factory: Optional injectable SMTP factory (for testing).

    Raises:
        RuntimeError: On SMTP failure or missing configuration.
    """
    if smtp_factory is None:
        smtp_factory = smtplib.SMTP

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg.email_from
    msg["To"] = cfg.email_to
    msg.set_content(markdown, cte="8bit")

    with smtp_factory(cfg.smtp_host, cfg.smtp_port) as smtp:
        smtp.starttls()
        smtp.login(cfg.smtp_user, password)
        smtp.send_message(msg)
