"""Import Slack messages for glossary extraction (`glossary.py`).

Two sources, both read with a Slack *user* token (`xoxp-…`) from `$VAS_SLACK_USER_TOKEN`:

- `import_from_slack`: the user's own messages via `search.messages` (scope `search:read`;
  not available to bot tokens).
- `import_channel_messages`: everything posted in the channels the user is a member of,
  via `users.conversations` + `conversations.history` (scopes `channels:read`,
  `groups:read`, `channels:history`, `groups:history`).

Nothing here reaches the network except through the `httpx.Client` passed in, which keeps
this testable with `httpx.MockTransport`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx

_BASE_URL = "https://slack.com/api"


def import_from_slack(
    client_http: httpx.Client, token: str, *, days: int = 90, limit: int = 1500
) -> list[str]:
    """Fetch up to `limit` of the user's own Slack messages from the last `days` days.

    Calls `auth.test` to find the user's own id, then paginates `search.messages` for
    `from:<@user_id> after:YYYY-MM-DD` (`count=100` per page) until `limit` messages are
    collected or there are no more pages. Raises `RuntimeError` if either call reports
    `ok: false`.
    """
    headers = {"Authorization": f"Bearer {token}"}

    auth_resp = client_http.post(f"{_BASE_URL}/auth.test", headers=headers)
    auth_data = auth_resp.json()
    if not auth_data.get("ok"):
        raise RuntimeError(f"Slack auth.test failed: {auth_data.get('error')}")
    user_id = auth_data["user_id"]

    after = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%d")
    query = f"from:<@{user_id}> after:{after}"

    messages: list[str] = []
    page = 1
    while len(messages) < limit:
        resp = client_http.get(
            f"{_BASE_URL}/search.messages",
            headers=headers,
            params={"query": query, "count": 100, "page": page},
        )
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"Slack search.messages failed: {data.get('error')}")

        matches = data.get("messages", {}).get("matches", [])
        if not matches:
            break
        for match in matches:
            messages.append(match.get("text", ""))
            if len(messages) >= limit:
                break

        pagination = data.get("messages", {}).get("pagination", {})
        page_count = pagination.get("page_count", page)
        if page >= page_count:
            break
        page += 1

    return messages[:limit]


# Message subtypes that carry no vocabulary (joins, topic changes, bot chatter, ...).
_SKIPPED_SUBTYPES = {
    "channel_join",
    "channel_leave",
    "channel_topic",
    "channel_purpose",
    "channel_name",
    "bot_message",
    "tombstone",
}


def list_user_channels(
    client_http: httpx.Client,
    token: str,
    *,
    types: str = "public_channel,private_channel",
) -> list[dict]:
    """Return `{"id", "name"}` for every non-archived channel the token's user belongs to."""
    headers = {"Authorization": f"Bearer {token}"}
    channels: list[dict] = []
    cursor: str | None = None
    while True:
        params = {"types": types, "exclude_archived": "true", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        data = client_http.get(
            f"{_BASE_URL}/users.conversations", headers=headers, params=params
        ).json()
        if not data.get("ok"):
            raise RuntimeError(f"Slack users.conversations failed: {data.get('error')}")
        channels.extend(
            {"id": c["id"], "name": c.get("name", c["id"])} for c in data.get("channels", [])
        )
        cursor = data.get("response_metadata", {}).get("next_cursor") or None
        if not cursor:
            return channels


def import_channel_messages(
    client_http: httpx.Client,
    token: str,
    *,
    days: int = 90,
    per_channel: int = 300,
    limit: int = 5000,
    channels: list[str] | None = None,
) -> list[str]:
    """Fetch recent human-written messages from the user's channels (all authors).

    `channels` restricts the scan to those channel names (without `#`). Each channel
    contributes at most `per_channel` messages, newest first, and `limit` caps the total.
    Every channel's messages are preceded by a `#name` header line so the extractor can
    treat the channel name itself as vocabulary.
    """
    headers = {"Authorization": f"Bearer {token}"}
    oldest = (datetime.now(UTC) - timedelta(days=days)).timestamp()
    wanted = {c.lstrip("#") for c in channels} if channels else None

    out: list[str] = []
    for channel in list_user_channels(client_http, token):
        if wanted is not None and channel["name"] not in wanted:
            continue
        if len(out) >= limit:
            break
        texts: list[str] = []
        cursor: str | None = None
        while len(texts) < per_channel and len(out) + len(texts) < limit:
            params = {"channel": channel["id"], "oldest": f"{oldest:.6f}", "limit": 200}
            if cursor:
                params["cursor"] = cursor
            data = client_http.get(
                f"{_BASE_URL}/conversations.history", headers=headers, params=params
            ).json()
            if not data.get("ok"):
                raise RuntimeError(
                    f"Slack conversations.history failed for #{channel['name']}: "
                    f"{data.get('error')}"
                )
            for msg in data.get("messages", []):
                if msg.get("subtype") in _SKIPPED_SUBTYPES or msg.get("bot_id"):
                    continue
                text = (msg.get("text") or "").strip()
                if text:
                    texts.append(text)
                if len(texts) >= per_channel or len(out) + len(texts) >= limit:
                    break
            cursor = data.get("response_metadata", {}).get("next_cursor") or None
            if not cursor or not data.get("has_more"):
                break
        if texts:
            out.append(f"#{channel['name']}")
            out.extend(texts)
    return out
