"""Import the user's own Slack messages, for glossary extraction (`glossary.py`).

Uses a Slack *user* token (`xoxp-…`, scope `search:read`) from `$VAS_SLACK_USER_TOKEN` -
`search.messages` is not available to bot tokens. Nothing here reaches the network except
through the `httpx.Client` passed in, which keeps this testable with `httpx.MockTransport`.
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
