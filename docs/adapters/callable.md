# Recipe: a Tier-2 callable (auth + session + SSE)

When you can't add an `/eval` endpoint and must drive the deployed app as-is —
auth handshake, server-owned session, streaming responses — write a Python
callable. It owns whatever the bot needs; aipsy-bench only ever sees
`message_history -> reply`.

This mirrors the canonical real app (an AI coach: Bearer-token auth, a
`start → message → stream` handshake, server-side history keeping the last N
messages, SSE responses).

```python
import requests
from aipsy_bench.targets import callable_target

class CoachSession:
    """Owns auth + the 3-call handshake + SSE reassembly for one conversation."""

    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {token}"}
        self.conversation_id: str | None = None

    def _ensure_session(self) -> None:
        if self.conversation_id is None:
            r = requests.post(f"{self.base_url}/api/ai-coach/start", headers=self.headers, timeout=30)
            r.raise_for_status()
            self.conversation_id = r.json()["conversationId"]

    def reply(self, messages: list[dict]) -> str:
        # session mode → messages is just the NEW user turn; the server owns history.
        self._ensure_session()
        new_turn = messages[-1]["content"]
        requests.post(
            f"{self.base_url}/api/ai-coach/message",
            headers=self.headers, timeout=30,
            json={"conversationId": self.conversation_id, "message": new_turn},
        ).raise_for_status()

        # reassemble the SSE stream into one reply string
        chunks: list[str] = []
        with requests.get(
            f"{self.base_url}/api/ai-coach/stream",
            headers=self.headers, params={"conversationId": self.conversation_id},
            stream=True, timeout=60,
        ) as s:
            s.raise_for_status()
            for line in s.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    chunks.append(line.removeprefix("data: "))
        return "".join(chunks)


session = CoachSession("https://your-app", token="...")
target = callable_target(session.reply, conversation="session", ref="my-coach")
```

Notes:

- **Errors are run failures.** If `reply` raises (timeout, rate limit, 5xx),
  aipsy-bench classifies the turn as `target_error` and marks the scenario a run
  failure — it is never scored as unsafe. Let exceptions propagate; don't return a
  fake "sorry" string (that *would* be scored).
- **Rate limits** will end a full run fast. If the deployed app caps free users
  (e.g. 10 messages), prefer the [`/eval` endpoint](./eval-endpoint.md) instead.
- **`conversation="session"`** means aipsy-bench sends only the new turn; your
  function reconstructs context server-side. Use `"stateless"` if your callable
  takes the full history each call.
