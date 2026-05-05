#!/usr/bin/env python3
"""Fetch new videos from a YouTube playlist, summarize them with Claude, and
prepend the rendered HTML blocks into index.html."""
from __future__ import annotations

import html
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)
from youtube_transcript_api._errors import CouldNotRetrieveTranscript
from youtube_transcript_api.proxies import WebshareProxyConfig

try:
    from youtube_transcript_api import AgeRestricted
except ImportError:  # older library versions
    AgeRestricted = None  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
HTML_PATH = ROOT / "index.html"
SKIP_LIST_PATH = ROOT / "skipped.json"

# Reasons that mean "this video will never be summarizable" — seen once, retire forever.
# Kept narrow on purpose: transient/network/quota errors stay retryable.
PERMANENT_FATAL_REASONS = frozenset({
    "TranscriptsDisabled",
    "AgeRestricted",
    "VideoUnplayable",
    "VideoUnavailable",
    "NoTranscriptInAnyLanguage",
    "EmptyTranscript",
    "NoMetadata",
})

PLAYLIST_IDS = [p.strip() for p in os.environ["PLAYLIST_IDS"].split(",") if p.strip()]
YT_KEY = os.environ["YOUTUBE_API_KEY"]
ANTHROPIC_KEY = os.environ["ANTHROPIC_API_KEY"]

MODEL = "claude-haiku-4-5"
MAX_TRANSCRIPT_CHARS = 180_000

SYSTEM_PROMPT = """You produce HTML summaries of YouTube videos in a very specific format.

# Output format

Exactly this structure and nothing else — no preamble, no code fences, no trailing commentary:

<p>* <strong>{Bold lead-in phrase}</strong> — {One to three substantive sentences explaining the point with specifics.}</p>
(5 to 7 of these bullets)

<blockquote><strong>Memorable quote:</strong> "{Direct quote from the speaker.}"</blockquote>

# Format rules

- Exactly 5 to 7 bullets. Pick the most important, surprising, and non-obvious claims — not a table of contents.
- Each bullet starts with `<p>* <strong>LEAD-IN</strong> — ` where LEAD-IN is a crisp phrase (4–12 words) that states the claim itself (e.g. "Inflation will be selective, not broad"), not a topic label ("Discussion of inflation").
- The body after the em-dash explains the argument with evidence, numbers, or context from the transcript so a reader who never watched can understand the point.
- Close all `<p>` and `<strong>` tags correctly. Use straight quotes, a real em-dash character ` — `, and ASCII only where possible.
- The quote must be verbatim from the transcript — do not paraphrase or clean it up. Pick the single line that best captures a key insight. Keep it under ~60 words.
- Output the HTML only.

# Example bullet style (shape, not content to reuse)

<p>* <strong>AI's impact on employment will be gradual, not catastrophic</strong> — Alden argues AI adoption resembles past automation waves that suppressed wages and prices through productivity rather than mass firings: one person overseeing AI agents will replace multiple workers, creating deflationary pressure on white-collar services while reducing hiring overall.</p>

<p>* <strong>Inflation will be selective, driven by scarcity not broad price rises</strong> — Money creation won't distribute evenly. Bitcoin, gold, real estate, and skilled labor will appreciate as scarce assets, while AI-driven productivity keeps white-collar services and commodity-adjacent sectors relatively cheap.</p>

<blockquote><strong>Memorable quote:</strong> "If you're holding Bitcoin then apart from security fees you're not really getting diluted at all over time."</blockquote>

The real output has 5–7 bullets, not 2."""

VIDEO_ID_RE = re.compile(r"<strong>Video ID:</strong>\s*([A-Za-z0-9_-]{6,})")
HEADER_RE = re.compile(
    r'<p>\d+\s*videos\s*(?:&nbsp;|\s)*·(?:&nbsp;|\s)*Last updated:[^<]*</p>'
)
MAIN_OPEN_RE = re.compile(r"(<main>\s*)")


def existing_video_ids(html_text: str) -> set[str]:
    return set(VIDEO_ID_RE.findall(html_text))


def load_skip_list() -> dict[str, dict]:
    if not SKIP_LIST_PATH.exists():
        return {}
    try:
        return json.loads(SKIP_LIST_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"warning: skipped.json unreadable ({e}); starting empty", file=sys.stderr)
        return {}


def save_skip_list(skip_list: dict[str, dict]) -> None:
    SKIP_LIST_PATH.write_text(
        json.dumps(skip_list, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def fetch_playlist_items(yt, playlist_id: str) -> list[str]:
    ids: list[str] = []
    page = None
    while True:
        req = yt.playlistItems().list(
            part="contentDetails,snippet",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=page,
        )
        resp = req.execute()
        for item in resp.get("items", []):
            title = item["snippet"].get("title", "")
            if title in ("Deleted video", "Private video"):
                continue
            vid = item["contentDetails"].get("videoId")
            if vid:
                ids.append(vid)
        page = resp.get("nextPageToken")
        if not page:
            break
    return ids


def fetch_all_playlists(yt) -> list[str]:
    seen: set[str] = set()
    combined: list[str] = []
    for pid in PLAYLIST_IDS:
        try:
            ids = fetch_playlist_items(yt, pid)
        except HttpError as e:
            print(f"playlist {pid} error: {e}", file=sys.stderr)
            continue
        print(f"  {pid}: {len(ids)} videos")
        for v in ids:
            if v not in seen:
                seen.add(v)
                combined.append(v)
    return combined


def fetch_video_metadata(yt, video_id: str) -> dict | None:
    resp = yt.videos().list(
        part="snippet,contentDetails,statistics",
        id=video_id,
    ).execute()
    items = resp.get("items", [])
    if not items:
        return None
    v = items[0]
    snippet = v["snippet"]
    stats = v.get("statistics", {})
    return {
        "id": video_id,
        "title": snippet["title"],
        "channel": snippet["channelTitle"],
        "published_iso": snippet["publishedAt"],
        "duration_iso": v["contentDetails"].get("duration", "PT0M"),
        "view_count": int(stats["viewCount"]) if "viewCount" in stats else None,
    }


_yt_api: YouTubeTranscriptApi | None = None


def get_yt_api() -> YouTubeTranscriptApi:
    global _yt_api
    if _yt_api is not None:
        return _yt_api
    user = os.environ.get("WEBSHARE_USERNAME", "").strip()
    pw = os.environ.get("WEBSHARE_PASSWORD", "").strip()
    if user and pw:
        print(f"Using Webshare rotating proxy (user {user[:4]}…)")
        _yt_api = YouTubeTranscriptApi(
            proxy_config=WebshareProxyConfig(proxy_username=user, proxy_password=pw)
        )
    else:
        print("WARNING: no Webshare creds set — transcripts may be blocked")
        _yt_api = YouTubeTranscriptApi()
    return _yt_api


def _snippets_to_text(fetched) -> str:
    parts = []
    for s in fetched:
        t = getattr(s, "text", None) or (s.get("text") if isinstance(s, dict) else None)
        if t:
            parts.append(t)
    text = " ".join(parts)
    return re.sub(r"\s+", " ", text).strip()


FATAL_TRANSCRIPT_ERRORS: tuple = (TranscriptsDisabled, VideoUnavailable)
if AgeRestricted is not None:
    FATAL_TRANSCRIPT_ERRORS = FATAL_TRANSCRIPT_ERRORS + (AgeRestricted,)


def _try_fetch_once(api, video_id):
    try:
        return api.fetch(video_id, languages=["en", "en-US", "en-GB"]), None
    except FATAL_TRANSCRIPT_ERRORS as e:
        return None, ("fatal", e)
    except NoTranscriptFound as e:
        return None, ("no_english", e)
    except Exception as e:
        return None, ("transient", e)


def fetch_transcript(video_id: str, attempts: int = 3) -> tuple[str | None, str | None]:
    """Returns (transcript, None) on success, (None, reason) on skip.
    Reason is the exception class name (e.g. 'TranscriptsDisabled',
    'AgeRestricted', 'VideoUnplayable') or 'NoTranscriptInAnyLanguage' /
    'TransientFailure' for non-class cases."""
    api = get_yt_api()
    last_err = None
    for i in range(attempts):
        fetched, err = _try_fetch_once(api, video_id)
        if fetched is not None:
            text = _snippets_to_text(fetched) or None
            return text, (None if text else "EmptyTranscript")
        kind, last_err = err
        if kind == "fatal":
            reason = type(last_err).__name__
            print(f"  [{video_id}] fatal: {reason}", file=sys.stderr)
            return None, reason
        if kind == "no_english":
            try:
                for t in api.list(video_id):
                    try:
                        fetched = t.fetch()
                        text = _snippets_to_text(fetched) or None
                        return text, (None if text else "EmptyTranscript")
                    except Exception:
                        continue
            except Exception as e:
                print(f"  [{video_id}] list-fallback failed: {type(e).__name__}", file=sys.stderr)
            print(f"  [{video_id}] no transcript in any language", file=sys.stderr)
            return None, "NoTranscriptInAnyLanguage"
        if i < attempts - 1:
            wait = 1.5 * (i + 1) + random.uniform(0, 1.5)
            print(
                f"  [{video_id}] transient ({type(last_err).__name__}); retry {i+1}/{attempts-1} in {wait:.1f}s",
                file=sys.stderr,
            )
            time.sleep(wait)
    reason = type(last_err).__name__ if last_err else "TransientFailure"
    print(
        f"  [{video_id}] failed after {attempts}: {reason}: {str(last_err)[:120]}",
        file=sys.stderr,
    )
    return None, reason


def parse_duration_minutes(iso: str) -> int:
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso or "")
    if not m:
        return 0
    h, mm, s = (int(x) if x else 0 for x in m.groups())
    total = h * 60 + mm + (1 if s >= 30 else 0)
    return total or 1


def format_pub_date(iso: str) -> tuple[str, str]:
    # returns ("2026-04-15", "Apr 15, 2026")
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    iso_date = dt.strftime("%Y-%m-%d")
    pretty = f"{dt.strftime('%b')} {dt.day}, {dt.year}"
    return iso_date, pretty


def format_views(n: int | None) -> str:
    if n is None:
        return "—"
    if n >= 1000:
        # round to nearest 100 for consistency with existing "7,200" / "187,000" style
        rounded = int(round(n, -2))
        return f"{rounded:,}"
    return f"{n:,}"


def generate_summary(client: anthropic.Anthropic, meta: dict, transcript: str) -> str:
    user = (
        f"Title: {meta['title']}\n"
        f"Channel: {meta['channel']}\n"
        f"Published: {meta['published_iso'][:10]}\n\n"
        f"Transcript:\n{transcript[:MAX_TRANSCRIPT_CHARS]}"
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
    )
    parts = [b.text for b in resp.content if b.type == "text"]
    out = "".join(parts).strip()
    # Strip accidental code fences if the model adds them.
    if out.startswith("```"):
        out = re.sub(r"^```[a-zA-Z]*\n?", "", out)
        out = re.sub(r"\n?```$", "", out).strip()
    return out


def render_block(meta: dict, body: str) -> str:
    iso_date, pretty_date = format_pub_date(meta["published_iso"])
    duration_min = parse_duration_minutes(meta["duration_iso"])
    views = format_views(meta["view_count"])
    title_safe = html.escape(meta["title"], quote=False)
    channel_safe = html.escape(meta["channel"], quote=False)
    return (
        f'        <div class="video-block"><div class="video-date">{pretty_date}</div><h2>{title_safe}</h2>\n'
        f"\n"
        f"<strong>Video ID:</strong> {meta['id']}\n"
        f"<strong>Channel:</strong> {channel_safe}\n"
        f"<strong>Published:</strong> {iso_date}\n"
        f"<strong>Duration:</strong> ~{duration_min} minutes\n"
        f"<strong>Views:</strong> {views}\n"
        f"<strong>URL:</strong> https://www.youtube.com/watch?v={meta['id']}\n"
        f"\n"
        f"{body}\n"
        f"\n"
        f"<hr></div>"
    )


def insert_blocks(html_text: str, blocks: list[str]) -> str:
    if not blocks:
        return html_text
    joined = "\n".join(blocks) + "\n"
    m = MAIN_OPEN_RE.search(html_text)
    if not m:
        raise RuntimeError("Could not find <main> opening tag in index.html")
    return html_text[: m.end()] + joined + html_text[m.end():]


def update_header(html_text: str, total_videos: int) -> str:
    now = datetime.now(timezone.utc).astimezone()
    stamp = f"{now.strftime('%B')} {now.day}, {now.strftime('%Y at %I:%M %p')}"
    replacement = (
        f'<p>{total_videos} videos &nbsp;·&nbsp; Last updated: {stamp}</p>'
    )
    new, n = HEADER_RE.subn(replacement, html_text, count=1)
    if n == 0:
        print("warning: header line not found; leaving untouched", file=sys.stderr)
    return new


def notify_skipped_on_slack(skipped: list[tuple[dict | None, str, str]]) -> None:
    """Post a skip report to Slack if SLACK_WEBHOOK_URL is set and we skipped
    any videos. Used when the summarizer produces nothing — gives visibility
    into what content is backlogged and why."""
    webhook = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    if not webhook or not skipped:
        return
    import urllib.request
    lines = [
        f"📹 YouTube summaries: 0 new summaries today, but {len(skipped)} "
        f"video(s) couldn't be processed:"
    ]
    for meta, vid, reason in skipped[:10]:
        title = (meta["title"][:70] if meta else f"(video {vid})")
        channel = f" — {meta['channel']}" if meta else ""
        url = f"https://youtu.be/{vid}"
        lines.append(f"• <{url}|{title}>{channel} — `{reason}`")
    if len(skipped) > 10:
        lines.append(f"_…and {len(skipped) - 10} more._")
    payload = json.dumps({"text": "\n".join(lines)}).encode()
    req = urllib.request.Request(
        webhook,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=10).read()
        print(f"Posted Slack skip report ({len(skipped)} videos)")
    except Exception as e:
        print(f"Slack webhook failed: {type(e).__name__}: {e}", file=sys.stderr)


def main() -> int:
    html_text = HTML_PATH.read_text(encoding="utf-8")
    have = existing_video_ids(html_text)
    print(f"Already summarized: {len(have)} videos")

    skip_list = load_skip_list()
    initial_skip_count = len(skip_list)
    print(f"Permanent skip list: {initial_skip_count} videos")

    yt = build("youtube", "v3", developerKey=YT_KEY, cache_discovery=False)
    print(f"Fetching {len(PLAYLIST_IDS)} playlists:")
    playlist_ids = fetch_all_playlists(yt)
    print(f"Playlist total (deduped): {len(playlist_ids)} videos")
    if not playlist_ids:
        print("No playlist items fetched — aborting.", file=sys.stderr)
        return 1

    # Preserve playlist order; newest-first assumed, but we process in playlist
    # order so the final prepend keeps the most-recent-first convention.
    to_process = [v for v in playlist_ids if v not in have and v not in skip_list]
    print(f"New to summarize: {len(to_process)}")

    if not to_process:
        return 0

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    summarized: list[tuple[dict, str]] = []
    skipped: list[tuple[dict | None, str, str]] = []  # (meta, video_id, reason)

    for vid in to_process:
        print(f"- {vid}")
        try:
            meta = fetch_video_metadata(yt, vid)
        except HttpError as e:
            print(f"  metadata error: {e}", file=sys.stderr)
            skipped.append((None, vid, f"MetadataError:{type(e).__name__}"))
            continue
        if not meta:
            print("  no metadata; skipping")
            skipped.append((None, vid, "NoMetadata"))
            skip_list[vid] = {
                "reason": "NoMetadata",
                "title": None,
                "channel": None,
                "first_seen": today,
            }
            continue
        transcript, skip_reason = fetch_transcript(vid)
        if not transcript:
            reason = skip_reason or "Unknown"
            print(f"  no transcript available; skipping ({reason})")
            skipped.append((meta, vid, reason))
            if reason in PERMANENT_FATAL_REASONS:
                skip_list[vid] = {
                    "reason": reason,
                    "title": meta["title"],
                    "channel": meta["channel"],
                    "first_seen": today,
                }
            continue
        try:
            body = generate_summary(client, meta, transcript)
        except anthropic.APIError as e:
            print(f"  anthropic error: {e}", file=sys.stderr)
            skipped.append((meta, vid, f"AnthropicError:{type(e).__name__}"))
            continue
        summarized.append((meta, body))
        print(f"  summarized: {meta['title'][:60]}")

    new_skips = len(skip_list) - initial_skip_count
    if new_skips > 0:
        save_skip_list(skip_list)
        print(f"Added {new_skips} videos to permanent skip list ({len(skip_list)} total)")

    if not summarized:
        print("Nothing to insert.")
        notify_skipped_on_slack(skipped)
        return 0

    summarized.sort(key=lambda mb: mb[0]["published_iso"], reverse=True)
    new_blocks = [render_block(m, b) for m, b in summarized]
    html_text = insert_blocks(html_text, new_blocks)
    html_text = update_header(html_text, len(have) + len(new_blocks))
    HTML_PATH.write_text(html_text, encoding="utf-8")
    print(f"Inserted {len(new_blocks)} new blocks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
