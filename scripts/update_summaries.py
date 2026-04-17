#!/usr/bin/env python3
"""Fetch new videos from a YouTube playlist, summarize them with Claude, and
prepend the rendered HTML blocks into index.html."""
from __future__ import annotations

import html
import os
import re
import sys
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

ROOT = Path(__file__).resolve().parent.parent
HTML_PATH = ROOT / "index.html"

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


def fetch_transcript(video_id: str) -> str | None:
    try:
        entries = YouTubeTranscriptApi.get_transcript(
            video_id, languages=["en", "en-US", "en-GB"]
        )
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        return None
    except Exception as e:
        # Fall back to whatever is available.
        try:
            listing = YouTubeTranscriptApi.list_transcripts(video_id)
            # Prefer manual, any language
            for t in listing:
                try:
                    entries = t.fetch()
                    break
                except Exception:
                    continue
            else:
                print(f"  transcript error ({e!r}); no fallback worked", file=sys.stderr)
                return None
        except Exception as e2:
            print(f"  transcript error ({e!r} / {e2!r})", file=sys.stderr)
            return None
    text = " ".join(e["text"] for e in entries if e.get("text"))
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


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


def main() -> int:
    html_text = HTML_PATH.read_text(encoding="utf-8")
    have = existing_video_ids(html_text)
    print(f"Already summarized: {len(have)} videos")

    yt = build("youtube", "v3", developerKey=YT_KEY, cache_discovery=False)
    print(f"Fetching {len(PLAYLIST_IDS)} playlists:")
    playlist_ids = fetch_all_playlists(yt)
    print(f"Playlist total (deduped): {len(playlist_ids)} videos")
    if not playlist_ids:
        print("No playlist items fetched — aborting.", file=sys.stderr)
        return 1

    # Preserve playlist order; newest-first assumed, but we process in playlist
    # order so the final prepend keeps the most-recent-first convention.
    to_process = [v for v in playlist_ids if v not in have]
    print(f"New to summarize: {len(to_process)}")

    if not to_process:
        return 0

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    summarized: list[tuple[dict, str]] = []

    for vid in to_process:
        print(f"- {vid}")
        try:
            meta = fetch_video_metadata(yt, vid)
        except HttpError as e:
            print(f"  metadata error: {e}", file=sys.stderr)
            continue
        if not meta:
            print("  no metadata; skipping")
            continue
        transcript = fetch_transcript(vid)
        if not transcript:
            print("  no transcript available; skipping")
            continue
        try:
            body = generate_summary(client, meta, transcript)
        except anthropic.APIError as e:
            print(f"  anthropic error: {e}", file=sys.stderr)
            continue
        summarized.append((meta, body))
        print(f"  summarized: {meta['title'][:60]}")

    if not summarized:
        print("Nothing to insert.")
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
