"""Web fetch tool (daily mode): open a URL and return its readable text."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from limbo.models import ToolResult
from limbo.tools.base import BaseTool, ToolError

MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36 limbo"
)
_TEXT_TYPES = (
    "text/plain", "text/markdown", "text/csv", "application/json", "application/xml", "text/xml"
)

# Subtrees that never hold the page's content.
_SKIP = frozenset(
    {"script", "style", "noscript", "svg", "template", "iframe", "nav", "footer",
     "header", "aside", "form", "button", "select", "head"}
)
_BLOCK = frozenset(
    {"p", "div", "section", "article", "main", "ul", "ol", "table", "tr",
     "blockquote", "pre", "figure", "figcaption", "dl", "dt", "dd", "br", "hr"}
)
_CONTENT_ROOTS = ("article", "main")


class _Extractor(HTMLParser):
    """HTML → Markdown-ish text; remembers what sits inside <article>/<main>."""

    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title = ""
        self._in_title = False
        # The skipped subtree's tag and how deep it nests in itself (other
        # tags inside are ignored, so unclosed <p>/<li> cannot unbalance it).
        self._skip_tag: str | None = None
        self._skip_depth = 0
        self._root_depth = 0
        self._pre = 0
        self._links: list[str | None] = []
        # (text, inside a content root)
        self.parts: list[tuple[str, bool]] = []

    def _emit(self, text: str) -> None:
        self.parts.append((text, self._root_depth > 0))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        if self._skip_tag is not None:
            if tag == self._skip_tag:
                self._skip_depth += 1
            return
        if tag in _SKIP:
            self._skip_tag, self._skip_depth = tag, 1
            return
        if tag in _CONTENT_ROOTS:
            self._root_depth += 1
        if tag == "pre":
            self._pre += 1
            self._emit("\n```\n")
        elif re.fullmatch(r"h[1-6]", tag):
            self._emit("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "li":
            self._emit("\n- ")
        elif tag in ("td", "th"):
            self._emit(" | ")
        elif tag in _BLOCK:
            self._emit("\n\n" if tag in ("p", "blockquote") else "\n")
        elif tag == "a":
            href = dict(attrs).get("href")
            if href and not href.startswith(("#", "javascript:")):
                self._links.append(urljoin(self.base_url, href))
                self._emit("[")
            else:
                self._links.append(None)
        elif tag == "img":
            alt = (dict(attrs).get("alt") or "").strip()
            if alt:
                self._emit(f"[image: {alt}]")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if self._skip_tag is not None:
            if tag == self._skip_tag:
                self._skip_depth -= 1
                if not self._skip_depth:
                    self._skip_tag = None
            return
        if tag == "a" and self._links:
            href = self._links.pop()
            if href is not None:
                self._emit(f"]({href})")
        elif tag == "pre" and self._pre:
            self._pre -= 1
            self._emit("\n```\n")
        elif re.fullmatch(r"h[1-6]", tag) or tag in _BLOCK:
            self._emit("\n")
        if tag in _CONTENT_ROOTS and self._root_depth:
            self._root_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        if self._skip_tag is not None:
            return
        self._emit(data if self._pre else re.sub(r"\s+", " ", data))

    def text(self) -> str:
        rooted = [t for t, inside in self.parts if inside]
        chosen = rooted if "".join(rooted).strip() else [t for t, _ in self.parts]
        text = "".join(chosen)
        text = re.sub(r"\[\s*\]\([^)]*\)", "", text)  # links with no text
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()


def html_to_text(html: str, base_url: str = "") -> tuple[str, str]:
    """(title, readable text) of an HTML page."""
    parser = _Extractor(base_url)
    parser.feed(html)
    parser.close()
    return re.sub(r"\s+", " ", parser.title).strip(), parser.text()


class WebFetchTool(BaseTool):
    name = "web_fetch"
    # Read-only: replaying it after a crash (or calling it under /plan)
    # changes nothing.
    replay = "safe"
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "http(s) URL to open."},
            "offset": {
                "type": "integer",
                "description": "Character offset to continue a long page from (default 0).",
            },
        },
        "required": ["url"],
    }

    def __init__(self, workdir: Path, max_chars: int = 20_000, timeout: float = 30.0):
        super().__init__(workdir)
        self.max_chars = max_chars
        self.timeout = timeout
        self.description = (
            "Open a web page and return its readable text as Markdown (title, "
            "headings, lists, links; navigation and scripts stripped). Use it to "
            "read search results or links the user gives you. Long pages come in "
            f"chunks of {max_chars} characters; pass offset to read further."
        )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        url = arguments.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ToolError("url must be a non-empty string")
        url = url.strip()
        if urlparse(url).scheme not in ("http", "https"):
            raise ToolError("url must start with http:// or https://")
        offset = arguments.get("offset", 0)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ToolError("offset must be a non-negative integer")

        final_url, content_type, body = self._download(url)
        if content_type == "application/pdf" or body[:5] == b"%PDF-":
            raise ToolError(
                "This URL is a PDF, which web_fetch cannot read yet. Download it "
                "with bash (e.g. curl -L -o file.pdf URL) and convert it locally."
            )
        text = body.decode(_charset(content_type, body), errors="replace")
        title = ""
        if "html" in content_type or (not content_type and "<html" in text[:2000].lower()):
            title, text = html_to_text(text, final_url)
        elif content_type and not content_type.startswith(_TEXT_TYPES):
            raise ToolError(f"Unsupported content type: {content_type}")
        return ToolResult(success=True, output=self._page(final_url, title, text, offset))

    def _download(self, url: str) -> tuple[str, str, bytes]:
        try:
            with httpx.Client(
                follow_redirects=True,
                timeout=self.timeout,
                headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*;q=0.8"},
            ) as client, client.stream("GET", url) as response:
                if response.status_code >= 400:
                    raise ToolError(f"HTTP {response.status_code} for {response.url}")
                chunks: list[bytes] = []
                size = 0
                for chunk in response.iter_bytes():
                    chunks.append(chunk)
                    size += len(chunk)
                    if size >= MAX_DOWNLOAD_BYTES:
                        break
                content_type = response.headers.get("content-type", "")
                return str(response.url), content_type, b"".join(chunks)
        except httpx.HTTPError as e:
            raise ToolError(f"Could not fetch {url}: {e}") from e

    def _page(self, url: str, title: str, text: str, offset: int) -> str:
        total = len(text)
        if offset and offset >= total:
            raise ToolError(f"offset {offset} is past the end of the page ({total} characters)")
        chunk = text[offset : offset + self.max_chars]
        end = offset + len(chunk)
        lines = [f"URL: {url}"]
        if title:
            lines.append(f"Title: {title}")
        out = "\n".join(lines) + "\n\n" + (chunk or "(no readable text)")
        if offset or end < total:
            out += f"\n\n[Characters {offset}-{end} of {total}."
            out += f" Call again with offset={end} for more.]" if end < total else "]"
        return out


def _charset(content_type: str, body: bytes) -> str:
    header = re.search(r"charset=([\w-]+)", content_type, re.I)
    if header:
        name = header.group(1)
    else:
        meta = re.search(rb"<meta[^>]+charset=[\"']?([\w-]+)", body[:4096], re.I)
        name = meta.group(1).decode("ascii", "ignore") if meta else "utf-8"
    try:
        "".encode(name)
    except LookupError:
        return "utf-8"
    return name
