"""Maç öncesi haber başlıkları (ücretsiz, anahtarsız): Google Haberler RSS (Türkçe) + ESPN maç haberleri.

Yerel dil modeli web araması yapamadığı için sakatlık, ceza, hoca değişikliği gibi
bilgileri buradan alır. Yalnızca başlık ve kısa özet okunur.
"""
import html
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

UA = {"User-Agent": "Mozilla/5.0 (personal research)"}


def _get(url, timeout=20):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def _clean(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(s or ""))).strip()


def google_news(query, days=10, limit=10):
    url = ("https://news.google.com/rss/search?q=" + urllib.parse.quote(query)
           + "&hl=tr&gl=TR&ceid=TR:tr")
    out = []
    try:
        x = _get(url)
    except Exception:  # noqa: BLE001
        return out
    since = datetime.now(timezone.utc) - timedelta(days=days)
    for item in re.findall(r"<item>(.*?)</item>", x, re.S):
        t = re.search(r"<title>(.*?)</title>", item, re.S)
        d = re.search(r"<pubDate>(.*?)</pubDate>", item, re.S)
        if not t or not d:
            continue
        try:
            when = parsedate_to_datetime(d.group(1))
        except (TypeError, ValueError):
            continue
        if when < since:
            continue
        out.append({"tarih": when, "baslik": _clean(t.group(1)), "kaynak": "Google Haberler"})
        if len(out) >= limit:
            break
    return out


def espn_news(home_en=None, away_en=None, event_id=None):
    """ESPN maç sayfasındaki haberler (İngilizce)."""
    if not event_id:
        return []
    try:
        s = json.loads(_get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/tur.1/summary?event={event_id}"))
    except Exception:  # noqa: BLE001
        return []
    n = s.get("news") or {}
    arts = n.get("articles", []) if isinstance(n, dict) else n
    out = []
    for a in arts[:8]:
        try:
            when = datetime.fromisoformat(a.get("published", "").replace("Z", "+00:00"))
        except ValueError:
            when = datetime.now(timezone.utc)
        out.append({"tarih": when, "baslik": _clean(a.get("headline")) + " — " + _clean(a.get("description"))[:200],
                    "kaynak": "ESPN"})
    return out


def find_espn_event(home, away, date):
    """ESPN maç kimliğini bulur (takım adları ESPN yazımıyla karşılaştırılır)."""
    from teams import normalize
    try:
        d = datetime.strptime(str(date)[:10], "%Y-%m-%d")
        s = json.loads(_get(f"https://site.api.espn.com/apis/site/v2/sports/soccer/tur.1/scoreboard?dates={d:%Y%m%d}"))
    except Exception:  # noqa: BLE001
        return None
    for e in s.get("events", []):
        c = {x["homeAway"]: normalize(x["team"]["displayName"]) for x in e["competitions"][0]["competitors"]}
        if c.get("home") == home and c.get("away") == away:
            return e["id"]
    return None


def gather_news(home, away, date, max_items=25):
    """Maçla ilgili son 10 günün haber başlıkları, en yeniden eskiye."""
    items = []
    queries = [f'"{home}" "{away}"', f"{home} sakat OR cezalı OR kadro", f"{away} sakat OR cezalı OR kadro",
               f"{home} teknik direktör", f"{away} teknik direktör"]
    for q in queries:
        items += google_news(q, limit=8)
    items += espn_news(event_id=find_espn_event(home, away, date))
    seen, uniq = set(), []
    for it in sorted(items, key=lambda x: x["tarih"], reverse=True):
        key = it["baslik"][:60].lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)
    return uniq[:max_items]


def news_text(items):
    if not items:
        return "Haber bulunamadı."
    return "\n".join(f"- {it['tarih']:%d.%m %H:%M} [{it['kaynak']}] {it['baslik']}" for it in items)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    print(news_text(gather_news("Beşiktaş", "Gençlerbirliği", "2026-11-08")))
