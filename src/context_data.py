"""Bağlamsal veriler: seyahat mesafesi, hava durumu, kupa/Avrupa maç tarihleri.

Hava durumu: Open-Meteo (ücretsiz, anahtar gerekmez). Geçmiş için arşiv, önümüzdeki
16 gün için tahmin servisi kullanılır. Şehir başına önbelleklenir: data/raw/weather/.
"""
import json
import math
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WEATHER_DIR = ROOT / "data" / "raw" / "weather"

# takım -> (şehir, enlem, boylam)  (stadyum yaklaşık konumu)
TEAM_LOC = {
    "Adana Demirspor": ("Adana", 37.00, 35.32), "Akhisar Belediyespor": ("Manisa", 38.92, 27.84),
    "Alanyaspor": ("Alanya", 36.54, 32.00), "Amed": ("Diyarbakır", 37.91, 40.23),
    "Ankaragücü": ("Ankara", 39.94, 32.85), "Antalyaspor": ("Antalya", 36.89, 30.68),
    "Başakşehir": ("İstanbul", 41.09, 28.80), "Beşiktaş": ("İstanbul", 41.04, 29.00),
    "Bodrum FK": ("Bodrum", 37.04, 27.43), "Bursaspor": ("Bursa", 40.22, 29.01),
    "Denizlispor": ("Denizli", 37.78, 29.09), "Erzurumspor": ("Erzurum", 39.90, 41.27),
    "Eyüpspor": ("İstanbul", 41.05, 28.93), "Fatih Karagümrük": ("İstanbul", 41.07, 28.77),
    "Fenerbahçe": ("İstanbul", 40.99, 29.04), "Galatasaray": ("İstanbul", 41.10, 28.99),
    "Gaziantep FK": ("Gaziantep", 37.07, 37.38), "Gençlerbirliği": ("Ankara", 39.94, 32.85),
    "Göztepe": ("İzmir", 38.40, 27.10), "Hatayspor": ("Hatay", 36.20, 36.16),
    "Kasımpaşa": ("İstanbul", 41.04, 28.97), "Kayserispor": ("Kayseri", 38.72, 35.49),
    "Kocaelispor": ("Kocaeli", 40.77, 29.92), "Konyaspor": ("Konya", 37.94, 32.50),
    "Pendikspor": ("İstanbul", 40.88, 29.25), "Samsunspor": ("Samsun", 41.29, 36.33),
    "Sivasspor": ("Sivas", 39.75, 37.02), "Trabzonspor": ("Trabzon", 40.99, 39.68),
    "Yeni Malatyaspor": ("Malatya", 38.35, 38.31), "Çaykur Rizespor": ("Rize", 41.02, 40.52),
    "Çorum": ("Çorum", 40.55, 34.95), "İstanbulspor AŞ": ("İstanbul", 41.03, 28.67),
    "Adanaspor": ("Adana", 37.00, 35.32), "Altay": ("İzmir", 38.43, 27.15),
    "Gaziantepspor": ("Gaziantep", 37.07, 37.38), "Giresunspor": ("Giresun", 40.91, 38.39),
    "Karabükspor": ("Karabük", 41.20, 32.62), "Osmanlıspor": ("Ankara", 39.94, 32.85),
    "Ümraniyespor": ("İstanbul", 41.02, 29.12),
}


def haversine_km(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def _get_json(url, params, tries=4):
    import urllib.error
    full = f"{url}?{urllib.parse.urlencode(params)}"
    for i in range(tries):
        try:
            with urllib.request.urlopen(full, timeout=60) as r:
                time.sleep(1.5)  # ücretsiz servise nazik ol
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code != 429 or i == tries - 1:
                raise
            time.sleep(20 * (i + 1))  # dakika sınırı: bekle ve tekrar dene


DAILY = "temperature_2m_max,precipitation_sum,wind_speed_10m_max"


def _city_weather(city, lat, lon, needed: set) -> dict:
    """Şehir için tarih -> (sıcaklık, yağış, rüzgar). Eksik tarihleri indirir."""
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    cache_p = WEATHER_DIR / f"{city}.json"
    cache = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.exists() else {}
    today = date.today()
    archive_end = today - timedelta(days=6)  # arşiv birkaç gün geriden gelir
    missing_past = sorted(d for d in needed if d not in cache and date.fromisoformat(d) <= archive_end)
    if missing_past:
        try:
            js = _get_json("https://archive-api.open-meteo.com/v1/archive", dict(
                latitude=lat, longitude=lon, start_date=missing_past[0], end_date=missing_past[-1],
                daily=DAILY, timezone="Europe/Istanbul"))
            dd = js["daily"]
            for i, d in enumerate(dd["time"]):
                cache[d] = [dd["temperature_2m_max"][i], dd["precipitation_sum"][i], dd["wind_speed_10m_max"][i]]
            cache_p.write_text(json.dumps(cache), encoding="utf-8")
        except Exception as e:  # ağ hatası tahmini durdurmasın
            print(f"Uyarı: {city} hava arşivi alınamadı ({e})")
    out = dict(cache)
    future = [d for d in needed if d not in cache and archive_end < date.fromisoformat(d) <= today + timedelta(days=15)]
    if future:
        fc_p = WEATHER_DIR / f"{city}_forecast.json"
        fc = None
        if fc_p.exists() and time.time() - fc_p.stat().st_mtime < 6 * 3600:
            fc = json.loads(fc_p.read_text(encoding="utf-8"))
        if fc is None:
            try:
                js = _get_json("https://api.open-meteo.com/v1/forecast", dict(
                    latitude=lat, longitude=lon, daily=DAILY, timezone="Europe/Istanbul",
                    past_days=7, forecast_days=16))
                dd = js["daily"]
                fc = {d: [dd["temperature_2m_max"][i], dd["precipitation_sum"][i], dd["wind_speed_10m_max"][i]]
                      for i, d in enumerate(dd["time"])}
                fc_p.write_text(json.dumps(fc), encoding="utf-8")
            except Exception as e:
                print(f"Uyarı: {city} hava tahmini alınamadı ({e})")
                fc = {}
        out.update({d: v for d, v in fc.items() if d not in out})
    return out


def build_context(df: pd.DataFrame, weather=True) -> dict:
    """(date, home, away) -> {seyahat, hava} sözlüğü."""
    ctx = {}
    need = {}
    for r in df[["date", "home", "away"]].itertuples(index=False):
        h, a = TEAM_LOC.get(r.home), TEAM_LOC.get(r.away)
        c = {}
        if h and a:
            c["away_travel_km"] = haversine_km(h[1:], a[1:])
            c["same_city"] = int(h[0] == a[0])
        if h:
            need.setdefault(h[0], (h[1], h[2], set()))[2].add(str(pd.Timestamp(r.date).date()))
        ctx[(r.date, r.home, r.away)] = c
    if weather:
        wx = {city: _city_weather(city, lat, lon, ds) for city, (lat, lon, ds) in need.items()}
        for (d, home, away), c in ctx.items():
            h = TEAM_LOC.get(home)
            if not h:
                continue
            v = wx.get(h[0], {}).get(str(pd.Timestamp(d).date()))
            if v and None not in v:
                c["w_temp_max"], c["w_rain_mm"], c["w_wind_max"] = v
                c["w_heavy_rain"] = int(v[1] >= 5)
    return ctx


def load_other_dates() -> dict:
    """Kupa/Avrupa maç tarihleri: ESPN (Avrupa kupaları) + API-Football (varsa, kupa dahil)."""
    parts = [pd.read_csv(ROOT / "data" / f, parse_dates=["date"])
             for f in ("espn_other_matches.csv", "api_other_matches.csv") if (ROOT / "data" / f).exists()]
    if not parts:
        return {}
    o = pd.concat(parts).drop_duplicates(["date", "team"])
    return {t: sorted(g.date) for t, g in o.groupby("team")}
