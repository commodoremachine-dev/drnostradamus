"""Claude ile yapay zekâ maç analizi.

İki adım:
  1) Araştırma: Claude web'de iki takımın güncel haberlerini arar (sakatlık, ceza,
     hoca değişikliği, kadro, motivasyon). Sonuç kaynaklı bir özet metindir.
  2) Çıkarım: Claude; istatistik modelinin olasılıklarını, tüm değişkenleri, son maçları
     ve haber özetini birlikte yorumlar. Yapılandırılmış JSON döner: gol çarpanları,
     gerekçe, güven düzeyi, riskler.
Gol çarpanları [0.85, 1.15] ile sınırlanır: YZ modeli düzeltir ama ezemez.

Anahtar: .env içine ANTHROPIC_API_KEY=... (veya `ant auth login` profili).
Kullanım:  python src/predict.py Galatasaray Fenerbahçe --yz
"""
import json
import re
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from models import markets, score_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = "claude-opus-5-5"
MULT_MIN, MULT_MAX = 0.85, 1.15

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "ozet": {"type": "string", "description": "Maçın 2-3 cümlelik özeti"},
        "onemli_etkenler": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "etken": {"type": "string"},
                    "lehine": {"type": "string", "enum": ["ev", "deplasman", "nötr"]},
                    "onem": {"type": "string", "enum": ["düşük", "orta", "yüksek"]},
                },
                "required": ["etken", "lehine", "onem"],
                "additionalProperties": False,
            },
        },
        "ev_gol_carpani": {"type": "number", "description": "0.85-1.15 arası; 1.0 = modele katılıyorum"},
        "dep_gol_carpani": {"type": "number", "description": "0.85-1.15 arası; 1.0 = modele katılıyorum"},
        "en_olasi_skor": {"type": "string", "description": "Örn. 2-1"},
        "guven": {"type": "string", "enum": ["düşük", "orta", "yüksek"]},
        "gerekce": {"type": "string", "description": "Çarpanların ve tahminin gerekçesi"},
        "riskler": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["ozet", "onemli_etkenler", "ev_gol_carpani", "dep_gol_carpani",
                 "en_olasi_skor", "guven", "gerekce", "riskler"],
    "additionalProperties": False,
}

SYSTEM = """Sen Türkiye Süper Ligi konusunda uzman, veriye dayalı çalışan bir futbol analistisin.
İstatistik modeli olasılıkları zaten hesapladı; senin görevin modelin GÖREMEDİĞİ bilgileri
(güncel sakatlık/ceza, hoca değişikliği, kadro rotasyonu, motivasyon, maç önemi) ve
verilerdeki dikkat çekici durumları değerlendirip modeli gerekiyorsa sınırlı ölçüde düzeltmek.
Model geçmiş testlerde iyi kalibre çıktı: güçlü bir neden yoksa çarpanları 1.0'a yakın tut.
Bilmediğin bir şeyi uydurma; haber bulunamadıysa bunu belirt. Yanıtların Türkçe olsun."""


def _load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                if k.strip() == "ANTHROPIC_API_KEY" and v.strip() and not os.environ.get(k.strip()):
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")


def _client():
    _load_env()
    import anthropic
    return anthropic, anthropic.Anthropic()


def _create(anthropic, client, **kw):
    """Opus 5.5 isteği; güvenlik sınıflandırıcısı reddederse sunucu taraflı yedek model devreye girer."""
    return client.beta.messages.create(
        model=MODEL, betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw)


def research_news(home: str, away: str, match_date: str) -> str:
    """Adım 1: web araması ile güncel haber özeti."""
    anthropic, client = _client()
    messages = [{"role": "user", "content": (
        f"{match_date} tarihli Süper Lig maçı: {home} (ev) - {away} (deplasman).\n"
        "Her iki takım için bu maçı etkileyebilecek GÜNCEL bilgileri ara: sakat ve cezalı oyuncular, "
        "muhtemel ilk 11 ve kilit eksikler, teknik direktör değişikliği, son maçlardaki durum, "
        "Avrupa/kupa yoğunluğu, motivasyon (şampiyonluk, küme düşme, derbi). "
        "Bulduklarını kaynak ve tarihleriyle kısa maddeler halinde Türkçe özetle. "
        "Bulamadığın konuları 'bilgi bulunamadı' diye belirt.")}]
    tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": 6}]
    for _ in range(5):
        resp = _create(anthropic, client, max_tokens=16000, tools=tools, messages=messages,
                       output_config={"effort": "medium"})
        if resp.stop_reason == "pause_turn":  # uzun arama: kaldığı yerden devam ettir
            messages.append({"role": "assistant", "content": resp.content})
            continue
        break
    if resp.stop_reason == "refusal":
        return "Haber araştırması yapılamadı (istek reddedildi)."
    return "\n".join(b.text for b in resp.content if b.type == "text").strip() or "Haber bulunamadı."


def _team_recent(matches: pd.DataFrame, team: str, before, n=6):
    m = matches[matches.played & (matches.date < before) & ((matches.home == team) | (matches.away == team))].tail(n)
    out = []
    for r in m.itertuples(index=False):
        venue = "E" if r.home == team else "D"
        opp = r.away if r.home == team else r.home
        gf, ga = (r.hg, r.ag) if r.home == team else (r.ag, r.hg)
        res = "G" if gf > ga else "B" if gf == ga else "M"
        out.append(f"{r.date:%d.%m.%Y} {venue} {opp} {int(gf)}-{int(ga)} ({res})")
    return out


def build_package(r, matches: pd.DataFrame, keys=None) -> dict:
    from predict import match_markets
    _, mk = match_markets(r)  # nihai tahmin (piyasa + model, kalibre)
    skip = {"season", "date", "home", "away", "hg", "ag", "played", "round"}
    feats = {k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v)
             for k, v in r._asdict().items()
             if k not in skip and not (isinstance(v, float) and np.isnan(v)) and (keys is None or k in keys)}
    feats = {k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in feats.items()}
    return {
        "mac": f"{r.home} (ev) - {r.away} (deplasman), {r.date:%d.%m.%Y}",
        "model_tahmini": {
            "beklenen_gol_ev": round(r.lh, 2), "beklenen_gol_dep": round(r.la, 2),
            "ev_kazanir": round(mk["p_home"], 3), "beraberlik": round(mk["p_draw"], 3),
            "dep_kazanir": round(mk["p_away"], 3), "ust_2_5": round(mk["p_over25"], 3),
            "kg_var": round(mk["p_btts"], 3),
            "en_olasi_skorlar": [(s, round(p, 3)) for s, p in mk["top_scores"]],
            "not": "Nihai tahmin (piyasa + model) geçmiş 1.121 maçlık testte maç sonucunu %57 doğru bildi.",
        },
        "son_maclar": {r.home: _team_recent(matches, r.home, r.date),
                       r.away: _team_recent(matches, r.away, r.date)},
        "degiskenler": feats,
    }


# ---------------------------------------------------------------- yerel model (Ollama)
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL_DEFAULT = "qwen3:8b"

# Yerel model işlemcide yavaş: yalnızca en önemli değişkenler gönderilir
KEY_FEATURES = ["elo_home", "elo_away", "pi_exp_gd", "home_ppg5", "away_ppg5", "home_season_ppg", "away_season_ppg",
                "home_rank", "away_rank", "home_gf_venue_ewm", "home_ga_venue_ewm", "away_gf_venue_ewm",
                "away_ga_venue_ewm", "home_home_pts_ewm", "away_away_pts_ewm", "home_promoted", "away_promoted",
                "home_prev_rank", "away_prev_rank", "home_rest_days", "away_rest_days", "home_other_comp_14d",
                "away_other_comp_14d", "home_regulars_missing", "away_regulars_missing", "h2h_n", "h2h_gd_home",
                "is_derby", "away_travel_km", "w_rain_mm", "odds_ph", "odds_pd", "odds_pa"]


def _env(name, default=None):
    _load_env_all()
    return os.environ.get(name, default)


def _load_env_all():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if v and not os.environ.get(k):
                    os.environ[k] = v


def ollama_ok(start=True) -> bool:
    """Ollama çalışıyor mu? Çalışmıyorsa penceresiz başlatmayı dener."""
    import subprocess
    import time
    import urllib.request

    def ping():
        try:
            urllib.request.urlopen(OLLAMA_URL + "/api/version", timeout=2)
            return True
        except Exception:  # noqa: BLE001
            return False
    if ping():
        return True
    if not start:
        return False
    exe = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    try:
        subprocess.Popen([str(exe) if exe.exists() else "ollama", "serve"], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except OSError:
        return False
    for _ in range(20):
        time.sleep(1)
        if ping():
            return True
    return False


def choose_backend() -> str:
    """AI_MOTORU=claude|yerel|otomatik (.env). Otomatik: Claude anahtarı varsa Claude, yoksa yerel model."""
    want = (_env("AI_MOTORU", "otomatik") or "otomatik").lower()
    if want == "claude" or (want == "otomatik" and _env("ANTHROPIC_API_KEY")):
        return "claude"
    return "yerel"


def backend_label() -> str:
    b = choose_backend()
    return "Claude (web araması)" if b == "claude" else f"Yerel model: {_env('OLLAMA_MODEL', OLLAMA_MODEL_DEFAULT)}"


def _user_prompt(pkg, news):
    return ("Aşağıda istatistik modelinin çıktısı, değişkenler ve son maçlar var (JSON). "
            "Değişken adları: home_/away_ önekleri ev/deplasman takımını; _ewm son maçlara ağırlık "
            "veren ortalamayı; pi_ Pi-rating'i; odds_ marjı temizlenmiş piyasa olasılığını; "
            "regulars_missing ilk 11'de olmayan asıl oyuncu sayısını; w_ hava durumunu belirtir.\n\n"
            f"```json\n{json.dumps(pkg, ensure_ascii=False, default=str)}\n```\n\n"
            f"Güncel haberler:\n{news}\n\n"
            "Bunları değerlendir ve modelin beklenen gollerine uygulanacak çarpanları belirle. "
            "Yalnızca istenen JSON'u döndür.")


def _local_analyze(pkg, news):
    import urllib.request
    model = _env("OLLAMA_MODEL", OLLAMA_MODEL_DEFAULT)
    keys = ", ".join(ANALYSIS_SCHEMA["required"])
    schema_hint = ("\n\nYanıt biçimi (yalnızca JSON): {\"ozet\": str, \"onemli_etkenler\": [{\"etken\": str, "
                   "\"lehine\": \"ev\"|\"deplasman\"|\"nötr\", \"onem\": \"düşük\"|\"orta\"|\"yüksek\"}], "
                   "\"ev_gol_carpani\": sayı 0.85-1.15, \"dep_gol_carpani\": sayı 0.85-1.15, \"en_olasi_skor\": \"2-1\", "
                   "\"guven\": \"düşük\"|\"orta\"|\"yüksek\", \"gerekce\": str, \"riskler\": [str]}"
                   "\n\nKurallar: 'dogrulanmis_bilgiler' kesindir, önce onlara dayan. Haber başlıklarında AÇIKÇA "
                   "yazmayan bir şeyi (hoca değişikliği, sakatlık vb.) iddia etme. Son 5 gün içinde Avrupa maçı "
                   "oynayan takımın yorgunluğunu dikkate al. Emin değilsen çarpanları 1.0'a yakın bırak ve güveni "
                   "'düşük' yaz. 'lehine' alanı etkenin hangi takıma yaradığını göstermeli.")
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": _user_prompt(pkg, news) + schema_hint}]
    last_err = None
    # Önce tam şema; yerel modeller bazen şema modunda takılır -> sınır + basit JSON yedeği
    for fmt in (ANALYSIS_SCHEMA, "json"):
        body = {"model": model, "stream": False, "think": False, "format": fmt, "keep_alive": "15m",
                "options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": 1500},
                "messages": msgs}
        req = urllib.request.Request(OLLAMA_URL + "/api/chat", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                out = json.loads(r.read().decode("utf-8"))
            txt = out["message"]["content"]
            m = re.search(r"\{.*\}", txt, re.S)
            data = json.loads(m.group(0) if m else txt)
            if all(k in data for k in ("ev_gol_carpani", "dep_gol_carpani")):
                return data
            last_err = f"eksik alanlar (beklenen: {keys})"
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(last_err)


def adjusted_final(r, mh, ma) -> dict:
    """YZ çarpanlarını NİHAİ tahmine (piyasa + model) oransal uygular.

    Çarpanların model olasılıklarını ne kadar değiştirdiği hesaplanır ve aynı oran nihai
    olasılıklara uygulanır (1X2 ve skor matrisi bölgeleri ile).
    """
    from predict import match_markets
    from stacking import rescale_matrix
    m_fin, fin = match_markets(r)
    base = markets(score_matrix(r.lh, r.la, r.dc_rho))
    new = markets(score_matrix(r.lh * mh, r.la * ma, r.dc_rho))
    p = np.array([fin["p_home"] * new["p_home"] / base["p_home"],
                  fin["p_draw"] * new["p_draw"] / base["p_draw"],
                  fin["p_away"] * new["p_away"] / base["p_away"]])
    p = p / p.sum()
    # skor dağılımı: gol beklentisi çarpanlarıyla kaydırılmış matris, düzeltilmiş 1X2'ye ölçeklenir
    out = markets(rescale_matrix(score_matrix(r.lh * mh, r.la * ma, r.dc_rho), p))
    out["onceki"] = fin
    return out


def analyze(r, matches: pd.DataFrame, news: str | None = None, backend: str | None = None) -> dict:
    """Yapılandırılmış YZ çıkarımı ve düzeltilmiş tahmin (Claude ya da yerel model)."""
    from news import gather_news, news_text
    backend = backend or choose_backend()
    rss = news_text(gather_news(r.home, r.away, pd.Timestamp(r.date).date()))
    if backend == "yerel":
        if not ollama_ok():
            return {"hata": "Yerel model (Ollama) çalışmıyor ve başlatılamadı. Ollama'yı açıp tekrar deneyin."}
        news = news or rss
        pkg = build_package(r, matches, keys=KEY_FEATURES)
        # Uygulamanın kendi hesapladığı, doğrulanmış gerekçeler: yerel modelin uydurmasını azaltır
        from explain import explain
        from predict import match_markets
        _, fin0 = match_markets(r)
        _, mod0 = match_markets(r, use_market=False)
        pkg["dogrulanmis_bilgiler"] = [f"[{s} lehine] {t}" if s != "nötr" else t for s, t in explain(r, fin0, mod0)]
        try:
            data = _local_analyze(pkg, news)
        except Exception as e:  # noqa: BLE001
            return {"hata": f"Yerel model yanıt veremedi: {e}", "haberler": news}
        data.setdefault("riskler", [])
        data.setdefault("onemli_etkenler", [])
        motor = f"yerel ({_env('OLLAMA_MODEL', OLLAMA_MODEL_DEFAULT)})"
    else:
        try:
            anthropic, client = _client()
        except Exception as e:  # noqa: BLE001
            return {"hata": f"Claude istemcisi kurulamadı: {e}"}
        try:
            if news is None:
                news = research_news(r.home, r.away, f"{r.date:%d.%m.%Y}") + "\n\nHaber başlıkları:\n" + rss
            pkg = build_package(r, matches)
            resp = _create(anthropic, client, max_tokens=16000, system=SYSTEM,
                           messages=[{"role": "user", "content": _user_prompt(pkg, news)}],
                           output_config={"effort": "high",
                                          "format": {"type": "json_schema", "schema": ANALYSIS_SCHEMA}})
            if resp.stop_reason == "refusal":
                return {"hata": "Analiz isteği reddedildi.", "haberler": news}
            data = json.loads(next(b.text for b in resp.content if b.type == "text"))
        except anthropic.AuthenticationError:
            return {"hata": "Claude API anahtarı geçersiz (.env içindeki ANTHROPIC_API_KEY'i kontrol edin)."}
        except TypeError as e:
            if "authentication" in str(e).lower():
                return {"hata": "Claude API anahtarı bulunamadı. .env dosyasına ANTHROPIC_API_KEY=... ekleyin."}
            raise
        except anthropic.APIStatusError as e:
            return {"hata": f"Claude API hatası ({e.status_code}): {e.message}"}
        except anthropic.APIConnectionError:
            return {"hata": "Claude API'ye bağlanılamadı."}
        motor = "Claude"

    def num(x, d=1.0):
        try:
            return float(x)
        except (TypeError, ValueError):
            return d
    mh = float(np.clip(num(data.get("ev_gol_carpani")), MULT_MIN, MULT_MAX))
    ma = float(np.clip(num(data.get("dep_gol_carpani")), MULT_MIN, MULT_MAX))
    adj = adjusted_final(r, mh, ma)
    data.update(haberler=news, ev_carpan_uygulanan=mh, dep_carpan_uygulanan=ma,
                duzeltilmis=adj, home=r.home, away=r.away, motor=motor)
    for k, v in (("ozet", ""), ("gerekce", ""), ("guven", "-"), ("en_olasi_skor", "-")):
        data.setdefault(k, v)
    return data


def print_analysis(a: dict):
    print(f"\n{'-' * 62}\nYAPAY ZEKÂ ANALİZİ ({a.get('motor', 'Claude')})\n{'-' * 62}")
    if "hata" in a:
        print("  " + a["hata"])
        return
    print(a["ozet"])
    print("\nÖnemli etkenler:")
    for e in a["onemli_etkenler"]:
        print(f"  [{e['onem']}] ({e['lehine']} lehine) {e['etken']}")
    print(f"\nGerekçe: {a['gerekce']}")
    d = a["duzeltilmis"]
    print(f"\nYZ düzeltmesi: ev gol x{a['ev_carpan_uygulanan']:.2f}, deplasman gol x{a['dep_carpan_uygulanan']:.2f}"
          f"   (güven: {a['guven']})")
    print(f"Düzeltilmiş 1 / X / 2:  %{d['p_home'] * 100:.1f} / %{d['p_draw'] * 100:.1f} / %{d['p_away'] * 100:.1f}"
          f"    2.5 Üst: %{d['p_over25'] * 100:.1f}")
    print(f"YZ'nin skor tahmini: {a['en_olasi_skor']}   |   "
          f"Düzeltilmiş en olası skorlar: " + ", ".join(f"{s} (%{p * 100:.1f})" for s, p in d["top_scores"][:3]))
    if a["riskler"]:
        print("Riskler: " + "; ".join(a["riskler"]))
    print("\nHaber araştırması:\n" + a["haberler"])
