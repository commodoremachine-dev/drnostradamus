"""Günlük otomatik güncelleme (Windows Görev Zamanlayıcı çalıştırır).

  python src/daily.py           # tam: veri + yeniden eğitim + değerlendirme + tahmin kaydı
  python src/daily.py --hizli   # hızlı: güncel oranlarla başlamamış maçların tahminini yenile

Her adım ayrı çalışır; biri hata verirse diğerleri devam eder.
Çıktı: reports/daily_log.txt (son çalıştırmalar), reports/son_durum.txt (son değerlendirme)
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
LOG = ROOT / "reports" / "daily_log.txt"


def run(name, args, out_file=None, timeout=3600):
    t0 = datetime.now()
    try:
        r = subprocess.run([PY, *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                           errors="replace", timeout=timeout, env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
        ok = r.returncode == 0
        tail = (r.stdout or "").strip().splitlines()[-3:] + ((r.stderr or "").strip().splitlines()[-2:] if not ok else [])
        if out_file:
            (ROOT / "reports" / out_file).write_text(r.stdout or "", encoding="utf-8")
    except Exception as e:
        ok, tail = False, [str(e)]
    dur = (datetime.now() - t0).seconds
    line = f"  [{'OK ' if ok else 'HATA'}] {name} ({dur} sn)" + "".join(f"\n        {x}" for x in tail)
    print(line, flush=True)
    return line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hizli", action="store_true")
    a = ap.parse_args()
    (ROOT / "reports").mkdir(exist_ok=True)
    year = str(datetime.now().year)
    old = LOG.read_text(encoding="utf-8") if LOG.exists() else ""

    class Lines(list):
        """Her adımdan sonra günlüğü yazar (işlem yarıda kesilse bile görünsün)."""
        def append(self, x):
            super().append(x)
            LOG.write_text(("\n".join(self) + "\n\n" + old)[:200_000], encoding="utf-8")

    lines = Lines()
    lines.append(f"=== {datetime.now():%Y-%m-%d %H:%M} {'HIZLI' if a.hizli else 'TAM'} güncelleme ===")
    lines.append(run("ESPN sonuç + oran", ["src/espn_source.py", "--yillar", year]))
    if not a.hizli:
        lines.append(run("livescore", ["src/livescore_source.py"]))
        if (ROOT / ".env").exists() and "API_FOOTBALL_KEY=" in (ROOT / ".env").read_text(encoding="utf-8") \
                and (ROOT / ".env").read_text(encoding="utf-8").split("API_FOOTBALL_KEY=")[1].split("\n")[0].strip():
            season = str(datetime.now().year if datetime.now().month >= 7 else datetime.now().year - 1)
            lines.append(run("API-Football", ["src/api_football.py", "--sezon", season]))
    lines.append(run("Kaynakları birleştir", ["src/parse_openfootball.py"]))
    if not a.hizli:
        lines.append(run("Model eğitimi + test", ["src/backtest.py"], "backtest_son.txt"))
        lines.append(run("Sürpriz modeli", ["src/upset_model.py"]))
        lines.append(run("Value bet testi", ["src/value.py", "test"], "value_son.txt"))
    lines.append(run("Tahminleri kaydet", ["src/tracker.py", "kaydet", "--gun", "7"]))
    lines.append(run("Canlı değerlendirme", ["src/tracker.py", "durum"], "son_durum.txt"))


if __name__ == "__main__":
    main()
