"""Nostradamus masaüstü arayüzü (Tkinter).

Başlatma: Nostradamus.pyw (çift tık) veya  .venv\\Scripts\\pythonw.exe Nostradamus.pyw
Test için: python src/app.py --ekran-goruntusu cikti.png [--sekme 0] [--mac]
"""
import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
import traceback
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PY = ROOT / ".venv" / "Scripts" / "python.exe"

# Renkler
BG = "#f4f6f8"
CARD = "#ffffff"
HEAD = "#14213d"
ACCENT = "#16a34a"
C_HOME, C_DRAW, C_AWAY = "#2563eb", "#9ca3af", "#dc2626"
MUTED = "#6b7280"
LEVEL_COLORS = {"düşük": "#16a34a", "orta": "#ca8a04", "yüksek": "#ea580c", "çok yüksek": "#dc2626"}
FONT = ("Segoe UI", 10)


def pct(x):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"%{100 * x:.0f}"


def read_env_key(name):
    p = ROOT / ".env"
    if os.environ.get(name):
        return True
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.startswith(name + "=") and line.split("=", 1)[1].strip():
                return True
    return False


# ====================================================================== yardımcılar
class Worker:
    """Ağır işleri arka planda çalıştırır; sonucu ana iş parçacığında geri çağırır."""

    def __init__(self, root):
        self.root = root
        self.q = queue.Queue()
        root.after(100, self._poll)

    def run(self, func, done, fail=None):
        def target():
            try:
                self.q.put((done, func(), None))
            except Exception as e:  # noqa: BLE001
                self.q.put((fail, None, (e, traceback.format_exc())))
        threading.Thread(target=target, daemon=True).start()

    def _poll(self):
        try:
            while True:
                cb, res, err = self.q.get_nowait()
                if err is not None:
                    if cb:
                        cb(err)
                    else:
                        messagebox.showerror("Hata", f"{err[0]}\n\n{err[1][-1500:]}")
                elif cb:
                    cb(res)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)


class Data:
    """Maç verisi ve tahmin hesapları (önbellekli)."""

    def __init__(self):
        self._matches = None
        self._stamp = None

    def matches(self):
        from features import load_matches
        p = ROOT / "data" / "matches.csv"
        st = p.stat().st_mtime
        if self._matches is None or st != self._stamp:
            self._matches, self._stamp = load_matches(), st
        return self._matches

    def teams(self):
        m = self.matches()
        cur = m.season.max()
        return sorted(set(m[m.season == cur].home) | set(m[m.season == cur].away))

    def fixture_row(self, season, date, home, away):
        """Bir maç için tüm hesaplamalar: tahmin, gerekçe, value, sürpriz."""
        from explain import explain
        from predict import match_markets, predict, upset_info
        from value import value_table
        m = self.matches()
        fx = pd.DataFrame([dict(season=season, date=pd.Timestamp(date).normalize(), home=home, away=away)])
        r = next(predict(fx, m).itertuples(index=False))
        mat, fin = match_markets(r)
        _, mod = match_markets(r, use_market=False)
        odds = m[(m.season == season) & (m.home == home) & (m.away == away)]
        o = odds.iloc[0] if len(odds) else None

        def g(c):
            return float(o[c]) if o is not None and c in o and pd.notna(o[c]) else None
        o1x2 = (g("dec_h"), g("dec_d"), g("dec_a")) if g("dec_h") else None
        oou = (g("dec_over25"), g("dec_under25")) if g("dec_over25") else None
        src = "ESPN/Bet365" if o1x2 else None
        # Bültendeki maçlar için güncel iddaa oranları (Nesine); yoksa yukarıdakiler kalır
        import nesine
        nz = nesine.find_odds(date, home, away)
        if nz and nz["o1x2"]:
            o1x2, oou = nz["o1x2"], nz["oou"] or oou
            src = f"iddaa (Nesine, {datetime.now():%H:%M} itibarıyla, MBS {nz['mbs']})"
        elif nesine.last_error and src:
            src += " (Nesine'ye ulaşılamadı)"
        return dict(r=r, mat=mat, fin=fin, mod=mod, upset=upset_info(r), why=explain(r, fin, mod),
                    value=value_table(r, o1x2, oou), o1x2=o1x2, oou=oou, src=src)


# ====================================================================== maç detayı
class DetailView(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, style="Card.TFrame", padding=10)
        self.app = app
        self.info = None
        self.title = ttk.Label(self, text="Bir maç seçin", style="Title.TLabel")
        self.title.grid(row=0, column=0, columnspan=2, sticky="w")
        self.sub = ttk.Label(self, text="", style="Muted.TLabel")
        self.sub.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 6))

        left = ttk.Frame(self, style="Card.TFrame")
        left.grid(row=2, column=0, sticky="nsew", padx=(0, 12))
        right = ttk.Frame(self, style="Card.TFrame")
        right.grid(row=2, column=1, sticky="nsew")
        self.columnconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)

        S = app.scale
        self.bars = tk.Canvas(left, width=int(360 * S), height=int(96 * S), bg=CARD, highlightthickness=0)
        self.bars.pack(anchor="w")
        self.facts = ttk.Label(left, text="", style="Card.TLabel", justify="left")
        self.facts.pack(anchor="w", pady=(4, 6))
        ttk.Label(left, text="Skor olasılıkları (%) — satır: ev golü, sütun: deplasman golü",
                  style="Muted.TLabel").pack(anchor="w")
        self.grid_cv = tk.Canvas(left, width=int(360 * S), height=int(330 * S), bg=CARD, highlightthickness=0)
        self.grid_cv.pack(anchor="w")

        ttk.Label(right, text="Neden?", style="H2.TLabel").pack(anchor="w")
        self.why = tk.Text(right, height=9, wrap="word", font=FONT, bd=0, bg="#f9fafb", padx=8, pady=6)
        self.why.pack(fill="x", pady=(2, 8))
        for k, c in (("ev", C_HOME), ("dep", C_AWAY), ("nötr", "#374151")):
            self.why.tag_configure(k, foreground=c)

        vh = ttk.Frame(right, style="Card.TFrame")
        vh.pack(fill="x")
        ttk.Label(vh, text="Value bet — iddaa oranları otomatik gelir; değiştirip Hesapla'ya basabilirsiniz", style="H2.TLabel").pack(side="left")
        self.odds = {}
        ef = ttk.Frame(right, style="Card.TFrame")
        ef.pack(fill="x", pady=4)
        for i, k in enumerate(("1", "X", "2", "2.5 Üst", "2.5 Alt")):
            ttk.Label(ef, text=k, style="Card.TLabel").grid(row=0, column=2 * i, padx=(0 if i == 0 else 8, 2))
            e = ttk.Entry(ef, width=6)
            e.grid(row=0, column=2 * i + 1)
            self.odds[k] = e
        ttk.Button(ef, text="Hesapla", command=self.recalc_value).grid(row=0, column=10, padx=8)

        cols = ("secim", "adil", "adil_oran", "oran", "ev_adil", "model", "ev_model", "isaret")
        self.vt = ttk.Treeview(right, columns=cols, show="headings", height=5)
        for c, t, w in zip(cols, ("Seçim", "Adil %", "Adil oran", "Oran", "EV (adil)", "Model %", "EV (model)", ""),
                           (70, 60, 70, 55, 75, 65, 80, 70)):
            self.vt.heading(c, text=t)
            self.vt.column(c, width=int(w * S), anchor="center")
        self.vt.tag_configure("value", background="#dcfce7")
        self.vt.pack(fill="x")
        self.vnote = ttk.Label(right, text="", style="Muted.TLabel", wraplength=int(600 * S), justify="left")
        self.vnote.pack(anchor="w", pady=(4, 6))

        bf = ttk.Frame(right, style="Card.TFrame")
        bf.pack(fill="x")
        self.ai_btn = ttk.Button(bf, text="Yapay zekâ analizi", style="Accent.TButton", command=self.run_ai)
        self.ai_btn.pack(side="left")
        from ai_analyst import backend_label, choose_backend
        ollama_exe = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        if choose_backend() == "yerel" and not ollama_exe.exists():
            self.ai_btn.state(["disabled"])
            ttk.Label(bf, text="  Claude anahtarı ya da Ollama gerekli", style="Muted.TLabel").pack(side="left")
        else:
            ttk.Label(bf, text=f"  Motor: {backend_label()}", style="Muted.TLabel").pack(side="left")

    # ------------------------------------------------------------------
    def loading(self, home, away):
        self.title.config(text=f"{home}  -  {away}")
        self.sub.config(text="Hesaplanıyor… (ilk seferde model eğitildiği için ~10-20 sn sürebilir)")

    def show(self, info):
        self.info = info
        r, fin = info["r"], info["fin"]
        self.title.config(text=f"{r.home}  -  {r.away}")
        pick = ["1", "X", "2"][int(np.argmax([fin["p_home"], fin["p_draw"], fin["p_away"]]))]
        self.sub.config(text=f"{pd.Timestamp(r.date):%d.%m.%Y}   •   Tahmin: {pick}   •   En olası skor: "
                             f"{fin['top_scores'][0][0]}")
        self._draw_bars(fin)
        u = info["upset"] or {}
        fav = (r.home if u.get("fav_is_home") else r.away) if u else "-"
        self.facts.config(text=(
            f"Beklenen gol: {fin['xg_home']:.2f} - {fin['xg_away']:.2f}\n"
            f"2.5 Üst: {pct(fin['p_over25'])}    Karşılıklı gol: {pct(fin['p_btts'])}\n"
            f"Sürpriz alarmı: {u.get('seviye', '-').upper()}  ({fav} kaybetme olasılığı "
            f"{pct(u.get('surpriz_olasiligi'))}; bu seviyede geçmişte {pct(u.get('gecmis_isabet'))})"))
        self._draw_grid(info["mat"])
        self.why.config(state="normal")
        self.why.delete("1.0", "end")
        for side, txt in info["why"]:
            mark = {"ev": "▲ ", "dep": "▼ ", "nötr": "• "}[side]
            self.why.insert("end", mark + txt + "\n", side)
        self.why.config(state="disabled")
        for k, e in self.odds.items():
            e.delete(0, "end")
        if info["o1x2"]:
            for k, v in zip(("1", "X", "2"), info["o1x2"]):
                self.odds[k].insert(0, f"{v:.2f}")
        if info["oou"]:
            for k, v in zip(("2.5 Üst", "2.5 Alt"), info["oou"]):
                self.odds[k].insert(0, f"{v:.2f}")
        self._fill_value(info["value"], src=info.get("src"))

    def _draw_bars(self, fin):
        c = self.bars
        c.delete("all")
        S = self.app.scale
        W = int(340 * S)
        vals = [(f"1  ({self.info['r'].home})", fin["p_home"], C_HOME), ("X  (beraberlik)", fin["p_draw"], C_DRAW),
                (f"2  ({self.info['r'].away})", fin["p_away"], C_AWAY)]
        for i, (lab, p, col) in enumerate(vals):
            y, hgt = int((6 + i * 30) * S), int(22 * S)
            c.create_rectangle(0, y, W, y + hgt, fill="#eef0f3", outline="")
            c.create_rectangle(0, y, max(2, W * p), y + hgt, fill=col, outline="")
            # yazı çubuğun rengine bakmaksızın okunsun: beyaz zeminli etiket
            t = c.create_text(int(8 * S), y + hgt / 2, text=f"{lab}", anchor="w", fill="#111827",
                              font=("Segoe UI", 9, "bold"))
            x0, y0, x1, y1 = c.bbox(t)
            c.tag_lower(c.create_rectangle(x0 - 3, y0, x1 + 3, y1, fill="white", outline=""), t)
            c.create_text(W - 4, y + hgt / 2, text=f"%{100 * p:.1f}", anchor="e", fill="#111827",
                          font=("Segoe UI", 10, "bold"))

    def _draw_grid(self, mat):
        c = self.grid_cv
        c.delete("all")
        S = self.app.scale
        n, cell, off = 6, int(50 * S), int(26 * S)
        mx = mat[:n, :n].max()
        for j in range(n):
            c.create_text(off + j * cell + cell / 2, int(10 * S), text=str(j), fill=MUTED, font=FONT)
        for i in range(n):
            c.create_text(int(10 * S), off + i * cell + cell / 2 - 8, text=str(i), fill=MUTED, font=FONT)
            for j in range(n):
                v = mat[i, j]
                t = v / mx if mx > 0 else 0
                col = C_HOME if i > j else C_AWAY if i < j else "#6b7280"
                r0, g0, b0 = (int(col[k:k + 2], 16) for k in (1, 3, 5))
                rgb = tuple(int(255 - (255 - x) * (0.12 + 0.88 * t)) for x in (r0, g0, b0))
                x0, y0 = off + j * cell, off + i * cell - 8
                c.create_rectangle(x0, y0, x0 + cell - 2, y0 + cell - 2, fill="#%02x%02x%02x" % rgb, outline="")
                c.create_text(x0 + cell / 2 - 1, y0 + cell / 2 - 1, text=f"{100 * v:.1f}",
                              fill="white" if t > 0.55 else "#111827", font=("Segoe UI", 9))

    def _fill_value(self, vt, src=None):
        from value import VALUE_MIN_EV
        self.vt.delete(*self.vt.get_children())
        for t in vt.itertuples():
            flag = "VALUE" if not pd.isna(t.ev_adil) and t.ev_adil > VALUE_MIN_EV else ""
            self.vt.insert("", "end", tags=("value",) if flag else (), values=(
                t.secim, f"%{100 * t.p_adil:.1f}", f"{t.adil_oran:.2f}", f"{t.oran:.2f}" if t.oran else "-",
                f"{100 * t.ev_adil:+.1f}%" if not pd.isna(t.ev_adil) else "-", f"%{100 * t.p_model:.1f}",
                f"{100 * t.ev_model:+.1f}%" if not pd.isna(t.ev_model) else "-", flag))
        note = ("Oranlar: " + (src or "sizin girdiğiniz") + ". " if src or vt.oran.notna().any() else "Oran girilmedi. ")
        R = REPORTS / "value_backtest.csv"
        if R.exists():
            d = pd.read_csv(R)
            row = d[d.strateji.str.contains("GÖRÜLMEMİŞ", regex=False)
                    & d.strateji.str.contains("Adil (piyasa+model) | 1X2 EV > %5", regex=False)]
            if len(row):
                x = row.iloc[0]
                note += (f"Geçmiş test (Bet365'e karşı, adil, EV>%5): ROI %{x['ROI_%']:+.1f} "
                         f"(aralık %{x['ROI_alt_%']:+.1f} / %{x['ROI_ust_%']:+.1f}). ")
        note += "Kanıtlanmış avantaj yok; EV(model) geçmişte zarar ettirdi. Kâr garantisi yoktur."
        self.vnote.config(text=note)

    def recalc_value(self):
        if not self.info:
            return
        from value import value_table

        def num(k):
            try:
                return float(self.odds[k].get().replace(",", "."))
            except ValueError:
                return None
        o1 = tuple(num(k) for k in ("1", "X", "2"))
        oo = tuple(num(k) for k in ("2.5 Üst", "2.5 Alt"))
        vt = value_table(self.info["r"], o1 if all(o1) else None, oo if all(oo) else None)
        self._fill_value(vt)

    def run_ai(self):
        if not self.info:
            return
        r = self.info["r"]
        win = tk.Toplevel(self)
        win.title(f"Yapay zekâ analizi — {r.home} - {r.away}")
        win.geometry("760x620")
        txt = tk.Text(win, wrap="word", font=FONT, padx=10, pady=10)
        txt.pack(fill="both", expand=True)
        from ai_analyst import backend_label
        txt.insert("end", f"{backend_label()} haberleri okuyor ve analiz ediyor… (yerel modelde ~20-60 sn, "
                          f"Claude'da 1-2 dk)\n")

        def work():
            from ai_analyst import analyze
            return analyze(r, self.app.data.matches())

        def done(a):
            txt.delete("1.0", "end")
            if "hata" in a:
                txt.insert("end", a["hata"])
                return
            d = a["duzeltilmis"]
            o = d.get("onceki", d)
            txt.insert("end", f"Motor: {a.get('motor', '-')}\n\n" + a["ozet"] + "\n\nÖnemli etkenler:\n")
            for e in a["onemli_etkenler"]:
                txt.insert("end", f"  [{e.get('onem', '-')}] ({e.get('lehine', '-')} lehine) {e.get('etken', '')}\n")
            txt.insert("end", f"\nGerekçe: {a['gerekce']}\n\nYZ düzeltmesi: ev x{a['ev_carpan_uygulanan']:.2f}, "
                              f"deplasman x{a['dep_carpan_uygulanan']:.2f} (güven: {a['guven']})\n"
                              f"1/X/2 önce:  {pct(o['p_home'])} / {pct(o['p_draw'])} / {pct(o['p_away'])}\n"
                              f"1/X/2 sonra: {pct(d['p_home'])} / {pct(d['p_draw'])} / {pct(d['p_away'])}\n"
                              "Not: YZ düzeltmesi geriye dönük test edilmedi; çarpanlar ±%15 ile sınırlı.\n"
                              f"YZ skor tahmini: {a['en_olasi_skor']}\n\nRiskler: {'; '.join(a['riskler'])}\n\n"
                              f"Haber araştırması:\n{a['haberler']}")
        self.app.worker.run(work, done)


# ====================================================================== sekmeler
class WeekTab(ttk.Frame):
    COLS = ("zaman", "ev", "dep", "p1", "px", "p2", "tahmin", "skor", "ust", "surpriz", "value")

    def __init__(self, master, app):
        super().__init__(master, style="TFrame", padding=8)
        self.app = app
        top = ttk.Frame(self)
        top.pack(fill="x")
        ttk.Label(top, text="Yaklaşan maçlar (son kaydedilen tahminler)", style="H2.TLabel").pack(side="left")
        ttk.Button(top, text="Tahminleri yenile", command=self.refresh_predictions).pack(side="right")
        self.tree = ttk.Treeview(self, columns=self.COLS, show="headings", height=8)
        heads = ("Tarih / saat", "Ev sahibi", "Deplasman", "1", "X", "2", "Tahmin", "Skor", "2.5 Üst", "Sürpriz", "Value")
        widths = (110, 150, 150, 50, 50, 50, 60, 60, 65, 90, 120)
        for c, h, w in zip(self.COLS, heads, widths):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=int(w * app.scale), anchor="center" if c not in ("ev", "dep") else "w")
        self.tree.tag_configure("orta", background="#fef9c3")
        self.tree.tag_configure("yüksek", background="#ffedd5")
        self.tree.tag_configure("çok yüksek", background="#fee2e2")
        self.tree.pack(fill="x", pady=6)
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        self.detail = DetailView(self, app)
        self.detail.pack(fill="both", expand=True)
        self.rows = {}
        self.load()

    def load(self):
        self.tree.delete(*self.tree.get_children())
        self.rows = {}
        p = ROOT / "data" / "predictions_log.csv"
        if not p.exists():
            return
        from tracker import load_log
        d = load_log()
        d["tr"] = d.kickoff_utc + pd.Timedelta(hours=3)
        d = d[d.tr >= pd.Timestamp.now() - pd.Timedelta(hours=3)].sort_values("tr")
        for r in d.itertuples():
            v = f"{r.value_pick} @ {r.value_odds:.2f}" if isinstance(getattr(r, "value_pick", None), str) else ""
            iid = self.tree.insert("", "end", tags=(r.upset_level,), values=(
                f"{r.tr:%d.%m %H:%M}", r.home, r.away, pct(r.p_home), pct(r.p_draw), pct(r.p_away), r.pick,
                r.top_score, pct(r.p_over25), r.upset_level, v))
            self.rows[iid] = r
        # tablo yüksekliği maç sayısına göre (detay paneline yer kalsın)
        self.tree.configure(height=max(3, min(len(self.rows), 7)))

    def refresh_predictions(self):
        self.app.run_script(["src/tracker.py", "kaydet", "--gun", "7"], "Tahminler yenileniyor", after=self.load)

    def on_select(self, _=None):
        sel = self.tree.selection()
        if not sel:
            return
        r = self.rows[sel[0]]
        season = f"{r.date.year if r.date.month >= 7 else r.date.year - 1}"
        season = f"{season}-{(int(season) + 1) % 100:02d}"
        self.detail.loading(r.home, r.away)
        self.app.worker.run(lambda: self.app.data.fixture_row(season, r.date, r.home, r.away), self.detail.show)


class PredictTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=8)
        self.app = app
        f = ttk.Frame(self)
        f.pack(fill="x")
        teams = app.data.teams()
        ttk.Label(f, text="Ev sahibi").pack(side="left")
        self.h = ttk.Combobox(f, values=teams, width=22, state="readonly")
        self.h.pack(side="left", padx=6)
        ttk.Label(f, text="Deplasman").pack(side="left")
        self.a = ttk.Combobox(f, values=teams, width=22, state="readonly")
        self.a.pack(side="left", padx=6)
        ttk.Label(f, text="Tarih (GG.AA.YYYY)").pack(side="left")
        self.d = ttk.Entry(f, width=12)
        self.d.insert(0, datetime.now().strftime("%d.%m.%Y"))
        self.d.pack(side="left", padx=6)
        ttk.Button(f, text="Tahmin et", style="Accent.TButton", command=self.go).pack(side="left", padx=6)
        self.detail = DetailView(self, app)
        self.detail.pack(fill="both", expand=True, pady=8)

    def go(self):
        home, away = self.h.get(), self.a.get()
        if not home or not away or home == away:
            messagebox.showwarning("Eksik", "İki farklı takım seçin.")
            return
        try:
            d = pd.Timestamp(datetime.strptime(self.d.get().strip(), "%d.%m.%Y"))
        except ValueError:
            messagebox.showwarning("Tarih", "Tarih GG.AA.YYYY biçiminde olmalı.")
            return
        y = d.year if d.month >= 7 else d.year - 1
        season = f"{y}-{(y + 1) % 100:02d}"
        self.detail.loading(home, away)
        self.app.worker.run(lambda: self.app.data.fixture_row(season, d, home, away), self.detail.show)


class LiveTab(ttk.Frame):
    COLS = ("zaman", "mac", "durum", "skor", "tahmin", "sonuc", "skor_t", "surpriz", "value")

    def __init__(self, master, app):
        super().__init__(master, padding=8)
        self.app = app
        top = ttk.Frame(self)
        top.pack(fill="x")
        ttk.Label(top, text="Tahminler ve gerçek sonuçlar", style="H2.TLabel").pack(side="left")
        self.stamp = ttk.Label(top, text="", style="Muted.TLabel")
        self.stamp.pack(side="left", padx=10)
        ttk.Button(top, text="Yenile", command=self.refresh).pack(side="right")
        self.summary = ttk.Label(self, text="", style="Summary.TLabel", justify="left", padding=10)
        self.summary.pack(fill="x", pady=6)
        self.tree = ttk.Treeview(self, columns=self.COLS, show="headings")
        heads = ("Maç saati", "Maç", "Durum", "Skor", "Tahmin", "Sonuç", "Tahmini skor", "Sürpriz", "Value")
        widths = (100, 280, 90, 60, 70, 80, 90, 90, 130)
        for c, h, w in zip(self.COLS, heads, widths):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=int(w * app.scale), anchor="w" if c == "mac" else "center")
        self.tree.tag_configure("hit", background="#dcfce7")
        self.tree.tag_configure("miss", background="#fee2e2")
        self.tree.tag_configure("live", background="#fef9c3")
        self.tree.pack(fill="both", expand=True)
        self.after_id = None

    def refresh(self):
        from tracker import status_frames
        self.stamp.config(text="yenileniyor…")
        self.app.worker.run(status_frames, self.show)

    def show(self, res):
        log, done, oz = res
        self.tree.delete(*self.tree.get_children())
        self.stamp.config(text=f"son yenileme {datetime.now():%H:%M:%S}")
        any_live = False
        if log.empty:
            self.summary.config(text="Henüz kayıtlı tahmin yok. Tahminler maçtan önce otomatik kaydedilir.")
            return
        dmap = done.set_index("espn_id") if len(done) else None
        for r in log.sort_values("kickoff_utc", ascending=False).itertuples():
            st = r.state
            score = f"{int(r.hs)}-{int(r.as_)}" if pd.notna(r.hs) else ""
            vtxt = f"{r.value_pick} @ {r.value_odds:.2f}" if isinstance(getattr(r, "value_pick", None), str) else ""
            if st == "in":
                any_live = True
                cur = "1" if r.hs > r.as_ else "X" if r.hs == r.as_ else "2"
                tag, res_txt, durum = "live", ("şu an tutuyor" if cur == r.pick else "şu an tutmuyor"), r.detail
            elif st == "post" and dmap is not None and r.espn_id in dmap.index:
                dd = dmap.loc[r.espn_id]
                tag = "hit" if dd.hit_1x2 else "miss"
                res_txt = ("✓ " if dd.hit_1x2 else "✗ ") + dd.res
                durum = "bitti"
                if isinstance(getattr(r, "value_pick", None), str) and pd.notna(dd.get("value_kar")):
                    vtxt += f"  ({dd.value_kar:+.2f})"
            else:
                tag, res_txt, durum = "", "", "başlamadı" if st == "pre" else (r.detail or "")
            self.tree.insert("", "end", tags=(tag,), values=(
                f"{r.kickoff_tr:%d.%m %H:%M}", f"{r.home} - {r.away}", durum, score, f"{r.pick} ({pct(max(r.p_home, r.p_draw, r.p_away))})",
                res_txt, r.top_score, r.upset_level, vtxt))
        if oz:
            n = oz["mac"]
            txt = (f"Biten {n} maç:   Maç sonucu isabeti {pct(oz['isabet_1x2'])}  (geçmiş testte ~%57)   •   "
                   f"Tam skor {pct(oz['isabet_skor'])}  (~%12)   •   2.5 alt/üst {pct(oz['isabet_o25'])}  (~%57)   •   "
                   f"Favori kaybetti: {oz['favori_kaybetti']}")
            if "value_bahis" in oz:
                txt += (f"\nValue bet (1 birim): {oz['value_bahis']} bahis, kâr {oz['value_kar']:+.2f} birim, "
                        f"ROI %{100 * oz['value_roi']:+.1f}")
            if n < 50:
                txt += f"\nNot: {n} maç az bir örnek; oranlar ±%{100 * 1.96 * np.sqrt(0.25 / n):.0f} oynayabilir."
            self.summary.config(text=txt)
        else:
            self.summary.config(text="Henüz biten kayıtlı maç yok; değerlendirme maçlar bittikçe birikecek.")
        if self.after_id:
            self.after_cancel(self.after_id)
        if any_live:  # canlı maç varsa her dakika yenile
            self.after_id = self.after(60_000, self.refresh)


class TableTab(ttk.Frame):
    """CSV'den tablo gösteren basit sekme."""

    def __init__(self, master, app, title, loader, button=None):
        super().__init__(master, padding=8)
        self.app, self.loader = app, loader
        top = ttk.Frame(self)
        top.pack(fill="x")
        ttk.Label(top, text=title, style="H2.TLabel").pack(side="left")
        if button:
            ttk.Button(top, text=button[0], command=button[1]).pack(side="right")
        self.note = ttk.Label(self, text="", style="Muted.TLabel", wraplength=1100, justify="left")
        self.note.pack(anchor="w", pady=4)
        self.frame = ttk.Frame(self)
        self.frame.pack(fill="both", expand=True)
        self.load()

    def load(self):
        for w in self.frame.winfo_children():
            w.destroy()
        blocks = self.loader()
        for head, df, note in blocks:
            if head:
                ttk.Label(self.frame, text=head, style="H3.TLabel").pack(anchor="w", pady=(8, 2))
            if df is None or df.empty:
                ttk.Label(self.frame, text=note or "Veri yok.", style="Muted.TLabel").pack(anchor="w")
                continue
            tv = ttk.Treeview(self.frame, columns=list(df.columns), show="headings", height=min(len(df), 18))
            for c in df.columns:
                tv.heading(c, text=c)
                tv.column(c, width=int(self.app.scale * max(70, min(260, 9 * max(len(str(c)), df[c].astype(str).str.len().max())))),
                          anchor="w" if df[c].dtype == object else "center")
            for row in df.itertuples(index=False):
                tv.insert("", "end", values=list(row))
            tv.pack(fill="x")
            if note:
                ttk.Label(self.frame, text=note, style="Muted.TLabel", wraplength=1100, justify="left").pack(anchor="w")


def sim_loader():
    files = sorted(REPORTS.glob("simulation_*.csv"))
    if not files:
        return [(None, None, "Simülasyon henüz çalıştırılmadı.")]
    df = pd.read_csv(files[-1])
    when = datetime.fromtimestamp(files[-1].stat().st_mtime)
    return [(f"{files[-1].stem.split('_')[1]} sezonu — 10.000 simülasyon ({when:%d.%m.%Y %H:%M})", df,
             "Yüzdeler: kalan maçlar model olasılıklarına göre 10.000 kez oynatıldığında görülen sıklık.")]


def perf_loader():
    out = []
    S = REPORTS / "backtest_summary.csv"
    if S.exists():
        d = pd.read_csv(S)
        d = d[d.season == "TOPLAM"][["model", "n", "acc_1x2", "logloss_1x2", "rps", "acc_o25", "exact_score_hit"]]
        d.columns = ["Model", "Maç", "1X2 doğruluk", "Log loss (düşük iyi)", "RPS", "2.5 Ü/A doğruluk", "Tam skor"]
        for c in ("1X2 doğruluk", "2.5 Ü/A doğruluk", "Tam skor"):
            d[c] = (100 * d[c]).round(1).astype(str) + "%"
        for c in ("Log loss (düşük iyi)", "RPS"):
            d[c] = d[c].round(3)
        out.append(("Geçmiş test: maç sonucu (her maç yalnızca önceki verilerle tahmin edildi)", d,
                    "NİHAİ = bahis piyasası + model (uygulamanın kullandığı). Bahis piyasası referans içindir."))
    L = REPORTS / "upset_levels.json"
    if L.exists():
        lv = pd.DataFrame(json.loads(L.read_text(encoding="utf-8")))
        lv = pd.DataFrame({"Alarm": lv.seviye, "Maç": lv.mac, "Maçların payı": (100 * lv.maclarin_payi).round(0).astype(int).astype(str) + "%",
                           "Favori gerçekte kaybetti": (100 * lv.gercek_surpriz).round(1).astype(str) + "%"})
        out.append(("Sürpriz alarmı (favorinin kaybetmesi)", lv, None))
    V = REPORTS / "value_backtest.csv"
    if V.exists():
        v = pd.read_csv(V)
        v = v[v.strateji.str.contains("GÖRÜLMEMİŞ") & (v.strateji.str.contains("Referans") | v.strateji.str.contains("EV > %5"))]
        v = v[["strateji", "bahis", "isabet_%", "ROI_%", "ROI_alt_%", "ROI_ust_%"]].round(1)
        v.columns = ["Strateji (görülmemiş sezonlar)", "Bahis", "İsabet %", "ROI %", "Alt %", "Üst %"]
        out.append(("Value bet geçmiş testi (1 birim sabit bahis, Bet365/ESPN oranları)", v,
                    "Alt/Üst: %95 güven aralığı. Aralık sıfırı kapsıyorsa sonuç şans olabilir."))
    return out or [(None, None, "Henüz test sonucu yok.")]


class LogTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=8)
        top = ttk.Frame(self)
        top.pack(fill="x")
        ttk.Label(top, text="Otomatik güncelleme günlüğü", style="H2.TLabel").pack(side="left")
        ttk.Button(top, text="Yenile", command=self.load).pack(side="right")
        self.txt = tk.Text(self, wrap="none", font=("Consolas", 9), bg="#0f172a", fg="#e2e8f0", padx=8, pady=6)
        self.txt.pack(fill="both", expand=True, pady=6)
        self.load()

    def load(self):
        p = REPORTS / "daily_log.txt"
        self.txt.delete("1.0", "end")
        self.txt.insert("end", p.read_text(encoding="utf-8")[:60000] if p.exists() else "Günlük yok.")

    def append(self, s):
        self.txt.insert("1.0", s + "\n")


# ====================================================================== ana pencere
def _dpi_aware():
    """Yüksek çözünürlüklü ekranlarda bulanıklığı önler; ölçek katsayısını döndürür."""
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
        return ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100
    except Exception:  # noqa: BLE001
        return 1.0


class App(tk.Tk):
    def __init__(self):
        scale = _dpi_aware()
        super().__init__()
        self.title("Nostradamus — Süper Lig Tahmin")
        self.scale = scale
        self.tk.call("tk", "scaling", 96 * scale / 72)
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(int(1320 * scale), sw - 40), min(int(860 * scale), sh - 80)
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{max(0, (sh - h) // 2 - 20)}")
        self.minsize(int(1000 * scale), int(680 * scale))
        self.configure(bg=BG)
        ico = ROOT / "assets" / "nostradamus.ico"
        if ico.exists():
            try:
                self.iconbitmap(str(ico))
            except tk.TclError:
                pass
        self._style()
        self.worker = Worker(self)
        self.data = Data()

        head = tk.Frame(self, bg=HEAD, height=56)
        head.pack(fill="x")
        logo = ROOT / "assets" / "nostradamus.png"
        if logo.exists():
            try:
                from PIL import Image, ImageTk
                sz = int(44 * self.scale)
                self._logo = ImageTk.PhotoImage(Image.open(logo).resize((sz, sz), Image.LANCZOS))
                tk.Label(head, image=self._logo, bg=HEAD).pack(side="left", padx=(12, 0), pady=6)
            except Exception:  # noqa: BLE001
                pass
        tk.Label(head, text="Nostradamus", bg=HEAD, fg="white", font=("Segoe UI", 16, "bold")).pack(side="left", padx=10, pady=10)
        tk.Label(head, text="Süper Lig skor tahmini • piyasa + model", bg=HEAD, fg="#cbd5e1",
                 font=("Segoe UI", 10)).pack(side="left")
        self.busy = tk.Label(head, text="", bg=HEAD, fg="#fde68a", font=("Segoe UI", 10))
        self.busy.pack(side="right", padx=12)
        ttk.Button(head, text="Tam güncelle", command=lambda: self.update_data(False)).pack(side="right", padx=4)
        ttk.Button(head, text="Hızlı güncelle", command=lambda: self.update_data(True)).pack(side="right", padx=4)

        self.status = tk.Label(self, text=self._last_update(), bg=BG, fg=MUTED, anchor="w", font=("Segoe UI", 9))
        self.status.pack(side="bottom", fill="x", padx=10, pady=(0, 6))
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=8, pady=8)
        self.week = WeekTab(self.nb, self)
        self.pred = PredictTab(self.nb, self)
        self.live = LiveTab(self.nb, self)
        self.sim = TableTab(self.nb, self, "Sezon simülasyonu", sim_loader,
                            ("Simülasyonu çalıştır", lambda: self.run_script(["src/simulate.py"], "Simülasyon",
                                                                             after=lambda: self.sim.load())))
        self.perf = TableTab(self.nb, self, "Model başarısı", perf_loader)
        self.log = LogTab(self.nb, self)
        for t, name in ((self.week, "Bu Hafta"), (self.pred, "Maç Tahmini"), (self.live, "Canlı Takip"),
                        (self.sim, "Sezon Simülasyonu"), (self.perf, "Model Başarısı"), (self.log, "Günlük")):
            self.nb.add(t, text=f"  {name}  ")
        self.nb.bind("<<NotebookTabChanged>>", self._tab_changed)

    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", font=FONT, background=BG)
        s.configure("TFrame", background=BG)
        s.configure("Card.TFrame", background=CARD)
        s.configure("TLabel", background=BG)
        s.configure("Card.TLabel", background=CARD)
        s.configure("Title.TLabel", background=CARD, font=("Segoe UI", 15, "bold"))
        s.configure("H2.TLabel", background=CARD, font=("Segoe UI", 11, "bold"))
        s.configure("H3.TLabel", background=BG, font=("Segoe UI", 11, "bold"))
        s.configure("Muted.TLabel", background=CARD, foreground=MUTED, font=("Segoe UI", 9))
        s.configure("Summary.TLabel", background="#e0f2fe", font=("Segoe UI", 10))
        s.configure("Treeview", rowheight=int(24 * self.scale), font=FONT)
        s.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
        s.configure("TNotebook.Tab", padding=(10, 6), font=("Segoe UI", 10))
        s.configure("Accent.TButton", foreground="white", background=ACCENT)
        s.map("Accent.TButton", background=[("active", "#15803d")])
        for st in ("H2.TLabel", "Muted.TLabel"):
            pass

    def _last_update(self):
        p = REPORTS / "daily_log.txt"
        if p.exists():
            first = p.read_text(encoding="utf-8").splitlines()[:1]
            if first:
                return "Son otomatik güncelleme: " + first[0].strip("= ").strip()
        return "Henüz otomatik güncelleme yok."

    def _tab_changed(self, _=None):
        cur = self.nb.nametowidget(self.nb.select())
        if cur is self.live and not self.live.tree.get_children():
            self.live.refresh()

    def run_script(self, args, label, after=None):
        self.busy.config(text=f"⏳ {label}…")

        def work():
            r = subprocess.run([str(PY), *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return r

        def done(r):
            self.busy.config(text="")
            self.log.append(f"[{datetime.now():%H:%M}] {label}: {'tamam' if r.returncode == 0 else 'HATA'}\n"
                            + (r.stdout or "")[-3000:] + (r.stderr or "")[-2000:])
            if r.returncode != 0:
                messagebox.showerror("Hata", f"{label} başarısız.\n\n{(r.stderr or r.stdout)[-1500:]}")
            elif after:
                after()

        self.worker.run(work, done)

    def update_data(self, quick):
        args = ["src/daily.py"] + (["--hizli"] if quick else [])
        label = "Hızlı güncelleme" if quick else "Tam güncelleme (birkaç dakika)"

        def after():
            self.week.load()
            self.perf.load()
            self.sim.load()
            self.log.load()
            self.status.config(text=self._last_update())
        self.run_script(args, label, after)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ekran-goruntusu")
    ap.add_argument("--sekme", type=int, default=0)
    ap.add_argument("--mac", action="store_true", help="ilk maçı seç (test)")
    a = ap.parse_args()
    app = App()
    if a.ekran_goruntusu:
        app.nb.select(a.sekme)
        state = {"t": 0}

        def snap():
            from PIL import ImageGrab
            app.update()
            x, y = app.winfo_rootx(), app.winfo_rooty()
            ImageGrab.grab(bbox=(x, y, x + app.winfo_width(), y + app.winfo_height())).save(a.ekran_goruntusu)
            app.destroy()

        def wait():
            state["t"] += 1
            ready = (not a.mac) or app.week.detail.info is not None or app.pred.detail.info is not None
            if a.sekme == 2:
                ready = bool(app.live.tree.get_children()) or state["t"] > 40
            if ready or state["t"] > 120:
                app.after(800, snap)
            else:
                app.after(500, wait)

        def start():
            if a.mac and a.sekme == 0 and app.week.tree.get_children():
                first = app.week.tree.get_children()[0]
                app.week.tree.selection_set(first)
            if a.sekme == 2:
                app.live.refresh()
            app.after(500, wait)
        app.after(1500, start)
    app.mainloop()


if __name__ == "__main__":
    main()
