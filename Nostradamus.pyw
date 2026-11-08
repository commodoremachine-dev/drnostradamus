"""Nostradamus masaüstü uygulaması — çift tıklayarak başlatın (konsol penceresi açılmaz)."""
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

try:
    import app
    app.main()
except Exception:  # pencere açılamazsa hatayı göster
    err = traceback.format_exc()
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "app_hata.txt").write_text(err, encoding="utf-8")
    try:
        import tkinter as tk
        from tkinter import messagebox
        r = tk.Tk()
        r.withdraw()
        messagebox.showerror("Nostradamus başlatılamadı", err[-1500:])
    except Exception:
        pass
