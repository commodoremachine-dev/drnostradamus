@echo off
rem Nostradamus - Super Lig tahmin uygulamasi gunluk guncelleme
rem Kullanim: guncelle.bat         (tam)
rem           guncelle.bat hizli   (hizli)
cd /d "%~dp0"
if /I "%1"=="hizli" (
    ".venv\Scripts\python.exe" src\daily.py --hizli
) else (
    ".venv\Scripts\python.exe" src\daily.py
)
