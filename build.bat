@echo off
python -m PyInstaller ^
    --noconfirm --log-level WARN ^
    --onefile --nowindow ^
    --hidden-import xlrd ^
    --hidden-import openpyxl ^
    --name NeuroChaT ^
    --icon NeuroChaT.ico ^
    --additional-hooks-dir=hooks ^
    cli.py