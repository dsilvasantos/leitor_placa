@echo off
echo Ativando ambiente virtual...
python -m venv venv
call venv\Scripts\activate
echo Iniciando API Flask...
cd app
python detecta.py
