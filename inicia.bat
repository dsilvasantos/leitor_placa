@echo off
echo Ativando ambiente virtual...
python -m venv venv
call venv\Scripts\activate
cd app
python main.py
