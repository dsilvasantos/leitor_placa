#!/bin/bash
echo "Ativando ambiente virtual..."
python3 -m venv venv
source venv/bin/activate
pip install -r app/requirements.txt

echo "Iniciando API Flask..."
cd app
python api.py
