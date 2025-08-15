@echo off
echo Desinstalando ambiente virtual...
python -m venv venv
call venv\Scripts\activate
pip uninstall flask -y
pip uninstall psycopg2-binary -y
pip uninstall ultralytics -y
pip uninstall requests -y
pip uninstall pytesseract -y
pip uninstall torch torchvision torchaudio -y
pip uninstall opencv-python
pip uninstall opencv-python-headless
pip uninstall opencv-contrib-python

