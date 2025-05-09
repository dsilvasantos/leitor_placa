@echo off
echo Ativando ambiente virtual...
python -m venv venv
call venv\Scripts\activate
pip uninstall opencv-python-headless -y
pip install -r app\requirements.txt
pip install -U ultralytics torch torchvision