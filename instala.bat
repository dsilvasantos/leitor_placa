@echo off
echo Ativando ambiente virtual...
python -m venv venv
call venv\Scripts\activate
pip uninstall opencv-python-headless -y
pip install -r app\requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics