@echo off
echo Ativando ambiente virtual...
python -m venv venv
call venv\Scripts\activate
pip install -r app\requirements.txt
pip uninstall opencv-python-headless -y
pip uninstall opencv-python -y
pip install opencv-python
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121