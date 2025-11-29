@echo off
echo Installing required dependencies...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
echo Installation complete!
pause
