@echo off
echo Setting up TiRex Fine-tuning for RTX 4070 Super on Windows 11
echo ================================================================

echo.
echo Step 1: Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo Step 2: Installing other requirements...
pip install -r requirements.txt

echo.
echo Step 3: Cloning and installing TiRex...
if not exist "tirex" (
    git clone https://github.com/NX-AI/tirex.git
)
cd tirex
pip install -e .
cd ..

echo.
echo Step 4: Testing CUDA installation...
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

echo.
echo Setup complete! You can now run:
echo   python train.py
echo.
pause