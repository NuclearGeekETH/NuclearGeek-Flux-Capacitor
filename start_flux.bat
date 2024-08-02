@echo off

If Not Exist "%~dp0%\.venv\Scripts\activate.bat" (
	python -m venv .venv
	call "%~dp0%\.venv\Scripts\activate"
	pip install git+https://github.com/huggingface/diffusers.git
	pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	pip install -r requirements.txt
)

call "%~dp0%\.venv\Scripts\activate"
python main.py
pause