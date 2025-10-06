@echo off
echo ================================================
echo WASTE CLASSIFICATION - QUICK START
echo ================================================
echo.

echo Step 1: Checking setup...
python check_setup.py
echo.

echo ================================================
echo To continue:
echo ================================================
echo.
echo 1. Install dependencies:
echo    pip install -r requirements.txt
echo.
echo 2. Organize your dataset in data/train/ folder:
echo    data/train/cardboard/*.jpg
echo    data/train/glass/*.jpg
echo    data/train/metal/*.jpg
echo    data/train/paper/*.jpg
echo    data/train/plastic/*.jpg
echo    data/train/trash/*.jpg
echo.
echo 3. Train the model:
echo    python train_model.py
echo.
echo 4. Start the web application:
echo    python app.py
echo.
echo 5. Open browser to http://localhost:5000
echo.
echo ================================================

pause
