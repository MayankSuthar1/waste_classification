@echo off
echo ================================================
echo WASTE CLASSIFICATION - DATASET SETUP
echo WasteClassificationNeuralNetwork (9 Classes)
echo ================================================
echo.

echo Checking if dataset is already downloaded...
if exist "data\WasteClassificationNeuralNetwork\WasteImagesDataset\" (
    echo Dataset found! Running verification...
    python setup_dataset.py
    goto :training
)

echo Dataset not found. Downloading from GitHub...
echo This may take a few minutes (approximately 200MB download)
echo.

cd data
git clone https://github.com/cardstdani/WasteClassificationNeuralNetwork.git
cd ..

if exist "data\WasteClassificationNeuralNetwork\WasteImagesDataset\" (
    echo.
    echo ================================================
    echo Dataset downloaded successfully!
    echo ================================================
    echo.
    python setup_dataset.py
) else (
    echo.
    echo ================================================
    echo ERROR: Dataset download failed
    echo ================================================
    echo.
    echo Please download manually:
    echo 1. Visit: https://github.com/cardstdani/WasteClassificationNeuralNetwork
    echo 2. Download ZIP and extract to: data\WasteClassificationNeuralNetwork\
    echo.
    pause
    exit /b 1
)

:training
echo.
echo ================================================
echo Ready to train?
echo ================================================
echo.
echo Press any key to start training the model...
echo (This will take 10-30 minutes depending on your hardware)
echo.
pause

python train_model.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================
    echo Training completed successfully!
    echo ================================================
    echo.
    echo Press any key to launch the web application...
    pause
    python app.py
) else (
    echo.
    echo ================================================
    echo Training failed. Check the error messages above.
    echo ================================================
    pause
)
