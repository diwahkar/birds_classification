# Installation
    pip install -r requirements.txt


# Run
    uvicorn main:app


# Project Flow
1. Data Preparation
2. Model Train
3. Model Predict


# Data Preparation
    python data_preparation.py
This will read image path from DATA_ROOT_DIR, create label and label id, and save in .csv file.


# Model Train
    python train.py
This reads the csv file, get the image path and train the model


# Model Predict
    python predict.py

Give image path to predict in the prompt, after running this file


Note:
Current model predicts limited species of bird.
This model can be trained to classify any images.
In order to for classification, DATA_ROOT_DIR should contain sub_folders named after label.
These sub-folders should contain image data34



# Link to images resources and video demo
https://drive.google.com/drive/folders/1My2pwsYRM3TtyOATCnbGMYQ5yNuIy8rN
