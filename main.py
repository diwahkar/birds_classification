import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from predict import predict_image
import os
import shutil

app = FastAPI(
    title="Bird Species Classifier API",
    description="Upload an image of a bird to get its predicted species and confidence.",
    version="1.0.0"
)

UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/api")
async def read_api_root():
    """
    Root endpoint to confirm the API is running.
    """
    return {"message": "Welcome to the Bird Species Classifier API! Visit /docs for more info."}

@app.post("/predict_bird/")
async def predict_bird_image(file: UploadFile = File(...)):
    """
    **Predicts the species of a bird from an uploaded image.**

    **Parameters:**
    - `file`: The image file to be uploaded. Accepted formats typically include JPG, PNG.

    **Returns:**
    - A JSON object containing:
        - `predicted_class`: The most likely bird species.
        - `confidence`: The confidence score for the predicted class (0.0 to 1.0).
        - `all_probabilities`: A dictionary of probabilities for all known bird classes.

    **Raises:**
    - `HTTPException`: If the file cannot be processed or an internal error occurs.
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    file_location = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"File saved temporarily at: {file_location}")

        prediction_results = predict_image(file_location)

        return prediction_results

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed due to an internal error: {e}")
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)
            print(f"Cleaned up temporary file: {file_location}")


app.mount("/", StaticFiles(directory="static_files", html=True), name="static")
