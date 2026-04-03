from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import asyncio
import random

from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="CerebroAI Backend")


def _parse_cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:3000")
    parts = [o.strip() for o in raw.split(",") if o.strip()]
    return parts if parts else ["http://localhost:3000"]


# Configure CORS for Next.js (origins from .env)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = os.getenv("TEMP_DIR", "temp")
os.makedirs(TEMP_DIR, exist_ok=True)


@app.post("/predict")
async def predict_mri(
    file: UploadFile = File(...),
    model_type: str = Form(...),
):
    try:
        if model_type not in ("binary", "multiclass"):
            raise HTTPException(
                status_code=400,
                detail='model_type must be "binary" or "multiclass"',
            )

        # 1. Validate file type
        if not (file.filename.endswith(".nii") or file.filename.endswith(".nii.gz")):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only .nii or .nii.gz allowed.",
            )

        # 2. Save the uploaded file temporarily
        file_location = os.path.join(TEMP_DIR, file.filename)
        with open(file_location, "wb+") as file_object:
            file_object.write(await file.read())

        # 3. Simulate processing time (2.5 seconds)
        print(f"Received {file.filename} for {model_type} analysis...")
        await asyncio.sleep(2.5)

        # 4. Generate mock results based on the requested model type
        if model_type == "binary":
            # Only AD or CN
            predicted_class = random.choice(["AD", "CN"])
            if predicted_class == "AD":
                confidence = {"AD": 88.5, "MCI": 0, "CN": 11.5}
            else:
                confidence = {"AD": 4.2, "MCI": 0, "CN": 95.8}
        else:
            # multiclass: AD, MCI, or CN
            predicted_class = random.choice(["AD", "MCI", "CN"])
            if predicted_class == "AD":
                confidence = {"AD": 82.1, "MCI": 12.4, "CN": 5.5}
            elif predicted_class == "MCI":
                confidence = {"AD": 15.3, "MCI": 75.2, "CN": 9.5}
            else:
                confidence = {"AD": 2.1, "MCI": 10.4, "CN": 87.5}

        # 5. Clean up the temp file
        if os.path.exists(file_location):
            os.remove(file_location)

        # 6. Return standard JSON response
        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "filename": file.filename,
            "model_used": model_type,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
