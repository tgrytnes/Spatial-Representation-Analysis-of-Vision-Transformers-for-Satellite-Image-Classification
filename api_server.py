"""
FastAPI Inference Server for RunPod GPU deployment.

This server handles model inference requests from the NiceGUI frontend.
Designed to run on RunPod with GPU acceleration.
"""

import io

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from eurosat_vit_analysis.inference import (
    load_model_for_inference,
    predict_single_image,
)

# Initialize FastAPI app
app = FastAPI(
    title="Spatial ViT Inference API",
    description="GPU-accelerated inference API for satellite image classification",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
model_cache = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Response models
class ModelInfo(BaseModel):
    name: str
    display_name: str
    num_params: int
    architecture: str


class PredictionResponse(BaseModel):
    predictions: list[str]
    confidences: list[float]
    all_probabilities: list[float]
    inference_time_ms: float
    model_info: ModelInfo


class HealthResponse(BaseModel):
    status: str
    device: str
    gpu_name: str | None = None
    cuda_available: bool
    models_loaded: list[str]


def get_or_load_model(model_name: str, checkpoint_path: str | None = None):
    """Load model from cache or create new instance."""
    cache_key = f"{model_name}_{checkpoint_path or 'pretrained'}"

    if cache_key not in model_cache:
        print(f"Loading model: {model_name}")
        model, model_info = load_model_for_inference(
            model_name,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        model_cache[cache_key] = (model, model_info)
        print(f"Model {model_name} loaded successfully")

    return model_cache[cache_key]


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Spatial ViT Inference API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    return HealthResponse(
        status="healthy",
        device=str(device),
        gpu_name=gpu_name,
        cuda_available=torch.cuda.is_available(),
        models_loaded=list(model_cache.keys()),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form("vit_base"),
    top_k: int = Form(3),
    checkpoint_path: str | None = Form(None),
):
    """
    Run inference on uploaded image.

    Args:
        file: Image file (PNG, JPG, JPEG)
        model_name: Model architecture (vit_base, resnet50, swin_t)
        top_k: Number of top predictions to return (1-10)
        checkpoint_path: Optional path to trained checkpoint

    Returns:
        PredictionResponse with predictions, confidences, and timing
    """
    try:
        # Validate model name
        valid_models = ["vit_base", "resnet50", "swin_t", "convnext_t"]
        if model_name not in valid_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_name. Must be one of: {valid_models}",
            )

        # Validate top_k
        if not 1 <= top_k <= 10:
            raise HTTPException(
                status_code=400, detail="top_k must be between 1 and 10"
            )

        # Read and validate image
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

        # Load model
        model, model_info = get_or_load_model(model_name, checkpoint_path)

        # Run inference
        result = predict_single_image(
            model=model,
            image=image,
            model_info=model_info,
            device=device,
            top_k=top_k,
        )

        # Convert to response format
        return PredictionResponse(
            predictions=result.predictions,
            confidences=result.confidences,
            all_probabilities=result.all_probabilities,
            inference_time_ms=result.inference_time_ms,
            model_info=ModelInfo(
                name=result.model_info.name,
                display_name=result.model_info.display_name,
                num_params=result.model_info.num_params,
                architecture=result.model_info.architecture,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/models", response_model=dict)
async def list_models():
    """List available models and their status."""
    return {
        "available_models": {
            "vit_base": {
                "display_name": "ViT-Base",
                "architecture": "Vision Transformer",
                "loaded": "vit_base_pretrained" in model_cache
                or "vit_base_None" in model_cache,
            },
            "resnet50": {
                "display_name": "ResNet-50",
                "architecture": "Residual CNN",
                "loaded": "resnet50_pretrained" in model_cache
                or "resnet50_None" in model_cache,
            },
            "swin_t": {
                "display_name": "Swin-Tiny",
                "architecture": "Swin Transformer",
                "loaded": "swin_t_pretrained" in model_cache
                or "swin_t_None" in model_cache,
            },
        },
        "loaded_models": list(model_cache.keys()),
    }


@app.post("/preload")
async def preload_models(model_names: list[str]):
    """
    Preload models into memory for faster inference.

    Args:
        model_names: List of model names to preload
    """
    loaded = []
    errors = []

    for model_name in model_names:
        try:
            get_or_load_model(model_name)
            loaded.append(model_name)
        except Exception as e:
            errors.append({"model": model_name, "error": str(e)})

    return {"loaded": loaded, "errors": errors, "total_cached": len(model_cache)}


if __name__ == "__main__":
    # Print startup info
    print("=" * 60)
    print("Spatial ViT Inference API Server")
    print("=" * 60)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("WARNING: No GPU detected, running on CPU")
    print("=" * 60)

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
