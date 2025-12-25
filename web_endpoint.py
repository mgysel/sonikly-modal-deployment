"""
Web API endpoint for Stable Audio Open

This script creates a FastAPI web endpoint for the model,
allowing you to generate audio via HTTP requests.
"""

import modal
import io
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Import the image and model from the main deployment
from stable_audio_modal import app, image, StableAudioModel

# Create FastAPI app
web_app = FastAPI(
    title="Stable Audio Open API",
    description="Generate audio from text prompts using Stability AI's Stable Audio Open model",
    version="1.0.0",
)


class GenerationRequest(BaseModel):
    """Request model for audio generation"""
    prompt: str = Field(..., description="Text description of the audio to generate", min_length=1)
    negative_prompt: str = Field("", description="Text description of what to avoid")
    duration: float = Field(10.0, description="Duration in seconds (max 47)", ge=1.0, le=47.0)
    steps: int = Field(100, description="Number of diffusion steps", ge=10, le=200)
    cfg_scale: float = Field(7.0, description="Classifier-free guidance scale", ge=1.0, le=20.0)
    seed: int = Field(-1, description="Random seed (-1 for random)")


class GenerationResponse(BaseModel):
    """Response model for audio generation"""
    message: str
    duration: float
    steps: int
    cfg_scale: float
    seed: int


@web_app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Stable Audio Open API",
        "version": "1.0.0",
        "endpoints": {
            "generate": "/generate",
            "health": "/health",
            "docs": "/docs",
        },
        "model": "stabilityai/stable-audio-open-1.0",
    }


@web_app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "model": "stable-audio-open-1.0"}


@web_app.post("/generate", response_class=Response)
async def generate_audio(request: GenerationRequest):
    """
    Generate audio from a text prompt
    
    Returns the audio file as a WAV stream
    """
    try:
        # Get the model instance
        model = StableAudioModel()
        
        # Generate audio
        audio_bytes = model.generate.remote(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            duration=request.duration,
            steps=request.steps,
            cfg_scale=request.cfg_scale,
            seed=request.seed,
        )
        
        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'attachment; filename="generated_audio.wav"',
                "X-Prompt": request.prompt,
                "X-Duration": str(request.duration),
                "X-Steps": str(request.steps),
                "X-CFG-Scale": str(request.cfg_scale),
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@web_app.post("/generate/info", response_model=GenerationResponse)
async def generate_audio_info(request: GenerationRequest):
    """
    Generate audio and return metadata (without the audio file)
    
    Useful for testing parameters before actual generation
    """
    return GenerationResponse(
        message="Audio generation parameters validated",
        duration=request.duration,
        steps=request.steps,
        cfg_scale=request.cfg_scale,
        seed=request.seed if request.seed != -1 else 0,
    )


# Mount the FastAPI app to Modal
@app.function(
    image=image,
    keep_warm=1,  # Keep one container warm for faster response
)
@modal.asgi_app()
def fastapi_app():
    """Serve the FastAPI app"""
    return web_app


if __name__ == "__main__":
    print("To deploy this web endpoint, run:")
    print("  modal deploy web_endpoint.py")
    print("\nAfter deployment, you can use it like:")
    print("  curl -X POST https://your-app-url/generate \\")
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"prompt": "A peaceful piano melody", "duration": 10}\' \\')
    print("    --output audio.wav")

