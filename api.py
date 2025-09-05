#!/usr/bin/env python3
"""
FastAPI Video Translation Service

Asynchronously processes Hindi video translation to Marathi
and returns the output video path.
"""

import asyncio
import os
import uuid
import shutil
from datetime import datetime
from typing import Dict, Optional
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, validator
import uvicorn

from video_translator import translate_video
from audio_separator import AudioSeparator


# Initialize FastAPI app
app = FastAPI(
    title="Video Translation API",
    description="Translate Hindi videos to Marathi with text and audio translation",
    version="1.0.0"
)

# In-memory storage for job status (in production, use Redis or database)
job_status: Dict[str, Dict] = {}

def ensure_api_directories():
    """Ensure all necessary directories exist for the API"""
    directories = [
        "frames",
        "translated_frames",
        "output",
        "temp",
        "logs",
        "bbox",
        "audios",
        "translated_audio"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"📁 Ensured directory exists: {directory}/")
        except Exception as e:
            print(f"❌ Error creating directory {directory}: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize directories and resources on startup"""
    print("🚀 Starting Video Translation API...")
    ensure_api_directories()
    print("✅ API initialization complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("🔄 Shutting down Video Translation API...")
    print("✅ Shutdown complete")

# Request/Response models
class VideoTranslationRequest(BaseModel):
    video_path: str
    font_path: Optional[str] = "noto-sans-devanagari/NotoSansDevanagari-Regular.ttf"
    cleanup: Optional[bool] = True
    env_file: Optional[str] = ".env"
    job_id: Optional[str] = None  # Optional custom job ID for consistent naming
    moving_text: Optional[bool] = False  # Whether video contains moving text
    advanced_ocr: Optional[bool] = True  # Whether to use advanced OCR with preprocessing
    
    @validator('video_path')
    def validate_video_path(cls, v):
        if not v or not v.strip():
            raise ValueError("video_path cannot be empty")
        return v.strip()

class VideoTranslationResponse(BaseModel):
    job_id: str
    status: str
    message: str
    video_path: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: Optional[str] = None
    video_path: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


def update_job_status(job_id: str, status: str, **kwargs):
    """Update job status in memory"""
    if job_id in job_status:
        job_status[job_id].update({
            'status': status,
            **kwargs
        })


def ensure_job_directories(job_id: str):
    """Ensure job-specific directories exist"""
    job_dirs = [
        f"frames/{job_id}",
        f"translated_frames/{job_id}",
        f"temp/{job_id}",
        f"logs/{job_id}"
    ]
    
    for directory in job_dirs:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            print(f"❌ Error creating job directory {directory}: {e}")
            return False
    return True


async def process_video_async(job_id: str, request: VideoTranslationRequest):
    """
    Asynchronously process video translation
    """
    try:
        # Ensure job directories exist
        if not ensure_job_directories(job_id):
            raise Exception("Failed to create job directories")
        
        # Update status to processing
        update_job_status(job_id, "processing", progress="Starting video translation...")
        
        # Run the translate_video function in a thread pool
        # Since translate_video is CPU-intensive, we run it in an executor
        loop = asyncio.get_event_loop()
        
        # Update progress
        update_job_status(job_id, "processing", progress="Extracting frames...")
        
        # Execute the translation in thread pool to avoid blocking
        output_video_path = await loop.run_in_executor(
            None,  # Use default thread pool
            lambda: translate_video(
                video_path=request.video_path,
                font_path=request.font_path,
                cleanup=request.cleanup,
                env_file=request.env_file,
                video_id=job_id,  # Use job_id as video_id for consistent naming
                moving_text=request.moving_text,  # Pass moving text flag
                advanced_ocr=request.advanced_ocr  # Pass advanced OCR flag
            )
        )
        
        # Move output video to output directory if it's not already there
        output_dir = "output"
        if not output_video_path.startswith(output_dir):
            final_output_path = os.path.join(output_dir, os.path.basename(output_video_path))
            try:
                shutil.move(output_video_path, final_output_path)
                output_video_path = final_output_path
            except Exception as e:
                print(f"Warning: Could not move output video to output directory: {e}")
        
        # Update status to completed
        update_job_status(
            job_id, 
            "completed", 
            video_path=output_video_path,
            completed_at=datetime.utcnow(),
            progress="Translation completed successfully!"
        )
        
        print(f"✅ Job {job_id} completed successfully: {output_video_path}")
        
    except FileNotFoundError as e:
        error_msg = f"File not found: {str(e)}"
        update_job_status(
            job_id, 
            "failed", 
            error=error_msg,
            completed_at=datetime.utcnow()
        )
        print(f"❌ Job {job_id} failed: {error_msg}")
        
    except ValueError as e:
        error_msg = f"Configuration error: {str(e)}"
        update_job_status(
            job_id, 
            "failed", 
            error=error_msg,
            completed_at=datetime.utcnow()
        )
        print(f"❌ Job {job_id} failed: {error_msg}")
        
    except Exception as e:
        error_msg = f"Translation failed: {str(e)}"
        traceback.print_exc()  # Log full traceback
        update_job_status(
            job_id, 
            "failed", 
            error=error_msg,
            completed_at=datetime.utcnow()
        )
        print(f"❌ Job {job_id} failed: {error_msg}")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Video Translation API is running",
        "version": "1.0.0",
        "status": "healthy"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    # Check if .env file exists
    env_exists = os.path.exists(".env")
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "env_file_exists": env_exists,
        "active_jobs": len([j for j in job_status.values() if j['status'] == 'processing'])
    }


@app.post("/translate", response_model=VideoTranslationResponse)
async def translate_video_async(
    request: VideoTranslationRequest, 
    background_tasks: BackgroundTasks
):
    """
    Start asynchronous video translation
    
    Returns immediately with a job_id for tracking progress.
    """
    # Validate video file exists
    if not os.path.exists(request.video_path):
        raise HTTPException(
            status_code=404,
            detail=f"Video file not found: {request.video_path}"
        )
    
    # Validate .env file exists
    if not os.path.exists(request.env_file):
        raise HTTPException(
            status_code=400,
            detail=f"Environment file not found: {request.env_file}. Please create .env file with your credentials."
        )
    
    # Use provided job ID or generate unique job ID
    job_id = request.job_id if request.job_id else str(uuid.uuid4())
    
    # Validate job_id is not already in use
    if job_id in job_status:
        raise HTTPException(
            status_code=409,
            detail=f"Job ID {job_id} is already in use. Please use a different job_id or omit it to auto-generate."
        )
    
    # Initialize job status
    start_time = datetime.utcnow()
    job_status[job_id] = {
        'job_id': job_id,
        'status': 'queued',
        'progress': 'Job queued for processing',
        'video_path': None,
        'started_at': start_time,
        'completed_at': None,
        'error': None,
        'request': request.dict()
    }
    
    # Start background processing
    background_tasks.add_task(process_video_async, job_id, request)
    
    return VideoTranslationResponse(
        job_id=job_id,
        status="queued",
        message="Video translation job started. Use /status/{job_id} to track progress.",
        started_at=start_time
    )


@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the status of a video translation job
    """
    if job_id not in job_status:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    job_data = job_status[job_id]
    
    return JobStatusResponse(
        job_id=job_id,
        status=job_data['status'],
        progress=job_data.get('progress'),
        video_path=job_data.get('video_path'),
        started_at=job_data['started_at'],
        completed_at=job_data.get('completed_at'),
        error=job_data.get('error')
    )


@app.get("/download/{job_id}")
async def download_video(job_id: str):
    """
    Download the translated video file
    """
    if job_id not in job_status:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    job_data = job_status[job_id]
    
    if job_data['status'] != 'completed':
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not completed. Status: {job_data['status']}"
        )
    
    video_path = job_data.get('video_path')
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(
            status_code=404,
            detail=f"Video file not found for job {job_id}"
        )
    
    return FileResponse(
        path=video_path,
        filename=f"translated_{job_id}.mp4",
        media_type="video/mp4"
    )


@app.get("/jobs")
async def list_jobs():
    """
    list all jobs and their statuses
    """
    return {
        "total_jobs": len(job_status),
        "jobs": [
            {
                "job_id": job_id,
                "status": data['status'],
                "started_at": data['started_at'],
                "completed_at": data.get('completed_at'),
                "video_path": data.get('video_path')
            }
            for job_id, data in job_status.items()
        ]
    }


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job and its associated files
    """
    if job_id not in job_status:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    job_data = job_status[job_id]
    
    # Delete video file if it exists
    video_path = job_data.get('video_path')
    if video_path and os.path.exists(video_path):
        try:
            os.remove(video_path)
        except Exception as e:
            print(f"Warning: Could not delete video file {video_path}: {e}")
    
    # Remove from job status
    del job_status[job_id]
    
    return {"message": f"Job {job_id} deleted successfully"}


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "type": type(exc).__name__
        }
    )


if __name__ == "__main__":
    # Check for .env file on startup
    if not os.path.exists(".env"):
        print("❌ Warning: .env file not found!")
        print("Please create .env file with your credentials:")
        print("  cp .env.example .env")
        print("  # Edit .env with your actual API keys")
        print()
    
    # Run the API server
    print("🚀 Starting Video Translation API Server...")
    print("📖 API Documentation: http://localhost:8001/docs")
    print("📊 Health Check: http://localhost:8001/health")
    print()
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )



class AudioSeparationResponse(BaseModel):
    job_id: str
    status: str
    voice_audio_path: Optional[str] = None
    music_audio_path: Optional[str] = None
    original_transcription: Optional[str] = None
    marathi_translation: Optional[str] = None

class TextTranslationData(BaseModel):
    hindi_text: str
    marathi_translation: str
    bbox: list[int]
    
class EditableTranslationsResponse(BaseModel):
    job_id: str
    audio_translations: dict
    text_translations: list[TextTranslationData]

class UpdateTranslationsRequest(BaseModel):
    job_id: str
    audio_translation: Optional[str] = None
    text_translations: Optional[list[TextTranslationData]] = None

# Add these new endpoints after existing ones

@app.post("/separate-audio/{job_id}")
async def separate_audio_endpoint(job_id: str, background_tasks: BackgroundTasks):
    """
    Stage 1a: Separate background music from voice in uploaded video
    """
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job_data = job_status[job_id]
    
    if job_data['status'] != 'completed':
        raise HTTPException(status_code=400, detail=f"Job {job_id} is not completed yet")
    
    try:
        # Initialize audio separator
        separator = AudioSeparator()
        
        # Get original audio path
        video_path = job_data['request']['video_path']
        audio_path = f"audios/audio_{job_id}.wav"
        
        # Separate audio
        voice_path, music_path = separator.separate_audio(audio_path, job_id)
        
        # Update job status with separation results
        job_status[job_id].update({
            'voice_audio_path': voice_path,
            'music_audio_path': music_path,
            'separation_completed': True
        })
        
        return AudioSeparationResponse(
            job_id=job_id,
            status="separated",
            voice_audio_path=voice_path,
            music_audio_path=music_path
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio separation failed: {str(e)}")


@app.get("/editable-translations/{job_id}", response_model=EditableTranslationsResponse)
async def get_editable_translations(job_id: str):
    """
    Stage 1c & 1d: Get editable translations for both audio and text
    """
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job_data = job_status[job_id]
    
    if job_data['status'] != 'completed':
        raise HTTPException(status_code=400, detail=f"Job {job_id} not completed yet")
    
    try:
        # Get audio transcription and translation (if available)
        audio_translations = {
            "hindi_transcription": job_data.get('hindi_transcription', ''),
            "marathi_translation": job_data.get('marathi_audio_translation', '')
        }
        
        # Get text translations from processing results
        text_translations = []
        if 'text_buffer' in job_data:
            for item in job_data['text_buffer']:
                if item.get('marathi_translation'):
                    text_translations.append(TextTranslationData(
                        hindi_text=item['all_text'],
                        marathi_translation=item['marathi_translation'],
                        bbox=list(item['bbox'])
                    ))
        
        return EditableTranslationsResponse(
            job_id=job_id,
            audio_translations=audio_translations,
            text_translations=text_translations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get translations: {str(e)}")


@app.post("/update-translations")
async def update_translations(request: UpdateTranslationsRequest):
    """
    Update editable translations before final video generation
    """
    job_id = request.job_id
    
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    try:
        job_data = job_status[job_id]
        
        # Update audio translation if provided
        if request.audio_translation:
            job_data['edited_marathi_audio'] = request.audio_translation
        
        # Update text translations if provided
        if request.text_translations:
            # Create lookup map for updates
            update_map = {}
            for item in request.text_translations:
                update_map[item.hindi_text] = item.marathi_translation
            
            # Update text buffer with edited translations
            if 'text_buffer' in job_data:
                for buffer_item in job_data['text_buffer']:
                    hindi_text = buffer_item['all_text']
                    if hindi_text in update_map:
                        buffer_item['marathi_translation'] = update_map[hindi_text]
        
        # Mark as ready for final processing
        job_data['translations_edited'] = True
        
        return {"success": True, "message": "Translations updated successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update translations: {str(e)}")


@app.post("/generate-final-video/{job_id}")
async def generate_final_video(job_id: str, background_tasks: BackgroundTasks):
    """
    Stage 2: Generate final video with edited translations
    """
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job_data = job_status[job_id]
    
    if not job_data.get('translations_edited', False):
        raise HTTPException(status_code=400, detail="Translations must be edited first")
    
    # Start final video generation in background
    background_tasks.add_task(generate_final_video_task, job_id)
    
    return {"job_id": job_id, "status": "generating_final", "message": "Final video generation started"}


async def generate_final_video_task(job_id: str):
    """Background task for final video generation"""
    try:
        update_job_status(job_id, "generating_final", progress="Starting final video generation...")
        
        job_data = job_status[job_id]
        
        # Use edited audio translation if available
        marathi_text = job_data.get('edited_marathi_audio') or job_data.get('marathi_audio_translation', '')
        
        # Generate new Marathi audio with edited text
        if marathi_text:
            from video_translator import VideoTranslator
            
            # Initialize translator components
            openai_api_key = os.getenv('OPENAI_API_KEY')
            google_credentials_path = os.getenv('GOOGLE_CREDENTIALS_PATH')
            
            translator = VideoTranslator(openai_api_key, google_credentials_path)
            
            # Generate new Marathi audio
            new_audio_path = f"translated_audio/final_marathi_{job_id}.mp3"
            translator.generate_marathi_audio(marathi_text, new_audio_path)
            
            # Combine with background music if separated
            if job_data.get('music_audio_path'):
                separator = AudioSeparator()
                final_audio_path = f"translated_audio/combined_{job_id}.mp3"
                
                separator.combine_voice_music(
                    new_audio_path, 
                    job_data['music_audio_path'], 
                    final_audio_path
                )
                audio_for_video = final_audio_path
            else:
                audio_for_video = new_audio_path
            
            # Create final video with edited translations
            frames_dir = f"translated_frames/{job_id}"
            total_frames = len([f for f in os.listdir(frames_dir) if f.endswith('.png')])
            frames_pattern = os.path.join(frames_dir, "frame_%04d.png")
            final_output = f"output/final_{job_id}.mp4"
            
            if translator.create_final_video(frames_pattern, audio_for_video, final_output, total_frames):
                update_job_status(
                    job_id, 
                    "final_completed",
                    video_path=final_output,
                    completed_at=datetime.utcnow(),
                    progress="Final video generation completed!"
                )
            else:
                raise Exception("Failed to create final video")
        
    except Exception as e:
        update_job_status(
            job_id,
            "failed", 
            error=f"Final video generation failed: {str(e)}",
            completed_at=datetime.utcnow()
        )