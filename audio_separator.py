#!/usr/bin/env python3
"""
Audio separation module for separating voice and background music
"""

import os
import subprocess
from pathlib import Path
from typing import Tuple, Optional
import logging


class AudioSeparator:
    """Handles separation of voice and background music from audio"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories"""
        directories = ["separated_audio", "temp_audio_separation"]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def separate_audio_spleeter(self, audio_path: str, job_id: str) -> Tuple[str, str]:
        """
        Separate audio using Spleeter (requires installation)
        Returns: (voice_path, music_path)
        """
        try:
            output_dir = os.path.join("separated_audio", job_id)
            os.makedirs(output_dir, exist_ok=True)
            
            # Use Spleeter to separate vocals and accompaniment
            cmd = [
                'spleeter', 'separate',
                '-p', 'spleeter:2stems-16kHz',
                '-o', output_dir,
                audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                audio_name = Path(audio_path).stem
                voice_path = os.path.join(output_dir, audio_name, "vocals.wav")
                music_path = os.path.join(output_dir, audio_name, "accompaniment.wav")
                
                if os.path.exists(voice_path) and os.path.exists(music_path):
                    self.logger.info(f"Successfully separated audio using Spleeter")
                    return voice_path, music_path
                else:
                    raise Exception("Spleeter output files not found")
            else:
                raise Exception(f"Spleeter failed: {result.stderr}")
                
        except Exception as e:
            self.logger.warning(f"Spleeter separation failed: {e}")
            return self._fallback_separation(audio_path, job_id)
    
    def separate_audio_ffmpeg(self, audio_path: str, job_id: str) -> Tuple[str, str]:
        """
        Basic audio separation using FFmpeg filters
        Returns: (voice_path, music_path)
        """
        try:
            voice_path = os.path.join("separated_audio", f"voice_{job_id}.wav")
            music_path = os.path.join("separated_audio", f"music_{job_id}.wav")
            
            # Extract vocal-focused audio (center channel isolation)
            voice_cmd = [
                'ffmpeg', '-i', audio_path,
                '-af', 'pan=mono|c0=0.5*c0+-0.5*c1',  # Center channel extraction
                '-ar', '22050',
                '-ac', '1',
                voice_path, '-y'
            ]
            
            # Extract music-focused audio (side channels)
            music_cmd = [
                'ffmpeg', '-i', audio_path,
                '-af', 'pan=stereo|c0=c0|c1=c1,lowpass=8000',  # Keep stereo, filter high frequencies
                '-ar', '44100',
                '-ac', '2',
                music_path, '-y'
            ]
            
            # Run voice extraction
            subprocess.run(voice_cmd, capture_output=True, text=True, check=True, timeout=120)
            
            # Run music extraction  
            subprocess.run(music_cmd, capture_output=True, text=True, check=True, timeout=120)
            
            if os.path.exists(voice_path) and os.path.exists(music_path):
                self.logger.info("Audio separated using FFmpeg filters")
                return voice_path, music_path
            else:
                raise Exception("FFmpeg separation failed to create output files")
                
        except Exception as e:
            self.logger.error(f"FFmpeg separation failed: {e}")
            return self._fallback_separation(audio_path, job_id)
    
    def _fallback_separation(self, audio_path: str, job_id: str) -> Tuple[str, str]:
        """Fallback: duplicate original audio for both voice and music"""
        self.logger.warning("Using fallback: duplicating original audio")
        
        voice_path = os.path.join("separated_audio", f"voice_{job_id}.wav") 
        music_path = os.path.join("separated_audio", f"music_{job_id}.wav")
        
        # Copy original audio as both voice and music
        import shutil
        shutil.copy2(audio_path, voice_path)
        shutil.copy2(audio_path, music_path)
        
        return voice_path, music_path
    
    def separate_audio(self, audio_path: str, job_id: str, method: str = "auto") -> Tuple[str, str]:
        """
        Main audio separation method
        
        Args:
            audio_path: Path to input audio file
            job_id: Unique job identifier
            method: "spleeter", "ffmpeg", or "auto"
            
        Returns:
            Tuple of (voice_path, music_path)
        """
        if method == "spleeter":
            return self.separate_audio_spleeter(audio_path, job_id)
        elif method == "ffmpeg":
            return self.separate_audio_ffmpeg(audio_path, job_id)
        else:  # auto
            # Try Spleeter first, fallback to FFmpeg
            try:
                return self.separate_audio_spleeter(audio_path, job_id)
            except:
                return self.separate_audio_ffmpeg(audio_path, job_id)
    
    def combine_voice_music(self, voice_path: str, music_path: str, output_path: str, 
                           voice_volume: float = 1.0, music_volume: float = 0.3) -> bool:
        """
        Combine translated voice with background music
        
        Args:
            voice_path: Path to translated voice audio
            music_path: Path to background music
            output_path: Path for final combined audio
            voice_volume: Volume multiplier for voice (1.0 = original)
            music_volume: Volume multiplier for music (0.3 = 30% of original)
            
        Returns:
            True if successful
        """
        try:
            cmd = [
                'ffmpeg',
                '-i', voice_path,
                '-i', music_path, 
                '-filter_complex', 
                f'[0:a]volume={voice_volume}[voice];[1:a]volume={music_volume}[music];[voice][music]amix=inputs=2:duration=first:dropout_transition=2',
                '-ar', '44100',
                '-ac', '2',
                output_path, '-y'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=180)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                self.logger.info(f"Successfully combined voice and music: {output_path}")
                return True
            else:
                raise Exception("Combined audio file is empty or missing")
                
        except Exception as e:
            self.logger.error(f"Failed to combine voice and music: {e}")
            return False