#!/usr/bin/env python3
"""
Video Translation Pipeline
Converts Hindi video content to Marathi with translated text overlays and audio
"""

import os
import uuid
import subprocess
import json
import logging
from glob import glob
from typing import Dict, List, Optional, Tuple
import shutil
from pathlib import Path
from datetime import datetime

# Core libraries
import easyocr
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm
from mutagen.mp3 import MP3
import cv2
from audio_separator import AudioSeparator


# API clients
from openai import OpenAI
from google.oauth2 import service_account
from google.cloud import texttospeech


class VideoTranslator:
    """Main video translation class"""
    
    def __init__(
        self,
        openai_api_key: str,
        google_credentials_path: str,
        font_path: str = "noto-sans-devanagari/NotoSansDevanagari-Regular.ttf"
    ):
        """
        Initialize the video translator
        
        Args:
            openai_api_key: OpenAI API key for translation
            google_credentials_path: Path to Google Cloud service account JSON
            font_path: Path to font file for text overlay
        """
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.font_path = font_path
        
        # Initialize Google TTS
        with open(google_credentials_path, 'r') as f:
            creds = json.load(f)
        credentials = service_account.Credentials.from_service_account_info(creds)
        self.tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
        
        # Initialize EasyOCR with GPU support for better performance
        try:
            self.ocr_reader = easyocr.Reader(['hi','mr'], gpu=True)
            print("📱 EasyOCR initialized with GPU acceleration")
        except Exception as e:
            print(f"⚠️ GPU not available for EasyOCR, falling back to CPU: {e}")
            self.ocr_reader = easyocr.Reader(['hi'], gpu=False)
        
        # Global tracker for consistent text display
        self.bbox_word_count_tracker = {}
        
        # OCR preprocessing scale factor - used for both image resize and coordinate scaling
        self.ocr_scale_factor = 3
        
        # Logger will be set up per job
        self.logger = None
        
        # Check FFmpeg capabilities for optimization
        self.ffmpeg_capabilities = self._check_ffmpeg_capabilities()
        
        # Ensure base directories exist
        self._ensure_base_directories()
    
    def _ensure_base_directories(self):
        """Create base directories if they don't exist"""
        base_dirs = [
            "frames",
            "translated_frames", 
            "output",
            "temp",
            "bbox",
            "logs",
            "audios",
            "translated_audio"
        ]
        
        for directory in base_dirs:
            os.makedirs(directory, exist_ok=True)
            print(f"📁 Ensured directory exists: {directory}/")
    
    def setup_logger(self, video_id: str) -> logging.Logger:
        """
        Set up job-specific logger that writes to both console and file
        
        Args:
            video_id: Job/video UUID for creating specific log file
            
        Returns:
            Configured logger instance
        """
        # Create logs directory structure
        log_dir = os.path.join("logs", video_id)
        os.makedirs(log_dir, exist_ok=True)
        
        # Create logger
        logger_name = f"video_translator_{video_id}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter('%(message)s')
        
        # File handler for detailed logs
        log_file = os.path.join(log_dir, f"translation_{video_id}.log")
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler for simple output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Log start of processing
        logger.info(f"🎬 Started video translation for job: {video_id}")
        logger.info(f"📁 Log file: {log_file}")
        logger.info(f"⏰ Start time: {datetime.now().isoformat()}")
        
        # Log FFmpeg capabilities
        if self.ffmpeg_capabilities.get('available'):
            logger.info(f"⚡ FFmpeg available: {self.ffmpeg_capabilities.get('version', 'Unknown version')}")
            hwaccel = self.ffmpeg_capabilities.get('hwaccel', [])
            if hwaccel:
                logger.info(f"🖥️  Hardware acceleration: {', '.join(hwaccel)}")
            encoders = self.ffmpeg_capabilities.get('encoders', [])
            if any('nvenc' in enc.lower() for enc in encoders):
                logger.info("🚀 NVIDIA hardware encoding available")
        else:
            logger.warning("⚠️  FFmpeg not detected - performance may be limited")
        
        logger.info("")
        
        self.logger = logger
        return logger
    
    def _check_ffmpeg_capabilities(self) -> dict:
        """Check FFmpeg capabilities for optimization"""
        capabilities = {
            'available': False,
            'hwaccel': [],
            'encoders': [],
            'version': None
        }
        
        try:
            # Check if FFmpeg is available
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                capabilities['available'] = True
                capabilities['version'] = result.stdout.split('\n')[0]
                
                # Check hardware acceleration support
                hwaccel_result = subprocess.run(['ffmpeg', '-hwaccels'], 
                                              capture_output=True, text=True, timeout=10)
                if hwaccel_result.returncode == 0:
                    hwaccels = hwaccel_result.stdout.strip().split('\n')[1:]  # Skip header
                    capabilities['hwaccel'] = [hw.strip() for hw in hwaccels if hw.strip()]
                
                # Check available encoders
                encoder_result = subprocess.run(['ffmpeg', '-encoders'], 
                                              capture_output=True, text=True, timeout=10)
                if encoder_result.returncode == 0:
                    lines = encoder_result.stdout.split('\n')
                    encoders = []
                    for line in lines:
                        if 'h264' in line.lower() or 'nvenc' in line.lower():
                            encoders.append(line.strip())
                    capabilities['encoders'] = encoders
                    
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            pass  # FFmpeg not available or error occurred
            
        return capabilities
    
    def preprocess_image_for_ocr(self, image_path: str) -> str:
        """
        Apply image preprocessing for better OCR accuracy
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Path to the preprocessed image (temporary file)
        """
        import tempfile
        
        # Load image
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Invert colors (since text is typically bright on dark backgrounds)
        inv = cv2.bitwise_not(gray)
        
        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl1 = clahe.apply(inv)
        
        # Sharpening kernel
        kernel_sharpening = np.array([[-1,-1,-1], 
                                      [-1, 9,-1],
                                      [-1,-1,-1]])
        sharp = cv2.filter2D(cl1, -1, kernel_sharpening)
        
        # Morphological closing (dilation + erosion) with small kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        closing = cv2.morphologyEx(sharp, cv2.MORPH_CLOSE, kernel)
        
        # Resize image for better OCR detail using class scale factor
        resized = cv2.resize(closing, None, fx=self.ocr_scale_factor, fy=self.ocr_scale_factor, interpolation=cv2.INTER_LINEAR)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_path = temp_file.name
        temp_file.close()
        
        cv2.imwrite(temp_path, resized)
        
        if self.logger:
            self.logger.debug(f"🔍 Preprocessed image for OCR: {image_path} -> {temp_path}")
        
        return temp_path
    
    def _ensure_directory_exists(self, directory_path: str):
        """Ensure a specific directory exists, create if not"""
        try:
            os.makedirs(directory_path, exist_ok=True)
            return True
        except Exception as e:
            print(f"❌ Error creating directory {directory_path}: {e}")
            return False
        
    def extract_video_frames(self, video_path: str, output_dir: str) -> bool:
        """Extract frames from video using optimized FFmpeg with hardware acceleration"""
        try:
            # Ensure output directory exists
            if not self._ensure_directory_exists(output_dir):
                return False
            
            if self.logger:
                self.logger.info(f"⚡ Extracting frames with FFmpeg optimization...")
            
            cmd = [
                'ffmpeg',
                '-threads', '0',  # Use all available CPU threads
                '-i', video_path,
                '-vf', 'fps=30',  # Extract at 30 FPS for consistent processing
                '-q:v', '1',  # High quality PNG output (1-31, lower is better)
                '-pix_fmt', 'rgb24',  # Consistent pixel format for PIL compatibility
                f'{output_dir}/frame_%04d.png',
                '-y',  # Overwrite existing files
                '-hide_banner',  # Cleaner output
                '-loglevel', 'warning'  # Reduce verbosity
            ]
            
            # Try hardware acceleration first, fallback to software
            hw_cmd = [
                'ffmpeg',
                '-threads', '0',
                '-hwaccel', 'auto',  # Auto-detect hardware acceleration
                '-i', video_path,
                '-vf', 'fps=30',
                '-q:v', '1',
                '-pix_fmt', 'rgb24',
                f'{output_dir}/frame_%04d.png',
                '-y',
                '-hide_banner',
                '-loglevel', 'warning'
            ]
            
            try:
                # Try hardware acceleration first
                result = subprocess.run(hw_cmd, capture_output=True, text=True, check=True, timeout=300)
                if self.logger:
                    self.logger.info("✅ Frames extracted with hardware acceleration")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                # Fallback to software processing
                if self.logger:
                    self.logger.info("🔄 Hardware acceleration failed, using software processing...")
                result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
                if self.logger:
                    self.logger.info("✅ Frames extracted with software processing")
            
            # Count extracted frames
            frame_count = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
            print(f"Successfully extracted {frame_count} frames to {output_dir}")
            
            if self.logger:
                self.logger.debug(f"📊 Frame extraction stats: {frame_count} frames extracted")
            
            return True
            
        except subprocess.TimeoutExpired:
            error_msg = "Frame extraction timed out (>5 minutes)"
            print(f"Error extracting frames: {error_msg}")
            if self.logger:
                self.logger.error(error_msg)
            return False
        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg error: {e.stderr}"
            print(f"Error extracting frames: {error_msg}")
            if self.logger:
                self.logger.error(error_msg)
            return False
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"Error extracting frames: {error_msg}")
            if self.logger:
                self.logger.error(error_msg)
            return False
    
    def extract_audio(self, video_path: str, output_audio: str) -> bool:
        """Extract audio from video using optimized FFmpeg"""
        try:
            if self.logger:
                self.logger.info(f"🎵 Extracting audio with FFmpeg optimization...")
            
            # Determine output format and codec based on file extension
            audio_ext = os.path.splitext(output_audio)[1].lower()
            
            if audio_ext == '.wav':
                # WAV format - uncompressed, best for processing
                cmd = [
                    'ffmpeg',
                    '-threads', '0',  # Use all available CPU threads
                    '-i', video_path,
                    '-vn',  # No video
                    '-acodec', 'pcm_s16le',  # 16-bit PCM for compatibility
                    '-ar', '44100',  # Standard sample rate
                    '-ac', '1',  # Mono for speech recognition (smaller, faster)
                    '-af', 'volume=0.8',  # Slightly reduce volume to prevent clipping
                    output_audio,
                    '-y',  # Overwrite existing files
                    '-hide_banner',
                    '-loglevel', 'warning'
                ]
            elif audio_ext == '.mp3':
                # MP3 format - compressed, good for final output
                cmd = [
                    'ffmpeg',
                    '-threads', '0',
                    '-i', video_path,
                    '-vn',
                    '-acodec', 'mp3',
                    '-b:a', '128k',  # Good quality for speech
                    '-ar', '44100',
                    '-ac', '2',  # Stereo for final output
                    output_audio,
                    '-y',
                    '-hide_banner',
                    '-loglevel', 'warning'
                ]
            else:
                # Default format - let FFmpeg decide
                cmd = [
                    'ffmpeg',
                    '-threads', '0',
                    '-i', video_path,
                    '-vn',
                    '-q:a', '0',  # Best quality
                    '-map', 'a',  # Map audio stream
                    output_audio,
                    '-y',
                    '-hide_banner',
                    '-loglevel', 'warning'
                ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120)
            
            # Verify output file exists and has content
            if os.path.exists(output_audio) and os.path.getsize(output_audio) > 0:
                file_size = os.path.getsize(output_audio) / (1024 * 1024)  # MB
                print(f"Successfully extracted audio to {output_audio} ({file_size:.1f} MB)")
                
                if self.logger:
                    self.logger.debug(f"📊 Audio extraction stats: {file_size:.1f} MB, format: {audio_ext}")
                
                return True
            else:
                raise Exception("Output audio file is empty or missing")
            
        except subprocess.TimeoutExpired:
            error_msg = "Audio extraction timed out (>2 minutes)"
            print(f"Error extracting audio: {error_msg}")
            if self.logger:
                self.logger.error(error_msg)
            return False
        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg error: {e.stderr}"
            print(f"Error extracting audio: {error_msg}")
            if self.logger:
                self.logger.error(error_msg)
            return False
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"Error extracting audio: {error_msg}")
            if self.logger:
                self.logger.error(error_msg)
            return False
    
    def is_hindi(self, text: str) -> bool:
        """Check if text contains significant Devanagari characters"""
        if not text:
            return False
        devanagari_chars = sum(1 for char in text if 0x0900 <= ord(char) <= 0x097F)
        return devanagari_chars / len(text) > 0.5
    
    def regions_overlap(self, bbox1: Tuple, bbox2: Tuple, threshold: int = 20) -> bool:
        """
        Check if two bounding boxes overlap significantly
        Uses larger threshold to better handle moving text scenarios
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate overlap area vs total area for more intelligent overlap detection
        overlap_x = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
        overlap_y = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
        overlap_area = overlap_x * overlap_y
        
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        min_area = min(area1, area2)
        
        # Consider boxes overlapping if they share >50% of the smaller box's area
        if min_area > 0:
            overlap_ratio = overlap_area / min_area
            return overlap_ratio > 0.5
        
        # Fallback to distance-based check with threshold
        return not (x2_1 + threshold < x1_2 or x2_2 + threshold < x1_1 or 
                   y2_1 + threshold < y1_2 or y2_2 + threshold < y1_1)
    
    def process_frames_ocr(self, frames_dir: str, video_id: str = None, advanced_ocr: bool = True) -> Tuple[List[Dict], Dict]:
        """
        Process all frames with OCR and extract Hindi text regions
        
        Args:
            frames_dir: Directory containing frame images
            video_id: Optional video ID for creating debug bbox directory
            advanced_ocr: Whether to use advanced OCR with image preprocessing and coordinate scaling
            
        Returns:
            Tuple of (text_buffer, page_data)
            
        Debug Output (if video_id provided):
            - {video_id}_preprocessed.png: Preprocessed frames with GREEN bounding boxes (original OCR coordinates, only if advanced_ocr=True)
            - {video_id}.png: Original frames with RED bounding boxes (scaled coordinates if advanced_ocr=True, original otherwise)
        """
        buffer = []
        page_data = {}
        
        # Create bbox directory for this video if video_id provided
        bbox_dir = None
        if video_id:
            bbox_dir = os.path.join("bbox", video_id)
            self._ensure_directory_exists(bbox_dir)
            print(f"📦 Creating debug frames in: {bbox_dir}/")
            if advanced_ocr:
                print(f"   🟢 *_preprocessed.png: Preprocessed frames (3x scaled) with GREEN boxes (original OCR coordinates)")
                print(f"   🔴 *.png: Original frames with RED boxes (scaled-back coordinates)")
            else:
                print(f"   🔴 *.png: Original frames with RED boxes (original coordinates, no scaling)")
        
        frame_files = sorted(glob(os.path.join(frames_dir, "*.png")))
        
        print("Processing frames with OCR...")
        for frame_path in tqdm(frame_files):
            preprocessed_path = None
            try:
                if advanced_ocr:
                    # Apply preprocessing for better OCR accuracy
                    preprocessed_path = self.preprocess_image_for_ocr(frame_path)
                    
                    # Run OCR on preprocessed image with optimized parameters
                    result = self.ocr_reader.readtext(
                        preprocessed_path,
                        contrast_ths=0.4,
                        adjust_contrast=0.8,
                        text_threshold=0.35,
                        low_text=0.25,
                        decoder='wordbeamsearch',
                        paragraph=True
                    )
                else:
                    # Run OCR on original image with default parameters
                    result = self.ocr_reader.readtext(frame_path)
                
                # Save debug frames BEFORE cleaning up preprocessed file (if bbox_dir exists)
                if bbox_dir and result:
                    if advanced_ocr:
                        # Save preprocessed frame with bounding boxes using original OCR coordinates
                        self.save_preprocessed_frame_with_bboxes(preprocessed_path, result, bbox_dir, frame_path)
                        
                        # Save original frame with bounding boxes using scaled coordinates  
                        self.save_frame_with_bboxes(frame_path, result, bbox_dir, scale_coordinates=True)
                    else:
                        # Save original frame with bounding boxes using original coordinates
                        self.save_frame_with_bboxes(frame_path, result, bbox_dir, scale_coordinates=False)
                    
            finally:
                # Clean up temporary preprocessed file after saving debug frames (only if advanced_ocr was used)
                if advanced_ocr and preprocessed_path and os.path.exists(preprocessed_path):
                    os.unlink(preprocessed_path)
            
            frame_data = []
            
            for sample in result:
                try:
                    # EasyOCR returns either [coordinates, text] or [coordinates, text, confidence]
                    # Coordinates are 4 corner points: [top_left, top_right, bottom_right, bottom_left]
                    coordinates = sample[0]
                    
                    # Debug logging to understand EasyOCR result format
                    if self.logger:
                        coords_len = len(coordinates) if coordinates and hasattr(coordinates, '__len__') else 'Invalid'
                        sample_format = f"[coords, text, conf]" if len(sample) == 3 else f"[coords, text]" if len(sample) == 2 else f"unknown({len(sample)})"
                        self.logger.debug(f"🔍 EasyOCR sample: {sample_format}, coords={coords_len} points")
                        self.logger.debug(f"   Raw sample: {sample}")
                    
                    if not coordinates or not hasattr(coordinates, '__len__'):
                        if self.logger:
                            self.logger.warning(f"⚠️ Invalid coordinates format: {type(coordinates)}")
                        continue
                        
                    if len(coordinates) == 4:
                        top_left, top_right, bottom_right, bottom_left = coordinates
                    elif len(coordinates) == 2:
                        # Sometimes EasyOCR returns just 2 points (top-left, bottom-right)
                        top_left, bottom_right = coordinates
                        top_right = [bottom_right[0], top_left[1]]
                        bottom_left = [top_left[0], bottom_right[1]]
                        if self.logger:
                            self.logger.debug(f"🔧 Converted 2-point to 4-point format")
                    else:
                        if self.logger:
                            self.logger.warning(f"⚠️ Unexpected coordinate format: {len(coordinates)} points")
                            self.logger.warning(f"   Raw coordinates: {coordinates}")
                        continue
                    
                    # Scale coordinates back to original image size only if advanced OCR was used
                    if advanced_ocr:
                        # CRITICAL FIX: Scale coordinates back to original image size
                        # Since we preprocessed with scale factor resize, all coordinates are scaled up
                        top_left = [int(top_left[0] / self.ocr_scale_factor), int(top_left[1] / self.ocr_scale_factor)]
                        top_right = [int(top_right[0] / self.ocr_scale_factor), int(top_right[1] / self.ocr_scale_factor)]
                        bottom_right = [int(bottom_right[0] / self.ocr_scale_factor), int(bottom_right[1] / self.ocr_scale_factor)]
                        bottom_left = [int(bottom_left[0] / self.ocr_scale_factor), int(bottom_left[1] / self.ocr_scale_factor)]
                        
                        if self.logger:
                            self.logger.debug(f"📐 Scaled coordinates back from {self.ocr_scale_factor}x: {coordinates} -> bbox({top_left[0]}, {top_left[1]}, {bottom_right[0]}, {bottom_right[1]})")
                    else:
                        # Use coordinates as-is for normal OCR (no preprocessing/scaling)
                        if self.logger:
                            self.logger.debug(f"📍 Using original coordinates (no scaling): bbox({top_left[0]}, {top_left[1]}, {bottom_right[0]}, {bottom_right[1]})")
                        
                    sample_text = sample[1]
                except (ValueError, IndexError) as e:
                    if self.logger:
                        self.logger.error(f"❌ Error unpacking EasyOCR result: {sample}, error: {e}")
                    continue
                
                frame_data.append({
                    'text': sample_text,
                    'bbox': (top_left[0], top_left[1], bottom_right[0], bottom_right[1])
                })
                
                # Skip if not Hindi text
                if not self.is_hindi(sample_text):
                    continue
                
                sample_bbox = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])
                
                # Check for overlapping regions in buffer
                found_match = False
                for idx, buffer_entry in enumerate(buffer):
                    buffer_bbox = buffer_entry['bbox']
                    
                    if self.regions_overlap(sample_bbox, buffer_bbox, threshold=20):
                        found_match = True
                        if self.logger:
                            self.logger.debug(f"🔗 Consolidating overlapping text: '{sample_text}' at {sample_bbox} with existing at {buffer_bbox}")
                        buffer_text = buffer_entry['all_text']
                        current_max_length = buffer_entry.get('max_length', len(buffer_text))
                        
                        # Update if current text is longer
                        if len(sample_text) > len(buffer_text):
                            buffer[idx]['all_text'] = sample_text
                            buffer[idx]['bbox'] = sample_bbox
                            buffer[idx]['max_length'] = max(len(sample_text), current_max_length)
                        elif len(sample_text) > current_max_length:
                            buffer[idx]['max_length'] = len(sample_text)
                        elif (len(sample_text) > len(buffer_text) and 
                              len(sample_text) <= current_max_length):
                            buffer[idx]['all_text'] = sample_text
                            buffer[idx]['bbox'] = sample_bbox
                        break
                
                # Add new region if no match found
                if not found_match:
                    if self.logger:
                        self.logger.debug(f"➕ Adding new text region: '{sample_text}' at {sample_bbox}")
                    buffer.append({
                        'bbox': sample_bbox,
                        'all_text': sample_text,
                        'max_length': len(sample_text)
                    })
            
            # Normalize path for OS independence before storing in page_data
            normalized_frame_path = os.path.normpath(frame_path)
            page_data[normalized_frame_path] = frame_data
        
        # DEBUG: Print buffer contents for debugging
        print(f"\n🔍 DEBUG: OCR Processing Complete")
        print(f"📊 Total buffer items: {len(buffer)}")
        print(f"📊 Total page_data frames: {len(page_data)}")
        
        if buffer:
            print(f"\n📝 Buffer Text Contents:")
            for i, item in enumerate(buffer, 1):
                bbox = item.get('bbox', 'No bbox')
                text = item.get('all_text', 'No text')
                translation = item.get('marathi_translation', 'No translation')
                print(f"  [{i}] Text: '{text}' | Bbox: {bbox}")
                if translation and translation != 'No translation':
                    print(f"      Translation: '{translation}'")
        else:
            print("⚠️ No text detected in buffer")
        
        return buffer, page_data
    
    def save_preprocessed_frame_with_bboxes(self, preprocessed_path: str, detections: List, bbox_dir: str, original_frame_name: str):
        """Save preprocessed frame with bounding boxes using original OCR coordinates for debugging"""
        try:
            # Load the preprocessed frame (3x scaled)
            image = Image.open(preprocessed_path)
            draw = ImageDraw.Draw(image)
            
            if self.logger:
                self.logger.debug(f"🔍 Creating preprocessed debug frame with {len(detections)} detections")
            
            # Draw green bounding boxes and labels (green to distinguish from original frame debug)
            for detection in detections:
                try:
                    coordinates = detection[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] - in 3x scaled space
                    text = detection[1]
                    confidence = detection[2] if len(detection) > 2 else 0.0
                    
                    # Convert coordinates to bounding box format - NO SCALING BACK since we're on preprocessed image
                    if coordinates and len(coordinates) >= 2:
                        x_coords = [point[0] for point in coordinates]
                        y_coords = [point[1] for point in coordinates]
                        x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
                    else:
                        continue  # Skip invalid coordinates
                except (IndexError, ValueError, TypeError) as e:
                    if self.logger:
                        self.logger.warning(f"⚠️ Error processing detection for preprocessed bbox drawing: {detection}, error: {e}")
                    continue
                
                # Draw GREEN rectangle around detected text (to distinguish from original frame)
                draw.rectangle([x1, y1, x2, y2], outline="green", width=4)
                
                # Add text label with confidence - larger font for 3x scaled image
                try:
                    font_size = 36  # Larger font for 3x scaled image
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()
                
                label = f"{text[:20]}... ({confidence:.2f})" if len(text) > 20 else f"{text} ({confidence:.2f})"
                
                # Position label above the box with more space
                label_y = max(0, y1 - 45)  # More space for larger font
                draw.text((x1, label_y), label, fill="green", font=font)
            
            # Save preprocessed debug frame with _preprocessed suffix
            frame_name = os.path.basename(original_frame_name)
            name_without_ext = os.path.splitext(frame_name)[0]
            debug_path = os.path.join(bbox_dir, f"{name_without_ext}_preprocessed.png")
            image.save(debug_path)
            
            if self.logger:
                self.logger.debug(f"💾 Saved preprocessed debug frame: {debug_path}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error saving preprocessed debug frame: {e}")
    
    def save_frame_with_bboxes(self, frame_path: str, detections: List, bbox_dir: str, scale_coordinates: bool = True):
        """Save original frame with red bounding boxes around detected text for debugging
        
        Args:
            frame_path: Path to the original frame image
            detections: EasyOCR detection results
            bbox_dir: Directory to save debug images
            scale_coordinates: Whether to scale coordinates back from OCR scale factor
        """
        try:
            # Load the original frame
            image = Image.open(frame_path)
            draw = ImageDraw.Draw(image)
            
            # Draw red bounding boxes and labels
            for detection in detections:
                try:
                    coordinates = detection[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text = detection[1]
                    confidence = detection[2] if len(detection) > 2 else 0.0
                    
                    # Conditionally scale coordinates back to original image space for display
                    if scale_coordinates:
                        # Scale coordinates back from OCR preprocessing scale factor
                        scaled_coordinates = []
                        for point in coordinates:
                            scaled_point = [int(point[0] / self.ocr_scale_factor), int(point[1] / self.ocr_scale_factor)]
                            scaled_coordinates.append(scaled_point)
                    else:
                        # Use coordinates as-is
                        scaled_coordinates = coordinates
                    
                    # Convert coordinates to bounding box format
                    if scaled_coordinates and len(scaled_coordinates) >= 2:
                        x_coords = [point[0] for point in scaled_coordinates]
                        y_coords = [point[1] for point in scaled_coordinates]
                        x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
                    else:
                        continue  # Skip invalid coordinates
                except (IndexError, ValueError, TypeError) as e:
                    if self.logger:
                        self.logger.warning(f"⚠️ Error processing detection for bbox drawing: {detection}, error: {e}")
                    continue
                
                # Draw red rectangle around detected text
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                
                # Add text label with confidence
                label = f"{text} ({confidence:.2f})"
                
                # Try to load a basic font, fallback to default
                try:
                    font = ImageFont.truetype("arial.ttf", 12)
                except:
                    try:
                        font = ImageFont.load_default()
                    except:
                        font = None
                
                # Draw text label above the bounding box
                label_y = max(0, y1 - 20)
                if font:
                    draw.text((x1, label_y), label, fill="red", font=font)
                else:
                    draw.text((x1, label_y), label, fill="red")
            
            # Save the annotated frame
            frame_filename = os.path.basename(frame_path)
            bbox_frame_path = os.path.join(bbox_dir, frame_filename)
            image.save(bbox_frame_path)
            
        except Exception as e:
            print(f"Warning: Could not save bbox frame for {frame_path}: {e}")
    
    def translate_text_buffer(self, text_buffer: List[Dict]) -> List[Dict]:
        """Translate Hindi text to Marathi using OpenAI"""
        print("Translating text using OpenAI...")
        
        for idx, buffer_entry in tqdm(enumerate(text_buffer)):
            hindi_text = buffer_entry['all_text']
            
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content":                             """
                            You are a professional Hindi-Marathi translator. Given text may contain Hindi words, phrases, or non-textual content like symbols and numbers.
                            
                            TRANSLATE TO MARATHI if the text is:
                            - Valid Hindi words or phrases with 2+ characters (including technical terms, compound words, scientific vocabulary)
                            - Multi-word Hindi phrases (like "छल्ला चुंबक", "छड़ चुंबक", etc.)
                            - Hindi technical/scientific terms from any domain (physics, chemistry, biology, mathematics, etc.)
                            - Meaningful Hindi text on any topic (education, science, technology, culture, religion, etc.)
                            
                            RESPOND WITH "NO" ONLY if the text is:
                            - Pure numbers or mathematical symbols (like "123", "=", "+", etc.)
                            - Single isolated symbols or punctuation marks
                            - Single letter Hindi characters (like "द", "क", "म", etc.) - these are likely OCR artifacts
                            - Gibberish or corrupted OCR text that makes no sense
                            - Mixed languages that are clearly OCR errors
                            
                            CRITICAL REQUIREMENTS FOR MARATHI TRANSLATIONS:
                            - Use ONLY Devanagari script characters (देवनागरी)
                            - NO English/Roman characters in the translation
                            - NO mixed scripts (like "magnet" or "chumbak")
                            - Use proper Marathi Devanagari spelling
                            
                            EXAMPLES:
                            - "छल्ला चुंबक" → "वलयाकार चुंबक" (correct: valid multi-character Hindi term)
                            - "छड़ चुंबक" → "दंड चुंबक" (correct: valid multi-character Hindi term)  
                            - "प्रकाश" → "प्रकाश" (correct: valid Hindi word)
                            - "द" → "NO" (single letter, likely OCR artifact)
                            - "क" → "NO" (single letter, likely OCR artifact)
                            - "123" → "NO"
                            - "=" → "NO"
                            
                            Respond ONLY with the Marathi translation in Devanagari script or "NO". Be inclusive of ALL Hindi topics and subjects (science, technology, culture, religion, education, etc.) but exclude single letters and OCR artifacts. Ensure pure Devanagari output.
                            """
                        },
                        {
                            "role": "user",
                            "content": hindi_text
                        }
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                
                verdict = response.choices[0].message.content
                if verdict == "NO":
                    text_buffer[idx]['marathi_translation'] = None
                else:
                    text_buffer[idx]['marathi_translation'] = verdict
                    
            except Exception as e:
                print(f"Error translating text '{hindi_text}': {e}")
                text_buffer[idx]['marathi_translation'] = None
        
        # POST-PROCESSING: Remove duplicate translations with OCR gibberish
        print(f"\n🧹 DEBUG: Starting sanity check for duplicate translations...")
        text_buffer = self.remove_duplicate_translations(text_buffer)
        
        # DEBUG: Print translation results for debugging
        print(f"\n🔍 DEBUG: Translation Processing Complete")
        translated_count = sum(1 for item in text_buffer if item.get('marathi_translation') is not None)
        skipped_count = sum(1 for item in text_buffer if item.get('marathi_translation') is None)
        
        print(f"📊 Translation Summary:")
        print(f"   ✅ Translated: {translated_count}")
        print(f"   ❌ Skipped/Failed: {skipped_count}")
        print(f"   📝 Total items: {len(text_buffer)}")
        
        if translated_count > 0:
            print(f"\n📝 Successfully Translated Items:")
            for i, item in enumerate(text_buffer, 1):
                hindi_text = item.get('all_text', 'No text')
                marathi_translation = item.get('marathi_translation')
                bbox = item.get('bbox', 'No bbox')
                
                if marathi_translation is not None:
                    print(f"  [{i}] Hindi: '{hindi_text}'")
                    print(f"      Marathi: '{marathi_translation}'")
                    print(f"      Bbox: {bbox}")
                    print()
        
        if skipped_count > 0:
            print(f"📝 Skipped/Failed Items:")
            for i, item in enumerate(text_buffer, 1):
                hindi_text = item.get('all_text', 'No text')
                marathi_translation = item.get('marathi_translation')
                
                if marathi_translation is None:
                    print(f"  [{i}] Skipped: '{hindi_text}' (not valid Hindi or single syllable)")
        
        return text_buffer
    
    def remove_duplicate_translations(self, text_buffer: List[Dict]) -> List[Dict]:
        """
        Remove duplicate translations at similar Y-coordinates, keeping the cleaner OCR version
        """
        if not text_buffer:
            return text_buffer
        
        # Group by similar Y-coordinates (within 50 pixels)
        y_groups = {}
        for idx, item in enumerate(text_buffer):
            if item.get('marathi_translation') is not None:
                bbox = item.get('bbox')
                if bbox and len(bbox) >= 4:
                    y_center = (bbox[1] + bbox[3]) / 2
                    
                    # Find existing group with similar Y-coordinate
                    found_group = None
                    for group_y in y_groups:
                        if abs(y_center - group_y) <= 50:  # 50 pixel threshold
                            found_group = group_y
                            break
                    
                    if found_group is not None:
                        y_groups[found_group].append((idx, item))
                    else:
                        y_groups[y_center] = [(idx, item)]
        
        # Check each group for duplicate translations
        items_to_remove = set()
        
        for group_y, group_items in y_groups.items():
            if len(group_items) <= 1:
                continue
                
            # Group by Marathi translation within this Y-group
            translation_groups = {}
            for idx, item in group_items:
                marathi_text = item['marathi_translation']
                if marathi_text in translation_groups:
                    translation_groups[marathi_text].append((idx, item))
                else:
                    translation_groups[marathi_text] = [(idx, item)]
            
            # For each translation that appears multiple times, keep only the cleanest version
            for marathi_text, duplicate_items in translation_groups.items():
                if len(duplicate_items) > 1:
                    print(f"🔍 Found {len(duplicate_items)} duplicates for '{marathi_text}' at Y≈{group_y:.0f}")
                    
                    # Use GPT to find the cleanest Hindi text
                    best_idx = self.find_cleanest_hindi_text(duplicate_items)
                    
                    # Mark others for removal
                    for idx, item in duplicate_items:
                        if idx != best_idx:
                            items_to_remove.add(idx)
                            print(f"  ❌ Removing: '{item['all_text'][:50]}...' (has OCR gibberish)")
                        else:
                            print(f"  ✅ Keeping: '{item['all_text'][:50]}...' (cleanest version)")
        
        # Remove marked items (in reverse order to maintain indices)
        for idx in sorted(items_to_remove, reverse=True):
            text_buffer[idx]['marathi_translation'] = None  # Mark as skipped instead of removing
        
        print(f"🧹 Removed {len(items_to_remove)} duplicate translations")
        return text_buffer
    
    def find_cleanest_hindi_text(self, duplicate_items: List[Tuple[int, Dict]]) -> int:
        """
        Use GPT to determine which Hindi text version has less OCR gibberish
        """
        if len(duplicate_items) <= 1:
            return duplicate_items[0][0] if duplicate_items else -1
        
        # Prepare comparison text for GPT
        hindi_options = []
        for i, (idx, item) in enumerate(duplicate_items):
            hindi_options.append(f"Option {i+1}: {item['all_text']}")
        
        comparison_text = "\n".join(hindi_options)
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": """
                        You are an OCR quality evaluator for Hindi text. Given multiple versions of the same text, choose the one with the LEAST OCR errors and gibberish.
                        
                        Look for these OCR problems:
                        - Random symbols: == - ~ = $ ़ 
                        - Broken/corrupted characters
                        - Mixed languages inappropriately  
                        - Incomplete words at the end
                        - Repeated characters or spaces
                        
                        Choose the option that:
                        - Has proper Hindi text structure
                        - Contains fewer random symbols/gibberish
                        - Has complete, meaningful sentences
                        - Is most readable and coherent
                        
                        Respond with ONLY the number (1, 2, 3, etc.) of the best option.
                        """
                    },
                    {
                        "role": "user",
                        "content": f"Which Hindi text version has the least OCR errors?\n\n{comparison_text}"
                    }
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            choice = response.choices[0].message.content.strip()
            if choice.isdigit():
                option_num = int(choice) - 1  # Convert to 0-based index
                if 0 <= option_num < len(duplicate_items):
                    return duplicate_items[option_num][0]  # Return the buffer index
                    
        except Exception as e:
            print(f"Error in OCR quality evaluation: {e}")
        
        # Fallback: choose the shortest text (often has less gibberish)
        shortest_idx = min(duplicate_items, key=lambda x: len(x[1]['all_text']))
        return shortest_idx[0]
    
    def process_frame_with_text_overlay(self, image_path: str, text_buffer: List[Dict], 
                                      page_data: Dict, logger: Optional[logging.Logger] = None) -> Image.Image:
        """Process frame with Marathi text overlay using the working pillow_basic function"""
        from pillow_basic import process_frame_with_text_buffer_basic
        
        # Normalize paths for OS independence - both storage and lookup use same normalization
        normalized_image_path = os.path.normpath(image_path)
        
        # Use class logger if no specific logger provided
        if logger is None:
            logger = self.logger
        
        # Debug logging for path investigation
        if logger:
            logger.debug(f"🔍 DEBUG - Frame: {os.path.basename(image_path)}")
            logger.debug(f"  📂 Original path: '{image_path}'")
            logger.debug(f"  📂 Normalized path: '{normalized_image_path}'")
            logger.debug(f"  📊 Page data keys: {len(page_data)}")
            logger.debug(f"  ✅ Normalized path in page_data: {normalized_image_path in page_data}")
            if normalized_image_path in page_data:
                logger.debug(f"  📝 Text detections in frame: {len(page_data[normalized_image_path])}")
            else:
                logger.warning(f"  ⚠️  No frame data found for this frame!")
                if len(page_data) > 0:
                    sample_key = list(page_data.keys())[0]
                    logger.debug(f"  🔑 Sample page_data key: '{sample_key}'")
            logger.debug(f"  🔄 Using OS-independent normalized paths...")
        
        # Use the existing working function with normalized path and logger
        return process_frame_with_text_buffer_basic(
            image_path=normalized_image_path,
            text_buffer=text_buffer, 
            page_data=page_data,
            font_path=self.font_path,
            logger=logger
        )
    
    def process_frame_with_page_data(self, frame_path: str, text_buffer: List[Dict], 
                                   page_data: Dict, logger: Optional[logging.Logger] = None):
        """
        Process a single frame with text overlay using page_data directly (moving text mode)
        
        This method directly uses page_data for each frame, translating text on-the-fly.
        Suitable for videos with moving text where buffer-based approach fails.
        """
        from PIL import Image, ImageDraw, ImageFont
        
        # Load the original frame
        image = Image.open(frame_path)
        image_copy = image.copy()
        draw = ImageDraw.Draw(image_copy)
        
        # Get the frame name for page_data lookup
        frame_name = os.path.normpath(frame_path)
        
        # Check if this frame has any detected text
        if frame_name not in page_data:
            if logger:
                logger.debug(f"No text data for frame: {frame_name}")
            return image_copy
        
        frame_texts = page_data[frame_name]
        if not frame_texts:
            if logger:
                logger.debug(f"No text detected in frame: {frame_name}")
            return image_copy
        
        # Create a lookup map for translations from text_buffer
        translation_map = {}
        for buffer_item in text_buffer:
            if buffer_item.get('marathi_translation'):
                hindi_text = buffer_item['all_text'].strip().lower()
                marathi_text = buffer_item['marathi_translation'].strip()
                translation_map[hindi_text] = marathi_text
        
        if logger:
            logger.debug(f"🔄 Moving Text Mode: Processing {len(frame_texts)} text regions for frame {os.path.basename(frame_path)}")
        
        # Process each detected text region in this frame
        for text_info in frame_texts:
            try:
                detected_text = text_info['text'].strip()
                bbox = text_info['bbox']  # [x1, y1, x2, y2]
                
                # Skip empty text
                if not detected_text:
                    continue
                
                # Look for translation in buffer
                marathi_translation = None
                detected_lower = detected_text.lower()
                
                # Try exact match first
                if detected_lower in translation_map:
                    marathi_translation = translation_map[detected_lower]
                else:
                    # Try partial match (for growing/partial text)
                    for hindi_key, marathi_val in translation_map.items():
                        if (detected_lower in hindi_key or 
                            hindi_key in detected_lower or
                            any(word in hindi_key for word in detected_lower.split() if len(word) > 2)):
                            marathi_translation = marathi_val
                            break
                
                if not marathi_translation:
                    if logger:
                        logger.debug(f"No translation found for: '{detected_text}'")
                    continue
                
                # Whiten the detected area
                x1, y1, x2, y2 = bbox
                draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
                
                # Calculate font size based on bounding box
                box_width = x2 - x1
                box_height = y2 - y1
                
                # Start with a reasonable font size
                font_size = max(12, min(int(box_height * 0.6), 48))
                
                try:
                    font = ImageFont.truetype(self.font_path, font_size)
                except:
                    font = ImageFont.load_default()
                
                # Get text dimensions
                text_bbox = font.getbbox(marathi_translation)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Adjust font size if text doesn't fit
                while (text_width > box_width or text_height > box_height) and font_size > 8:
                    font_size -= 2
                    try:
                        font = ImageFont.truetype(self.font_path, font_size)
                    except:
                        font = ImageFont.load_default()
                    text_bbox = font.getbbox(marathi_translation)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                
                # Center the text in the bounding box
                text_x = x1 + (box_width - text_width) // 2
                text_y = y1 + (box_height - text_height) // 2
                
                # Ensure text stays within bounds
                text_x = max(x1, min(text_x, x2 - text_width))
                text_y = max(y1, min(text_y, y2 - text_height))
                
                # Draw the Marathi text
                draw.text((text_x, text_y), marathi_translation, font=font, fill=(0, 0, 0))
                
                if logger:
                    logger.debug(f"✅ Added translation: '{detected_text}' -> '{marathi_translation}' at ({text_x}, {text_y})")
                
            except Exception as e:
                if logger:
                    logger.warning(f"Error processing text region: {e}")
                continue
        
        return image_copy
    
    def process_all_frames(self, frames_dir: str, output_dir: str, text_buffer: List[Dict], 
                          page_data: Dict, logger: Optional[logging.Logger] = None, moving_text: bool = False, advanced_ocr: bool = True) -> bool:
        """
        Process all frames with text overlay
        
        Args:
            frames_dir: Directory containing extracted frames
            output_dir: Directory to save processed frames
            text_buffer: Buffer containing text translations
            page_data: Frame-by-frame text detection data
            logger: Optional logger
            moving_text: Whether to use moving text mode (page_data) or static text mode (buffer)
            advanced_ocr: Whether advanced OCR was used (affects coordinate scaling in moving text mode)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            if not self._ensure_directory_exists(output_dir):
                return False
            
            frame_files = sorted(glob(os.path.join(frames_dir, "*.png")))
            
            # Use class logger if no specific logger provided
            if logger is None:
                logger = self.logger
                
            if logger:
                logger.info("Processing frames with text overlay...")
            
            for frame_path in tqdm(frame_files):
                if moving_text:
                    # Moving text mode: Use page_data for frame-by-frame text placement
                    result_frame = self.process_frame_with_page_data(
                        frame_path, text_buffer, page_data, logger
                    )
                else:
                    # Static text mode: Use optimized buffer-based text placement
                    result_frame = self.process_frame_with_text_overlay(
                        frame_path, text_buffer, page_data, logger
                    )
                
                frame_name = os.path.basename(frame_path)
                output_path = os.path.join(output_dir, frame_name)
                result_frame.save(output_path)
            
            return True
            
        except Exception as e:
            print(f"Error processing frames: {e}")
            return False
    
    def transcribe_and_translate_audio(self, audio_path: str, text_buffer: List[Dict] = None) -> Optional[str]:
        """Transcribe audio and translate to Marathi using visual text translations for consistency"""
        try:
            # Transcribe audio
            with open(audio_path, "rb") as audio_file:
                transcription = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language='hi'
                )
            
            hindi_text = transcription.text
            print(f"Transcribed Hindi text: {hindi_text}")
            
            # Create translation dictionary from visual text buffer for consistency
            translation_dict = {}
            if text_buffer:
                for buffer_entry in text_buffer:
                    if buffer_entry.get('marathi_translation') and buffer_entry['marathi_translation'] != 'NO':
                        hindi_phrase = buffer_entry['all_text']
                        marathi_translation = buffer_entry['marathi_translation']
                        translation_dict[hindi_phrase] = marathi_translation
                        
                print(f"🔗 Using {len(translation_dict)} visual text translations as reference for audio consistency")
                if translation_dict:
                    print("📝 Translation dictionary:")
                    for i, (hindi, marathi) in enumerate(translation_dict.items(), 1):
                        print(f"  {i}. '{hindi}' = '{marathi}'")
            
            # Build translation dictionary string
            translation_reference = ""
            if translation_dict:
                dict_entries = []
                for hindi_phrase, marathi_translation in translation_dict.items():
                    dict_entries.append(f"{hindi_phrase} = {marathi_translation}")
                translation_reference = f"""


Use this translation dictionary as reference

{chr(10).join(dict_entries)}
"""
            
            # Translate to Marathi with dictionary reference  
            prompt = f"""Translate the given text from hindi to marathi without losing context {{{hindi_text}}}{translation_reference}"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1  # Lower temperature for more consistent translations
            )
            
            marathi_text = response.choices[0].message.content
            print(f"Translated Marathi text: {marathi_text}")
            
            return marathi_text
            
        except Exception as e:
            print(f"Error in audio processing: {e}")
            return None
    
    def generate_marathi_audio(self, text: str, output_path: str) -> bool:
        """Generate Marathi audio using Google TTS"""
        try:
            voice = texttospeech.VoiceSelectionParams(
                language_code='mr-IN',
                name='mr-IN-Standard-A'  # Using standard voice as Chirp3-HD might not be available
            )
            
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=0.7
            )
            
            input_text = texttospeech.SynthesisInput(text=text)
            
            response = self.tts_client.synthesize_speech(
                input=input_text,
                voice=voice,
                audio_config=audio_config,
            )
            
            with open(output_path, "wb") as out:
                out.write(response.audio_content)
                print(f'Audio content written to file "{output_path}"')
            
            return True
            
        except Exception as e:
            print(f"Error generating audio: {e}")
            return False
    
    def get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds"""
        try:
            audio = MP3(audio_path)
            return audio.info.length
        except Exception as e:
            print(f"Error getting audio duration: {e}")
            return 5.0  # Default fallback
    
    def create_final_video(self, frames_pattern: str, audio_file: str, 
                          output_file: str, total_frames: int) -> bool:
        """Create final video with synchronized audio using optimized FFmpeg"""
        try:
            if self.logger:
                self.logger.info(f"🎬 Creating final video with FFmpeg optimization...")
            
            audio_duration = self.get_audio_duration(audio_file)
            required_fps = max(15, min(60, total_frames / audio_duration))  # Clamp FPS to reasonable range
            
            if self.logger:
                self.logger.debug(f"📊 Video specs: {total_frames} frames, {audio_duration:.2f}s, {required_fps:.2f} FPS")
            
            # Optimized FFmpeg command for faster encoding
            cmd = [
                'ffmpeg',
                '-threads', '0',  # Use all available CPU threads
                '-framerate', f'{required_fps:.2f}',
                '-i', frames_pattern,
                '-i', audio_file,
                
                # Video encoding optimization
                '-c:v', 'libx264',
                '-preset', 'medium',  # Balance between speed and compression
                '-crf', '23',  # Constant Rate Factor for good quality (18-28 range)
                '-pix_fmt', 'yuv420p',  # Compatible pixel format
                '-movflags', '+faststart',  # Optimize for streaming
                
                # Audio encoding optimization  
                '-c:a', 'aac',
                '-b:a', '128k',  # Good quality for speech
                '-ar', '44100',  # Standard sample rate
                
                # Timing and synchronization
                '-t', str(audio_duration),
                '-shortest',  # End when shortest stream ends
                '-avoid_negative_ts', 'make_zero',  # Fix timing issues
                
                # Performance optimizations
                '-threads', '0',  # Multi-threading for encoding
                '-tune', 'film',  # Optimize for typical video content
                
                # Output options
                '-y',  # Overwrite existing files
                '-hide_banner',
                '-loglevel', 'warning',
                output_file
            ]
            
            # Try hardware encoding first for even better performance
            hw_cmd = [
                'ffmpeg',
                '-threads', '0',
                '-framerate', f'{required_fps:.2f}',
                '-i', frames_pattern,
                '-i', audio_file,
                
                # Hardware video encoding (if available)
                '-c:v', 'h264_nvenc',  # NVIDIA hardware encoder
                '-preset', 'fast',
                '-cq', '23',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                
                # Audio (same as before)
                '-c:a', 'aac',
                '-b:a', '128k',
                '-ar', '44100',
                
                '-t', str(audio_duration),
                '-shortest',
                '-avoid_negative_ts', 'make_zero',
                
                '-y',
                '-hide_banner',
                '-loglevel', 'warning',
                output_file
            ]
            
            # Check if hardware encoding is available
            if (self.ffmpeg_capabilities.get('available', False) and 
                any('nvenc' in enc.lower() for enc in self.ffmpeg_capabilities.get('encoders', []))):
                try:
                    # Try hardware encoding first
                    result = subprocess.run(hw_cmd, capture_output=True, text=True, check=True, timeout=600)
                    encoding_method = "hardware (NVENC)"
                    if self.logger:
                        self.logger.info("⚡ Video created with hardware acceleration (NVENC)")
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    # Fallback to software encoding
                    if self.logger:
                        self.logger.info("🔄 Hardware encoding failed, using optimized software encoding...")
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
                    encoding_method = "software (x264)"
                    if self.logger:
                        self.logger.info("✅ Video created with software encoding")
            else:
                # Use software encoding directly (hardware not available)
                if self.logger:
                    self.logger.info("💻 Using optimized software encoding (hardware not detected)...")
                result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
                encoding_method = "software (x264)"
                if self.logger:
                    self.logger.info("✅ Video created with software encoding")
            
            # Verify output file
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
                print(f"Video created successfully with {required_fps:.1f} fps using {encoding_method}")
                print(f"Output file: {output_file} ({file_size:.1f} MB)")
                
                if self.logger:
                    self.logger.debug(f"📊 Final video: {file_size:.1f} MB, {required_fps:.1f} FPS, {encoding_method}")
                
                return True
            else:
                raise Exception("Output video file is empty or missing")
            
        except subprocess.TimeoutExpired:
            error_msg = "Video creation timed out (>10 minutes)"
            print(f"Error creating video: {error_msg}")
            if self.logger:
                self.logger.error(error_msg)
            return False
        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg error: {e.stderr}"
            print(f"Error creating video: {error_msg}")
            if self.logger:
                self.logger.error(error_msg)
            return False
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"Error creating video: {error_msg}")
            if self.logger:
                self.logger.error(error_msg)
            return False
    
    def cleanup_temp_files(self, temp_dirs: List[str], temp_files: List[str]):
        """Clean up temporary files and directories"""
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"Cleaned up directory: {temp_dir}")
        
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"Cleaned up file: {temp_file}")
    
    def process_video(self, video_path: str, video_id: Optional[str] = None, cleanup: bool = True, moving_text: bool = False, advanced_ocr: bool = True) -> str:
        """
        Main video processing pipeline
        
        Args:
            video_path: Path to input video
            video_id: Optional specific video ID to use (if None, generates new UUID)
            cleanup: Whether to cleanup temporary files
            moving_text: Whether video contains moving text (uses different processing logic)
            advanced_ocr: Whether to use advanced OCR with image preprocessing and coordinate scaling
            
        Returns:
            Path to output video file
        """
        # Use provided video ID or generate unique video ID
        if video_id is None:
            video_id = str(uuid.uuid4())
        else:
            print(f"Using provided video ID: {video_id}")
        
        # Setup job-specific logging
        logger = self.setup_logger(video_id)
        logger.info(f"📹 Processing video: {os.path.basename(video_path)}")
        logger.info(f"🆔 Job ID: {video_id}")
        logger.info("")
        
        # Ensure base directories exist
        self._ensure_directory_exists("frames")
        self._ensure_directory_exists("translated_frames")
        
        # Create directories using OS-independent path joining
        frames_dir = os.path.join("frames", video_id)
        translated_frames_dir = os.path.join("translated_frames", video_id)
        
        # Organized file paths with proper folder structure and video_id traceability
        audio_file = os.path.join("audios", f"audio_{video_id}.wav")
        marathi_audio_file = os.path.join("translated_audio", f"marathi_audio_{video_id}.mp3")
        output_video = os.path.join("output", f"output_{video_id}.mp4")
        
        temp_dirs = [frames_dir, translated_frames_dir]
        temp_files = [audio_file, marathi_audio_file]
        
        try:
            # Step 1: Extract frames
            print("Step 1: Extracting video frames...")
            if not self.extract_video_frames(video_path, frames_dir):
                raise Exception("Failed to extract frames")
            
            # Step 2: Extract audio
            print("Step 2: Extracting audio...")
            if not self.extract_audio(video_path, audio_file):
                raise Exception("Failed to extract audio")
            
            # Step 3: Process frames with OCR
            print("Step 3: Processing frames with OCR...")
            text_buffer, page_data = self.process_frames_ocr(frames_dir, video_id, advanced_ocr)
            
            # Step 4: Translate text
            print("Step 4: Translating text...")
            text_buffer = self.translate_text_buffer(text_buffer)
            
            # Step 5: Process frames with text overlay
            if moving_text:
                logger.info("Step 5: Creating frames with Marathi text overlay (Moving Text Mode - using page_data)...")
                print("🔄 Moving Text Mode: Using page_data for frame-by-frame text placement")
            else:
                logger.info("Step 5: Creating frames with Marathi text overlay (Static Text Mode - using buffer)...")
                print("🔧 Static Text Mode: Using optimized buffer-based text placement")
            
            if not self.process_all_frames(frames_dir, translated_frames_dir, text_buffer, page_data, logger, moving_text, advanced_ocr):
                raise Exception("Failed to process frames with text overlay")
            
            # Step 6: Transcribe and translate audio
            print("Step 6: Processing audio...")
            marathi_text = self.transcribe_and_translate_audio(audio_file, text_buffer)
            if not marathi_text:
                raise Exception("Failed to process audio")
            
            # Step 7: Generate Marathi audio
            print("Step 7: Generating Marathi audio...")
            if not self.generate_marathi_audio(marathi_text, marathi_audio_file):
                raise Exception("Failed to generate Marathi audio")
            
            # Step 8: Create final video
            print("Step 8: Creating final video...")
            total_frames = len(glob(os.path.join(translated_frames_dir, "*.png")))
            frames_pattern = os.path.join(translated_frames_dir, "frame_%04d.png")
            
            if not self.create_final_video(frames_pattern, marathi_audio_file, output_video, total_frames):
                raise Exception("Failed to create final video")
            
            print(f"✅ Video translation completed successfully!")
            print(f"Output video: {output_video}")
            
            # Cleanup if requested
            if cleanup:
                self.cleanup_temp_files(temp_dirs, temp_files)
            
            return output_video
            
        except Exception as e:
            print(f"❌ Error in video processing: {e}")
            
            # Cleanup on error if requested
            if cleanup:
                self.cleanup_temp_files(temp_dirs, temp_files)
            
            raise e

    def create_naked_video_frames(self, frames_dir: str, output_dir: str, text_buffer: List[Dict], 
                                page_data: Dict, logger: Optional[logging.Logger] = None) -> bool:
        """
        Stage 1a: Create naked frames by removing Hindi text and replacing with white boxes
        """
        try:
            if not self._ensure_directory_exists(output_dir):
                return False
            
            frame_files = sorted(glob(os.path.join(frames_dir, "*.png")))
            
            if logger:
                logger.info("Creating naked frames (removing Hindi text)...")
            
            for frame_path in tqdm(frame_files):
                # Load the original frame
                image = Image.open(frame_path)
                image_copy = image.copy()
                draw = ImageDraw.Draw(image_copy)
                
                # Get frame-specific text data
                frame_name = os.path.normpath(frame_path)
                if frame_name in page_data:
                    frame_texts = page_data[frame_name]
                    
                    # White out all detected Hindi text regions
                    for text_info in frame_texts:
                        detected_text = text_info['text'].strip()
                        
                        # Only process Hindi text
                        if self.is_hindi(detected_text):
                            bbox = text_info['bbox']  # [x1, y1, x2, y2]
                            x1, y1, x2, y2 = bbox
                            
                            # Fill with white
                            draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
                            
                            if logger:
                                logger.debug(f"Removed Hindi text: '{detected_text}' at {bbox}")
                
                # Save naked frame
                frame_name = os.path.basename(frame_path)
                output_path = os.path.join(output_dir, frame_name)
                image_copy.save(output_path)
            
            return True
            
        except Exception as e:
            if logger:
                logger.error(f"Error creating naked frames: {e}")
            return False

    def separate_background_music(self, audio_path: str, job_id: str, 
                                logger: Optional[logging.Logger] = None) -> Tuple[str, str]:
        """
        Stage 1b: Separate background music from voice audio
        
        Returns:
            Tuple of (voice_path, music_path)
        """
        try:
            separator = AudioSeparator(logger)
            voice_path, music_path = separator.separate_audio(audio_path, job_id)
            
            if logger:
                logger.info(f"Audio separated - Voice: {voice_path}, Music: {music_path}")
            
            return voice_path, music_path
            
        except Exception as e:
            if logger:
                logger.error(f"Audio separation failed: {e}")
            raise e

    def get_editable_translations(self, text_buffer: List[Dict], 
                                hindi_transcription: str = None, 
                                marathi_translation: str = None) -> Dict:
        """
        Stage 1c & 1d: Prepare editable translation data for UI
        """
        # Prepare audio translations
        audio_translations = {
            'hindi_transcription': hindi_transcription or '',
            'marathi_translation': marathi_translation or ''
        }
        
        # Prepare text translations
        text_translations = []
        for item in text_buffer:
            if item.get('marathi_translation') and item['marathi_translation'] != 'NO':
                text_translations.append({
                    'hindi_text': item['all_text'],
                    'marathi_translation': item['marathi_translation'],
                    'bbox': list(item['bbox'])
                })
        
        return {
            'audio_translations': audio_translations,
            'text_translations': text_translations
        }

    def apply_edited_translations(self, text_buffer: List[Dict], 
                                edited_text_translations: List[Dict]) -> List[Dict]:
        """
        Apply user-edited translations back to text buffer
        """
        # Create lookup map for edits
        edit_map = {}
        for item in edited_text_translations:
            edit_map[item['hindi_text']] = item['marathi_translation']
        
        # Apply edits to buffer
        for buffer_item in text_buffer:
            hindi_text = buffer_item['all_text']
            if hindi_text in edit_map:
                buffer_item['marathi_translation'] = edit_map[hindi_text]
                
        return text_buffer

    def create_final_video_with_music(self, frames_pattern: str, voice_audio: str, 
                                    music_audio: str, output_file: str, total_frames: int,
                                    voice_volume: float = 1.0, music_volume: float = 0.3) -> bool:
        """
        Stage 2: Create final video with separated voice and background music
        """
        try:
            if self.logger:
                self.logger.info("Creating final video with voice and background music...")
            
            # First combine voice and music
            separator = AudioSeparator(self.logger)
            combined_audio = f"temp_audio_separation/combined_{uuid.uuid4().hex}.mp3"
            
            if not separator.combine_voice_music(voice_audio, music_audio, combined_audio, 
                                            voice_volume, music_volume):
                raise Exception("Failed to combine voice and music")
            
            # Create video with combined audio
            success = self.create_final_video(frames_pattern, combined_audio, output_file, total_frames)
            
            # Cleanup temp combined audio
            if os.path.exists(combined_audio):
                os.remove(combined_audio)
                
            return success
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating final video with music: {e}")
            return False

    def process_video_with_editing_workflow(self, video_path: str, video_id: Optional[str] = None, 
                                        cleanup: bool = True, moving_text: bool = False, 
                                        advanced_ocr: bool = True) -> Tuple[str, Dict]:
        """
        Modified video processing pipeline that supports the editing workflow
        
        Returns:
            Tuple of (naked_video_path, editing_data)
        """
        # Use provided video ID or generate unique video ID
        if video_id is None:
            video_id = str(uuid.uuid4())
        
        # Setup job-specific logging
        logger = self.setup_logger(video_id)
        logger.info(f"Processing video with editing workflow: {os.path.basename(video_path)}")
        
        # Create directories
        frames_dir = os.path.join("frames", video_id)
        naked_frames_dir = os.path.join("naked_frames", video_id)
        
        # File paths
        audio_file = os.path.join("audios", f"audio_{video_id}.wav")
        
        temp_dirs = [frames_dir, naked_frames_dir]
        temp_files = [audio_file]
        
        try:
            # Step 1: Extract frames and audio
            logger.info("Step 1: Extracting video frames and audio...")
            if not self.extract_video_frames(video_path, frames_dir):
                raise Exception("Failed to extract frames")
            
            if not self.extract_audio(video_path, audio_file):
                raise Exception("Failed to extract audio")
            
            # Step 2: Process frames with OCR
            logger.info("Step 2: Processing frames with OCR...")
            text_buffer, page_data = self.process_frames_ocr(frames_dir, video_id, advanced_ocr)
            
            # Step 3: Translate text
            logger.info("Step 3: Translating text...")
            text_buffer = self.translate_text_buffer(text_buffer)
            
            # Step 4: Create naked video frames (remove Hindi text)
            logger.info("Step 4: Creating naked frames...")
            if not self.create_naked_video_frames(frames_dir, naked_frames_dir, text_buffer, page_data, logger):
                raise Exception("Failed to create naked frames")
            
            # Step 5: Separate audio
            logger.info("Step 5: Separating background music...")
            voice_path, music_path = self.separate_background_music(audio_file, video_id, logger)
            
            # Step 6: Process audio transcription and translation
            logger.info("Step 6: Processing audio...")
            marathi_text = self.transcribe_and_translate_audio(voice_path, text_buffer)
            if not marathi_text:
                logger.warning("Audio processing failed, using fallback")
                marathi_text = "Audio translation unavailable"
            
            # Prepare editing data
            editing_data = {
                'text_buffer': text_buffer,
                'page_data': page_data,
                'voice_audio_path': voice_path,
                'music_audio_path': music_path,
                'hindi_transcription': '',  # Would need to implement transcription
                'marathi_audio_translation': marathi_text,
                'naked_frames_dir': naked_frames_dir,
                'original_frames_dir': frames_dir
            }
            
            logger.info("Video processing with editing workflow completed successfully!")
            
            # Don't cleanup yet - files needed for editing
            return naked_frames_dir, editing_data
            
        except Exception as e:
            logger.error(f"Error in video processing workflow: {e}")
            
            # Cleanup on error if requested
            if cleanup:
                self.cleanup_temp_files(temp_dirs, temp_files)
            
            raise e

def translate_video(
    video_path: str,
    font_path: str = "noto-sans-devanagari/NotoSansDevanagari-Regular.ttf",
    cleanup: bool = True,
    env_file: str = ".env",
    video_id: Optional[str] = None,
    moving_text: bool = False,
    advanced_ocr: bool = True
) -> str:
    """
    Translate Hindi video to Marathi with credentials from .env file
    
    Args:
        video_path: Path to input video file
        font_path: Path to Devanagari font file
        cleanup: Whether to cleanup temporary files
        env_file: Path to .env file containing credentials
        video_id: Optional specific video ID to use (if None, generates new UUID)
        moving_text: Whether video contains moving text (uses different processing logic)
        advanced_ocr: Whether to use advanced OCR with image preprocessing and coordinate scaling
        
    Returns:
        Path to output video file
        
    Raises:
        FileNotFoundError: If required files don't exist
        ValueError: If required environment variables are missing
        Exception: If translation process fails
    """
    from dotenv import load_dotenv
    
    # Ensure base directories exist before starting
    base_directories = ["frames", "translated_frames", "output", "temp", "bbox", "audios", "translated_audio"]
    for directory in base_directories:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            print(f"❌ Error creating base directory {directory}: {e}")
    
    # Load environment variables
    load_dotenv(env_file)
    
    # Get credentials from environment
    openai_api_key = os.getenv('OPENAI_API_KEY')
    google_credentials_path = os.getenv('GOOGLE_CREDENTIALS_PATH')
    
    # Validate environment variables
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    if not google_credentials_path:
        raise ValueError("GOOGLE_CREDENTIALS_PATH not found in environment variables")
    
    # Validate file paths
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not os.path.exists(google_credentials_path):
        raise FileNotFoundError(f"Google credentials file not found: {google_credentials_path}")
    
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Font file not found: {font_path}")
    
    try:
        # Initialize translator
        translator = VideoTranslator(
            openai_api_key=openai_api_key,
            google_credentials_path=google_credentials_path,
            font_path=font_path
        )
        
        # Process video
        output_video = translator.process_video(
            video_path=video_path,
            video_id=video_id,
            cleanup=cleanup,
            moving_text=moving_text,
            advanced_ocr=advanced_ocr
        )
        
        print(f"\n🎉 Translation completed successfully!")
        print(f"📹 Output video: {output_video}")
        
        return output_video
        
    except Exception as e:
        print(f"\n❌ Translation failed: {e}")
        raise e


def main():
    """Command line interface for video translation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Translate Hindi video to Marathi")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--font-path", default="noto-sans-devanagari/NotoSansDevanagari-Regular.ttf", 
                       help="Path to Devanagari font file")
    parser.add_argument("--no-cleanup", action="store_true", help="Don't cleanup temporary files")
    parser.add_argument("--env-file", default=".env", help="Path to .env file")
    
    args = parser.parse_args()
    
    try:
        output_video = translate_video(
            video_path=args.video_path,
            font_path=args.font_path,
            cleanup=not args.no_cleanup,
            env_file=args.env_file
        )
        
        return 0
        
    except Exception as e:
        return 1


if __name__ == "__main__":
    exit(main())
