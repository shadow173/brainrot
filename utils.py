from functools import lru_cache
import hashlib
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Optional, Tuple, List, Callable

import gdown
import streamlit as st
from PIL import ImageFont
import numpy as np
import cv2
from gtts import gTTS
from pydub import AudioSegment
from PIL import Image, ImageDraw, ImageFont
import time
import base64
import random
import platform

logger = logging.getLogger(__name__)


def get_cache_key(url: str) -> str:
    """
    Generate a unique cache key for a video URL.

    Args:
        url (str): The URL of the video

    Returns:
        str: A unique hash key for caching
    """
    return hashlib.md5(url.encode()).hexdigest()


@st.cache_data
def cache_background_video(url: str) -> Tuple[str, Optional[str]]:
    """
    Download and cache background video with Streamlit's caching.

    Args:
        url (str): The URL of the video to download

    Returns:
        Tuple[str, Optional[str]]: Path to cached video and any error message
    """
    try:
        cache_dir = Path(tempfile.gettempdir()) / "streamlit_backgrounds"
        cache_dir.mkdir(exist_ok=True)

        # Generate cache key from URL
        cache_key = get_cache_key(url)
        cached_file = cache_dir / f"background_{cache_key}.mp4"

        # If cached file exists and is valid, return it
        if cached_file.exists():
            is_valid, error_msg = verify_video_file(str(cached_file))
            if is_valid:
                logger.info(f"Using cached video: {cached_file}")
                return str(cached_file), None
            else:
                logger.warning(f"Invalid cached video: {error_msg}")
                cached_file.unlink()

        # Extract file ID from Google Drive URL
        if '/file/d/' in url:
            file_id = url.split('/file/d/')[1].split('/')[0]
        else:
            raise Exception("Invalid Google Drive URL format")

        logger.info(f"Downloading video with file ID: {file_id}")

        # Use gdown to download
        download_url = f'https://drive.google.com/uc?id={file_id}'
        success = gdown.download(download_url,
                                 str(cached_file),
                                 quiet=False,
                                 fuzzy=True,
                                 use_cookies=False)

        if not success:
            raise Exception("gdown download failed")

        # Verify downloaded file
        is_valid, error_msg = verify_video_file(str(cached_file))
        if not is_valid:
            raise Exception(f"Invalid downloaded video: {error_msg}")

        logger.info(f"Successfully cached video to: {cached_file}")
        return str(cached_file), None

    except Exception as e:
        error_msg = f"Error caching background video: {str(e)}"
        logger.error(error_msg)
        return "", error_msg


@st.cache_data
def get_background_video(url: str) -> Tuple[str, Optional[str]]:
    """
    Get background video with caching support.

    Args:
        url (str): The URL of the video

    Returns:
        Tuple[str, Optional[str]]: Path to video file and any error message
    """
    return cache_background_video(url)


@lru_cache(maxsize=10)
def get_font(font_size: int, system: str):
    """
    Cache font loading to avoid repeated disk access.
    """
    try:
        if system == 'Windows':
            return ImageFont.truetype("arialbd.ttf", font_size)
        elif system == 'Darwin':  # macOS
            return ImageFont.truetype("/Library/Fonts/Arial Bold.ttf",
                                      font_size)
        else:  # Linux
            return ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                font_size)
    except OSError:
        logger.warning("Preferred font not found, using default")
        return ImageFont.load_default()


def split_into_chunks(text: str, max_words: int = 2) -> List[dict]:
    """
    Split text into small chunks with timing information.

    Args:
        text (str): Text to split into chunks
        max_words (int): Maximum words per chunk

    Returns:
        List[dict]: List of chunks with text and estimated duration
    """
    # Remove any existing newlines and extra spaces
    text = ' '.join(text.split())
    words = text.split()
    chunks = []
    current_chunk = []
    word_count = 0

    for i, word in enumerate(words):
        current_chunk.append(word)
        word_count += 1

        # Estimate duration based on word length and punctuation
        duration_factor = 1.0
        if any(word.endswith(p) for p in ['.', '!', '?']):
            duration_factor = 1.5  # Longer pause for sentence endings
        elif any(word.endswith(p) for p in [',', ';', ':']):
            duration_factor = 1.2  # Slight pause for phrases

        # Calculate estimated duration (roughly 0.3 seconds per word)
        estimated_duration = len(
            ' '.join(current_chunk)) * 0.06 * duration_factor

        should_break = (any(
            word.endswith(p) for p in ['.', '!', '?', ',', ';', ':'])
                        or word_count >= max_words or i == len(words) - 1)

        if should_break:
            chunks.append({
                'text': ' '.join(current_chunk),
                'estimated_duration': estimated_duration
            })
            current_chunk = []
            word_count = 0

    return chunks


def create_explanation_video(
        explanation: str,
        progress_callback: callable = None) -> Tuple[str, Optional[str]]:
    """
    Create a video with synchronized audio from the explanation text.

    Args:
        explanation (str): Text to convert to video
        progress_callback (callable): Function to update progress bar

    Returns:
        Tuple[str, Optional[str]]: Path to video file and any error message
    """
    temp_dir = None
    try:
        # Update progress - Starting
        if progress_callback:
            progress_callback(0.1)

        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        video_path = str(Path(temp_dir) / "video.mp4")
        audio_path = str(Path(temp_dir) / "audio.wav")
        output_path = str(Path(temp_dir) / "final.mp4")

        # Get background video
        if progress_callback:
            progress_callback(0.2)

        background_video_url = "https://drive.google.com/file/d/1VYAqXn2b3ujHdcCDJ1y7KUP7AMy33VLG/view?usp=sharing"
        logger.info(f"Getting background video from: {background_video_url}")

        background_path, bg_error = get_background_video(background_video_url)
        if bg_error:
            raise Exception(f"Background video error: {bg_error}")

        # Verify the video file
        is_valid, error_msg = verify_video_file(background_path)
        if not is_valid:
            raise Exception(f"Invalid background video: {error_msg}")
        logger.info("Background video verified successfully")

        # Update progress - Background video downloaded
        if progress_callback:
            progress_callback(0.3)

        # Generate audio with timing information
        logger.info("Generating audio...")
        complete_audio, chunks = create_complete_audio(explanation,
                                                       output_path=audio_path)
        if complete_audio is None:
            raise Exception("Failed to create audio")

        # Verify audio file exists
        if not os.path.exists(audio_path):
            raise Exception("Audio file was not created successfully")

        # Get audio duration
        audio_duration = len(complete_audio) / 1000.0  # Convert to seconds
        logger.info(f"Audio duration: {audio_duration} seconds")

        # Update progress - Audio generated
        if progress_callback:
            progress_callback(0.5)

        # Create filter string for each chunk with precise timings
        filter_complex = []
        for chunk in chunks:
            safe_text = escape_text_for_ffmpeg(chunk['text'])
            filter_complex.append(
                f"drawtext=text='{safe_text}'"
                f":fontsize=80"
                f":fontcolor=white"
                f":box=1"
                f":boxcolor=black@0.5"
                f":boxborderw=5"
                f":x=(w-text_w)/2"
                f":y=(h-text_h)/2"
                f":enable='between(t,{chunk['start_time']},{chunk['start_time'] + chunk['duration']})'"
            )

        # Update progress - Text filters prepared
        if progress_callback:
            progress_callback(0.6)

        # Join all filters
        complete_filter = ','.join(filter_complex)

        # Get background video duration
        bg_duration, duration_error = get_video_duration(background_path)
        if duration_error:
            raise Exception(
                f"Error getting background duration: {duration_error}")

        # Calculate required duration from audio
        required_duration = len(complete_audio) / 1000.0  # Convert to seconds

        # Get random start time
        start_time = get_random_start_time(bg_duration, required_duration)
        logger.info(f"Starting background video at {start_time:.2f} seconds")

        # Modify FFmpeg command for better streaming performance
        try:
            run_ffmpeg_with_timing([
                'ffmpeg',
                '-y',
                '-ss',
                str(start_time),  # Add start time offset
                '-i',
                background_path,
                '-vf',
                complete_filter,
                '-c:v',
                'libx264',
                '-preset',
                'ultrafast',
                '-t',
                str(required_duration),
                video_path
            ])
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error output: {e.stderr}")
            raise Exception(f"FFmpeg command failed: {e.stderr}")

        # Verify video file exists
        if not os.path.exists(video_path):
            raise Exception("Video file was not created successfully")
        # Update progress - Video created
        if progress_callback:
            progress_callback(0.8)

        # Combine with audio - Modified command with explicit audio mapping
        logger.info("Combining video with audio...")
        try:
            subprocess.run(
                [
                    'ffmpeg',
                    '-y',
                    '-i',
                    video_path,  # First input: video
                    '-i',
                    audio_path,  # Second input: audio
                    '-c:v',
                    'copy',  # Copy video stream without re-encoding
                    '-c:a',
                    'aac',  # Use AAC codec for audio
                    '-strict',
                    'experimental',
                    '-map',
                    '0:v:0',  # Map video from first input
                    '-map',
                    '1:a:0',  # Map audio from second input
                    '-shortest',  # End when shortest input ends
                    output_path
                ],
                check=True,
                capture_output=True)

            # Add debug logging
            logger.info(
                f"Video file size: {os.path.getsize(video_path)} bytes")
            logger.info(
                f"Audio file size: {os.path.getsize(audio_path)} bytes")
            logger.info(
                f"Output file size: {os.path.getsize(output_path)} bytes")

            # Verify the output has both streams
            result = subprocess.run([
                'ffprobe', '-v', 'error', '-select_streams', 'a',
                '-show_entries', 'stream=codec_name', '-of',
                'default=noprint_wrappers=1:nokey=1', output_path
            ],
                                    capture_output=True,
                                    text=True)

            if not result.stdout.strip():
                raise Exception("No audio stream found in output file")

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error output: {e.stderr}")
            raise Exception(f"FFmpeg command failed: {e.stderr}")

        # Update progress - Final video created
        if progress_callback:
            progress_callback(0.9)

        # Move to permanent location
        permanent_dir = Path(tempfile.gettempdir()) / "streamlit_videos"
        permanent_dir.mkdir(exist_ok=True)
        permanent_path = str(permanent_dir /
                             f"explanation_{os.urandom(8).hex()}.mp4")
        shutil.copy2(output_path, permanent_path)

        # Update progress - Complete
        if progress_callback:
            progress_callback(1.0)

        # Verify audio file
        def verify_audio_file(audio_path: str) -> Tuple[bool, Optional[str]]:
            """Verify that the audio file is valid and contains audio data."""
            try:
                # Check file exists and has size
                if not os.path.exists(audio_path):
                    return False, "Audio file does not exist"

                if os.path.getsize(audio_path) == 0:
                    return False, "Audio file is empty"

                # Check audio properties using FFprobe
                result = subprocess.run([
                    'ffprobe', '-v', 'error', '-select_streams', 'a',
                    '-show_entries', 'stream=codec_name,duration', '-of',
                    'json', audio_path
                ],
                                        capture_output=True,
                                        text=True)

                if result.returncode != 0:
                    return False, "Failed to probe audio file"

                return True, None

            except Exception as e:
                return False, str(e)

        # Verify the generated audio
        is_valid, audio_error = verify_audio_file(audio_path)
        if not is_valid:
            raise Exception(f"Invalid audio file: {audio_error}")

        logger.info("Audio file verified successfully")

        # Add explicit file closing and verification
        if output_path and os.path.exists(output_path):
            # Verify the file is completely written
            try:
                with open(permanent_path, 'rb') as f:
                    f.seek(0, 2)  # Seek to end
                    file_size = f.tell()
                    if file_size == 0:
                        raise Exception("Output video file is empty")

                # Verify video can be opened
                cap = cv2.VideoCapture(permanent_path)
                if not cap.isOpened():
                    raise Exception("Cannot open output video")
                cap.release()

                logger.info(
                    f"Video file verified: {permanent_path} ({file_size} bytes)"
                )
            except Exception as e:
                raise Exception(f"Video verification failed: {str(e)}")

        return permanent_path, None

    except Exception as e:
        error_msg = f"Error creating video: {str(e)}"
        logger.error(error_msg)
        return "", error_msg
    finally:
        if temp_dir and Path(temp_dir).exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.error(
                    f"Error cleaning up temporary directory: {str(e)}")


def create_audio_chunk(text: str,
                       language: str = 'en') -> Optional[AudioSegment]:
    """
    Create an audio segment from text using Google Text-to-Speech.

    Args:
        text (str): Text to convert to speech
        language (str): Language code for speech generation

    Returns:
        Optional[AudioSegment]: Audio segment or None if creation fails
    """
    mp3_file = None
    wav_file = None
    try:
        logger.info(f"Creating audio for text: '{text}'")

        # Create temporary files
        temp_dir = tempfile.gettempdir()
        mp3_file = os.path.join(temp_dir, f'speech_{os.urandom(8).hex()}.mp3')
        wav_file = os.path.join(temp_dir, f'speech_{os.urandom(8).hex()}.wav')

        # Create gTTS object and save to MP3
        tts = gTTS(text=text.strip(), lang=language, slow=False)
        tts.save(mp3_file)

        # Verify MP3 file exists and has content
        if not os.path.exists(mp3_file):
            raise Exception("Failed to create MP3 file")

        file_size = os.path.getsize(mp3_file)
        if file_size == 0:
            raise Exception("Created MP3 file is empty")

        logger.info(f"Created MP3 file of size {file_size} bytes")

        # Convert MP3 to WAV using ffmpeg
        subprocess.run([
            'ffmpeg', '-y', '-i', mp3_file, '-acodec', 'pcm_s16le', '-ac', '2',
            '-ar', '44100', wav_file
        ],
                       check=True,
                       capture_output=True)

        # Load the WAV file
        audio_segment = AudioSegment.from_wav(wav_file)

        if len(audio_segment) == 0:
            raise Exception("Created audio segment is empty")

        # Speed up the audio by 30%
        audio_segment = audio_segment.speedup(playback_speed=1.3)

        logger.info(
            f"Successfully created audio segment of length {len(audio_segment)}ms"
        )

        return audio_segment

    except Exception as e:
        logger.error(f"Error creating audio for text '{text}': {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

    finally:
        # Clean up temporary files
        for temp_file in [mp3_file, wav_file]:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    logger.info(f"Cleaned up temporary file: {temp_file}")
                except Exception as e:
                    logger.error(f"Error removing temporary file: {str(e)}")


def create_complete_audio(
        text: str,
        output_path: str = None) -> Tuple[Optional[AudioSegment], List[dict]]:
    """
    Create audio and calculate precise word timings.

    Args:
        text (str): Complete text to convert to speech
        output_path (str): Path to save the final audio file

    Returns:
        Tuple[Optional[AudioSegment], List[dict]]: Audio segment and timing information
    """
    try:
        # Create temporary files
        temp_dir = tempfile.gettempdir()
        mp3_file = os.path.join(temp_dir, f'speech_{os.urandom(8).hex()}.mp3')
        wav_file = os.path.join(temp_dir, f'speech_{os.urandom(8).hex()}.wav')

        # Split text into chunks with estimated durations
        chunks = split_into_chunks(text)

        # Create complete audio
        tts = gTTS(text=text.strip(), lang='en', slow=False)
        tts.save(mp3_file)

        # Convert to WAV
        subprocess.run([
            'ffmpeg', '-y', '-i', mp3_file, '-acodec', 'pcm_s16le', '-ac', '2',
            '-ar', '44100', wav_file
        ],
                       check=True,
                       capture_output=True)

        # Load and speed up audio
        audio_segment = AudioSegment.from_wav(wav_file)
        audio_segment = audio_segment.speedup(playback_speed=1.3)

        # Calculate actual durations based on total audio length
        total_duration = len(audio_segment) / 1000.0  # Convert to seconds
        total_estimated = sum(chunk['estimated_duration'] for chunk in chunks)

        # Adjust timings proportionally
        current_time = 0
        for chunk in chunks:
            proportion = chunk['estimated_duration'] / total_estimated
            actual_duration = total_duration * proportion
            chunk['start_time'] = current_time
            chunk['duration'] = actual_duration
            current_time += actual_duration

        # Save the final audio with explicit format settings
        if output_path:
            audio_segment = audio_segment.set_frame_rate(44100)
            audio_segment = audio_segment.set_channels(2)
            audio_segment = audio_segment.set_sample_width(2)  # 16-bit

            # Export with specific format parameters
            audio_segment.export(output_path,
                                 format="wav",
                                 parameters=[
                                     "-acodec", "pcm_s16le", "-ac", "2", "-ar",
                                     "44100"
                                 ])

            # Verify exported file
            if not os.path.exists(output_path):
                raise Exception("Failed to export audio file")

            logger.info(f"Audio exported successfully: {output_path}")

        return audio_segment, chunks

    except Exception as e:
        logger.error(f"Error creating audio: {str(e)}")
        return None, []


def draw_text_with_stroke(draw,
                          text: str,
                          x: int,
                          y: int,
                          font,
                          text_color: str = "white",
                          stroke_color: str = "black",
                          stroke_width: int = 3) -> None:
    """
    Draw text with a stroke effect.

    Args:
        draw: ImageDraw object
        text (str): Text to draw
        x (int): X coordinate
        y (int): Y coordinate
        font: ImageFont object
        text_color (str): Color of the main text
        stroke_color (str): Color of the stroke
        stroke_width (int): Width of the stroke
    """
    # Draw stroke by offsetting text in all directions
    for offset_x in range(-stroke_width, stroke_width + 1):
        for offset_y in range(-stroke_width, stroke_width + 1):
            draw.text((x + offset_x, y + offset_y),
                      text,
                      font=font,
                      fill=stroke_color)

    # Draw the main text
    draw.text((x, y), text, font=font, fill=text_color)


def create_frame_with_background(text: str,
                                 background_frame: np.ndarray,
                                 size: Tuple[int, int],
                                 initial_font_size: int = 100) -> np.ndarray:
    """
    Create a frame with text overlaid on background video frame.
    """
    # Resize background frame to match target size
    background_resized = cv2.resize(background_frame, size)

    # Convert to PIL Image for text rendering
    pil_img = Image.fromarray(
        cv2.cvtColor(background_resized, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Calculate margin (10% of width)
    margin = int(size[0] * 0.1)
    max_text_width = size[0] - (2 * margin)

    # Get font
    def get_cached_font(font_size):
        return get_font(font_size, platform.system())

    # Font size adjustment
    font_size = initial_font_size
    font = get_cached_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]

    while text_width > max_text_width and font_size > 20:
        font_size -= 5
        font = get_cached_font(font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]

    # Calculate text position
    text_height = bbox[3] - bbox[1]
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2

    # Draw text with stroke effect
    draw_text_with_stroke(draw, text, x, y, font)

    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def escape_text_for_ffmpeg(text: str) -> str:
    """
    Escape text for use in FFmpeg filter strings.

    Args:
        text (str): Text to escape

    Returns:
        str: Escaped text safe for FFmpeg
    """
    # Replace backslashes first
    text = text.replace('\\', '\\\\')
    # Replace single quotes
    text = text.replace("'", "'\\\\\\''")
    # Replace colons
    text = text.replace(':', '\\:')
    # Replace commas
    text = text.replace(',', '\\,')
    return text


def verify_video_file(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Verify that a video file is valid and can be processed by FFmpeg.
    """
    try:
        # Check file exists and is readable
        if not os.path.exists(file_path):
            return False, "Video file does not exist"
        if not os.access(file_path, os.R_OK):
            return False, "Video file is not readable"

        # Try to open with OpenCV first
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return False, "OpenCV cannot open video file"
        cap.release()

        # Check with FFprobe
        result = subprocess.run([
            'ffprobe', '-v', 'warning', '-show_format', '-show_streams',
            file_path
        ],
                                capture_output=True,
                                text=True)

        if result.returncode != 0:
            return False, f"FFprobe error: {result.stderr}"

        if 'codec_type=video' not in result.stdout:
            return False, "No video stream found"

        return True, None

    except Exception as e:
        return False, f"Error verifying video: {str(e)}"


def get_video_duration(video_path: str) -> Tuple[float, Optional[str]]:
    """
    Get the duration of a video file using FFprobe.

    Args:
        video_path (str): Path to the video file

    Returns:
        Tuple[float, Optional[str]]: Duration in seconds and any error message
    """
    try:
        # Use a more reliable FFprobe command
        cmd = [
            'ffprobe', '-i', video_path, '-show_entries', 'format=duration',
            '-v', 'quiet', '-of', 'csv=p=0'
        ]

        result = subprocess.run(cmd,
                                capture_output=True,
                                text=True,
                                check=True)
        duration = float(result.stdout.strip())

        if duration <= 0:
            return 0, "Invalid video duration"

        return duration, None

    except subprocess.CalledProcessError as e:
        return 0, f"FFprobe error: {e.stderr}"
    except Exception as e:
        return 0, f"Error getting duration: {str(e)}"


def get_video_download_link(video_path: str) -> str:
    """
    Generate a download link for the video file.

    Args:
        video_path (str): Path to the video file

    Returns:
        str: HTML string containing the download link
    """
    try:
        with open(video_path, 'rb') as f:
            video_bytes = f.read()

        b64_video = base64.b64encode(video_bytes).decode()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        video_filename = f"brainrot-{timestamp}.mp4"

        href = f'<a href="data:video/mp4;base64,{b64_video}" download="{video_filename}">Download Video</a>'
        return href
    except Exception as e:
        logger.error(f"Error creating download link: {str(e)}")
        return ""


def get_random_start_time(video_duration: float,
                          required_duration: float) -> float:
    """
    Calculate a random start time for the background video.

    Args:
        video_duration (float): Total duration of background video in seconds
        required_duration (float): Duration needed for the output video in seconds

    Returns:
        float: Random start time in seconds
    """
    if video_duration <= required_duration:
        return 0.0

    # Leave some buffer at the end
    max_start = video_duration - required_duration - 1
    if max_start <= 0:
        return 0.0

    return random.uniform(0, max_start)


def run_ffmpeg_with_timing(command: List[str]) -> None:
    """
    Run FFmpeg command with performance monitoring and verification.
    """
    start_time = time.time()
    try:
        # Run FFmpeg command
        result = subprocess.run(command,
                                check=True,
                                capture_output=True,
                                text=True)

        # Get output file path (last argument in command)
        output_path = command[-1]

        # Verify output file
        if not os.path.exists(output_path):
            raise Exception("FFmpeg did not create output file")

        # Verify file size
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            raise Exception("FFmpeg created empty output file")

        duration = time.time() - start_time
        logger.info(
            f"FFmpeg encoding completed in {duration:.2f} seconds. Output size: {file_size} bytes"
        )

        # Verify the output file is a valid video
        is_valid, error = verify_video_playability(output_path)
        if not is_valid:
            raise Exception(f"FFmpeg output validation failed: {error}")

        return result

    except subprocess.CalledProcessError as e:
        logger.error(
            f"FFmpeg failed after {time.time() - start_time:.2f} seconds")
        logger.error(f"Error output: {e.stderr}")
        raise


def verify_video_playability(video_path: str) -> Tuple[bool, Optional[str]]:
    """
    Verify that a video file is complete and playable.

    Args:
        video_path (str): Path to video file

    Returns:
        Tuple[bool, Optional[str]]: Success status and error message if any
    """
    try:
        # Check file exists and has size
        if not os.path.exists(video_path):
            return False, "Video file does not exist"

        file_size = os.path.getsize(video_path)
        if file_size == 0:
            return False, "Video file is empty"

        # Check video metadata with FFprobe
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries',
            'stream=codec_type,codec_name,width,height,duration', '-of',
            'json', video_path
        ],
                                capture_output=True,
                                text=True)

        if result.returncode != 0:
            return False, f"FFprobe error: {result.stderr}"

        # Verify with OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Cannot open video with OpenCV"

        # Try to read first and last frame
        ret, first_frame = cap.read()
        if not ret:
            return False, "Cannot read first frame"

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, last_frame = cap.read()
        if not ret:
            return False, "Cannot read last frame"

        cap.release()

        logger.info(
            f"Video verified successfully: {video_path} ({file_size} bytes)")
        return True, None

    except Exception as e:
        return False, f"Video verification error: {str(e)}"
