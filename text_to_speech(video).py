from gtts import gTTS
from moviepy.editor import (TextClip, AudioFileClip, concatenate_videoclips,
                            CompositeVideoClip)
import io
from pydub import AudioSegment
import numpy as np
import tempfile
import os
import logging
from concurrent.futures import ThreadPoolExecutor
import mmap
import hashlib

# Configure moviepy to use ImageMagick
os.environ["IMAGEMAGICK_BINARY"] = "/usr/local/bin/convert"  # Adjust the path if needed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory for caching audio files
CACHE_DIR = 'audio_cache'
os.makedirs(CACHE_DIR, exist_ok=True)


def hash_text(text):
    """Generate a unique hash for the given text."""
    return hashlib.sha256(text.encode()).hexdigest()


def text_to_audio(text, lang='en', tld='co.uk'):
    """Convert text to audio, with caching."""
    audio_hash = hash_text(text)
    cache_path = os.path.join(CACHE_DIR, f"{audio_hash}.wav")

    if os.path.exists(cache_path):
        logging.info("Using cached audio.")
        return AudioSegment.from_file(cache_path, format="wav")

    logging.info("Starting text to audio conversion.")
    tts = gTTS(text=text, lang=lang, tld=tld)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    audio_data = AudioSegment.from_file(mp3_fp, format="mp3")
    logging.info("Text to audio conversion completed.")

    # Save to cache
    audio_data.export(cache_path, format="wav")
    return audio_data


def make_text_clip_batch(start_time, end_time, duration, text):
    """Generate a text clip for a batch of frames."""
    start_index = int(start_time / duration * len(text))
    end_index = int(end_time / duration * len(text))
    highlighted_text = text[start_index:end_index]
    remaining_text = text[end_index:]

    txt_clip = TextClip(
        f"{highlighted_text}{remaining_text}", fontsize=24, color='white',
        size=(800, 400), method='caption', align='West'
    )
    txt_clip = txt_clip.set_position("center").set_duration(end_time - start_time).on_color(
        color=(0, 0, 0), col_opacity=0.8
    )
    return txt_clip


def create_video(text_file_path, lang='en', tld='co.uk', output_file='output_video.mp4'):
    """Create the final video from text."""
    logging.info("Reading text from file.")
    with open(text_file_path, 'r') as file:
        text = file.read()

    logging.info("Starting video creation.")
    audio_data = text_to_audio(text, lang=lang, tld=tld)

    audio_file_path = os.path.join(tempfile.gettempdir(), 'audio.wav')
    audio_data.export(audio_file_path, format="wav")

    audio = AudioFileClip(audio_file_path)
    duration = audio.duration
    logging.info(f"Audio duration: {duration} seconds.")

    text_clips = []
    batch_size = 5  # Number of text clips to process in each batch
    batch_intervals = np.arange(0, duration, batch_size)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for i in range(len(batch_intervals) - 1):
            start_time = batch_intervals[i]
            end_time = min(batch_intervals[i + 1], duration)
            futures.append(executor.submit(make_text_clip_batch, start_time, end_time, duration, text))
        for future in futures:
            text_clips.append(future.result())

    concatenated_text_clips = concatenate_videoclips(text_clips, method="compose")

    final_clip = CompositeVideoClip([concatenated_text_clips])
    final_clip = final_clip.set_audio(audio)
    logging.info("Writing final video file.")
    final_clip.write_videofile(output_file, fps=24)
    logging.info("Video creation completed.")

    os.remove(audio_file_path)


text_file_path = "input_text.txt"  # Path to the text file containing the text to be converted to speech
output_video_file = "text_to_speech_video.mp4"  # Output video file path

logging.info("Starting text to speech video creation process.")
create_video(text_file_path, output_file=output_video_file)
logging.info("Text to speech video creation process completed.")
