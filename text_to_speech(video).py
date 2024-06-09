import os
import tempfile
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from moviepy.editor import TextClip, CompositeVideoClip, AudioFileClip


def text_to_audio(text, lang='en', tld='co.uk'):
    """Convert text to audio using gTTS."""
    from gtts import gTTS
    tts = gTTS(text=text, lang=lang, tld=tld)
    return tts


def make_text_clip_batch(start_time, end_time, total_duration, text):
    """Create text clips for a batch."""
    text_clips = []
    for i, line in enumerate(text.split('\n')):
        line_duration = end_time - start_time
        text_clip = TextClip(line, fontsize=70, color='white', bg_color='black', size=(1920, None))
        text_clip = text_clip.set_duration(line_duration).set_position(('center', 'center'))
        text_clips.append(text_clip)
    return text_clips


def create_video(text_file_path, lang='en', tld='co.uk', output_file='output_video.mp4'):
    """Create the final video from text."""
    os.environ["IMAGEMAGICK_BINARY"] = "/usr/local/bin/convert"  # Adjust the path if needed

    logging.info("Reading text from file.")
    with open(text_file_path, 'r') as file:
        text = file.read()

    logging.info("Starting video creation.")
    audio_data = text_to_audio(text, lang=lang, tld=tld)

    audio_file_path = os.path.join(tempfile.gettempdir(), 'audio.wav')
    audio_data.save(audio_file_path)

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
            text_clips.extend(future.result())

    concatenated_text_clips = CompositeVideoClip(text_clips, method="compose")

    final_clip = CompositeVideoClip([concatenated_text_clips])
    final_clip = final_clip.set_audio(audio)
    logging.info("Writing final video file.")

    # Specify video codec and format options
    final_clip.write_videofile(output_file, fps=24, codec="libx264", audio_codec="aac", threads=os.cpu_count(),
                               temp_audiofile="temp-audio.m4a", remove_temp=True, preset="ultrafast")
    
    logging.info("Video creation completed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    text_file_path = "input_text.txt" # Adjust the path if needed
    create_video(text_file_path)
