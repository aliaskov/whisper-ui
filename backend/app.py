from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import whisper
import os
import tempfile
import json
from datetime import timedelta
import requests
import yt_dlp
import re
import xml.etree.ElementTree as ET

app = Flask(__name__)
CORS(app)

# Ensure the 'download' folder exists
DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'download')
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Define available model sizes
AVAILABLE_MODELS = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large": "large",
    "turbo": "turbo"
}

# Define supported languages
AVAILABLE_LANGUAGES = {
    "auto": "Auto Detection",
    "ar": "Arabic",
    "zh": "Chinese",
    "nl": "Dutch",
    "en": "English",
    "fr": "French",
    "de": "German",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese"
}

# Global variables, but do not load the model immediately
current_model = "turbo"
model = None


def load_model_if_needed(model_name):
    """Load the model when needed"""
    global model, current_model
    if model is None or current_model != model_name:
        current_model = model_name
        # Explicitly specify to use CPU (FP32)
        model = whisper.load_model(model_name, device="cpu")
    return model


def format_timestamp(seconds):
    """Convert seconds to VTT/SRT timestamp format"""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def generate_vtt(segments):
    """Generate VTT formatted subtitles"""
    vtt_lines = ["WEBVTT\n"]
    for i, segment in enumerate(segments, 1):
        start = format_timestamp(segment['start'])
        end = format_timestamp(segment['end'])
        vtt_lines.append(f"\n{start} --> {end}")
        vtt_lines.append(segment['text'].strip())
    return "\n".join(vtt_lines)


def generate_srt(segments):
    """Generate SRT formatted subtitles"""
    srt_lines = []
    for i, segment in enumerate(segments, 1):
        start = format_timestamp(segment['start']).replace('.', ',')
        end = format_timestamp(segment['end']).replace('.', ',')
        srt_lines.append(str(i))
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(segment['text'].strip())
        srt_lines.append("")
    return "\n".join(srt_lines)


def generate_tsv(segments):
    """Generate TSV formatted transcription"""
    tsv_lines = ["start\tend\ttext"]
    for segment in segments:
        start = format_timestamp(segment['start'])
        end = format_timestamp(segment['end'])
        text = segment['text'].strip().replace('\t', ' ')
        tsv_lines.append(f"{start}\t{end}\t{text}")
    return "\n".join(tsv_lines)


def get_base_filename(original_filename):
    """Extract the base filename (without extension) from the original filename"""
    return os.path.splitext(original_filename)[0]


@app.route('/models', methods=['GET'])
def get_models():
    return jsonify(list(AVAILABLE_MODELS.keys()))


@app.route('/model', methods=['POST'])
def change_model():
    global current_model
    new_model = request.json.get('model')
    if new_model not in AVAILABLE_MODELS:
        return jsonify({'error': 'Invalid model'}), 400
    current_model = new_model
    return jsonify({'success': True, 'current_model': current_model})


@app.route('/languages', methods=['GET'])
def get_languages():
    """Return list of available languages"""
    return jsonify(AVAILABLE_LANGUAGES)


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Get advanced parameters
    data = request.form
    temperature = float(data.get('temperature', 0.0))
    no_speech_threshold = float(data.get('no_speech_threshold', 0.8))
    hallucination_silence_threshold = float(data.get('hallucination_silence_threshold', 2.0))
    word_timestamps = data.get('word_timestamps', 'true').lower() == 'true'
    initial_prompt = data.get('initial_prompt', None)
    language = data.get('language', 'auto')  # Added language parameter

    base_filename = get_base_filename(file.filename)

    try:
        # Load the model only when needed
        model = load_model_if_needed(current_model)
        print(f"Model loaded: {current_model}")  # Debug info

        # Create a temporary file to save the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            file.save(temp_file.name)
            # Use Whisper for transcription with configurable parameters
            result = model.transcribe(
                temp_file.name,
                temperature=temperature,
                verbose=True,
                no_speech_threshold=no_speech_threshold,
                word_timestamps=word_timestamps,
                hallucination_silence_threshold=hallucination_silence_threshold,
                initial_prompt=initial_prompt if initial_prompt else None,
                language=None if language == 'auto' else language,  # Set language parameter
                condition_on_previous_text=False
            )

            # Delete the temporary file
            os.unlink(temp_file.name)

            # Prepare outputs in various formats
            formats = {
                'txt': result['text'],
                'json': result,
                'vtt': generate_vtt(result['segments']),
                'srt': generate_srt(result['segments']),
                'tsv': generate_tsv(result['segments'])
            }

            return jsonify({
                'text': result['text'],
                'segments': result['segments'],
                'formats': formats,
                'filename': base_filename,
                'detected_language': result.get('language', 'unknown')  # Return detected language
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download/<format_type>', methods=['POST'])
def download_transcription(format_type):
    if not request.is_json:
        return jsonify({'error': 'Invalid request'}), 400

    data = request.get_json()
    content = data.get('content')
    filename = data.get('filename', 'transcription')

    if not content:
        return jsonify({'error': 'No content provided'}), 400

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=f'.{format_type}', encoding='utf-8') as temp_file:
        if format_type == 'json':
            json.dump(content, temp_file, ensure_ascii=False, indent=2)
        else:
            temp_file.write(content)

    # Send the file
    return send_file(
        temp_file.name,
        as_attachment=True,
        download_name=f'{filename}.{format_type}',
        mimetype='text/plain'
    )


def get_podcast_info(url):
    """Fetch podcast information from an Apple Podcasts URL"""
    try:
        # Extract podcast ID and episode ID from the URL
        podcast_id = re.search(r'id(\d+)', url)
        episode_id = re.search(r'i=(\d+)', url)

        if not podcast_id or not episode_id:
            raise Exception("Invalid Apple Podcasts URL")

        # First, retrieve all episodes of the podcast
        api_url = f"https://itunes.apple.com/lookup?id={podcast_id.group(1)}&entity=podcastEpisode&limit=200"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }

        print(f"Fetching podcast info: {api_url}")  # Debug info
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        data = response.json()

        if not data.get('results'):
            raise Exception("Podcast information not found")

        # Search for the target episode among all episodes
        target_episode_id = episode_id.group(1)
        episode_info = None

        print(f"Searching for episode ID: {target_episode_id}")  # Debug info
        for result in data['results']:
            if str(result.get('trackId')) == target_episode_id:
                episode_info = result
                break

        if not episode_info:
            raise Exception(f"Specified episode not found (ID: {target_episode_id})")

        print(f"Found episode info: {episode_info.get('trackName')}")  # Debug info
        return {
            'podcast_name': episode_info.get('collectionName', ''),
            'episode_name': episode_info.get('trackName', ''),
            'episode_url': episode_info.get('episodeUrl', ''),
            'duration': episode_info.get('trackTimeMillis', 0) / 1000  # Convert to seconds
        }
    except Exception as e:
        print(f"Error fetching podcast info: {str(e)}")  # Debug info
        raise Exception(f"Failed to fetch podcast info: {str(e)}")


def download_episode(episode_info):
    """Download a podcast episode"""
    try:
        if not episode_info.get('episode_url'):
            raise Exception("Audio file URL not found")

        # Prepare the filename
        podcast_name = re.sub(r'[<>:"/\\|?*]', '_', episode_info['podcast_name'])
        episode_name = re.sub(r'[<>:"/\\|?*]', '_', episode_info['episode_name'])
        filename = f"{podcast_name} - {episode_name}.mp3"
        filepath = os.path.join(DOWNLOAD_DIR, filename)

        print(f"Preparing to download to: {filepath}")  # Debug info
        print(f"Audio URL: {episode_info['episode_url']}")  # Debug info

        # Download the audio using yt-dlp
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': filepath,
            'quiet': False,  # Show download progress
            'no_warnings': False,  # Show warnings
            'verbose': True,  # Show detailed information
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([episode_info['episode_url']])

        if not os.path.exists(filepath):
            raise Exception("Download completed but file not found")

        print(f"Download completed: {filepath}")  # Debug info
        return {
            'filename': filename,
            'title': f"{episode_info['podcast_name']} - {episode_info['episode_name']}",
            'duration': episode_info['duration']
        }
    except Exception as e:
        print(f"Error downloading audio file: {str(e)}")  # Debug info
        raise Exception(f"Failed to download audio file: {str(e)}")


@app.route('/download-podcast', methods=['POST'])
def download_podcast():
    if not request.is_json:
        return jsonify({'error': 'Invalid request'}), 400

    data = request.get_json()
    url = data.get('url')

    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    try:
        print(f"Received download request: {url}")  # Debug info
        # Check if the URL is for Apple Podcasts
        if 'podcasts.apple.com' in url:
            # Fetch podcast information
            podcast_info = get_podcast_info(url)
            print(f"Podcast info obtained: {podcast_info}")  # Debug info
            # Download the audio
            result = download_episode(podcast_info)
            return jsonify({
                'success': True,
                'filename': result['filename'],
                'title': result['title'],
                'duration': result['duration']
            })
        else:
            raise Exception("Currently only Apple Podcasts links are supported")

    except Exception as e:
        print(f"Error processing download request: {str(e)}")  # Debug info
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>', methods=['GET'])
def get_downloaded_file(filename):
    try:
        file_path = os.path.join(DOWNLOAD_DIR, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404

        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='audio/mpeg'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5510)