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


# 確保 download 資料夾存在
DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'download')
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# 設定可用的模型大小
AVAILABLE_MODELS = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large": "large",
    "turbo": "turbo"
}

# 設定支援的語言
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

# 改為全域變數，但不立即載入模型
current_model = "turbo"
model = None

def load_model_if_needed(model_name):
    """根據需要載入模型"""
    global model, current_model
    if model is None or current_model != model_name:
        current_model = model_name
        model = whisper.load_model(model_name)
    return model

def format_timestamp(seconds):
    """將秒數轉換為 VTT/SRT 時間戳格式"""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def generate_vtt(segments):
    """生成 VTT 格式的字幕"""
    vtt_lines = ["WEBVTT\n"]
    for i, segment in enumerate(segments, 1):
        start = format_timestamp(segment['start'])
        end = format_timestamp(segment['end'])
        vtt_lines.append(f"\n{start} --> {end}")
        vtt_lines.append(segment['text'].strip())
    return "\n".join(vtt_lines)

def generate_srt(segments):
    """生成 SRT 格式的字幕"""
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
    """生成 TSV 格式的轉錄"""
    tsv_lines = ["start\tend\ttext"]
    for segment in segments:
        start = format_timestamp(segment['start'])
        end = format_timestamp(segment['end'])
        text = segment['text'].strip().replace('\t', ' ')
        tsv_lines.append(f"{start}\t{end}\t{text}")
    return "\n".join(tsv_lines)

def get_base_filename(original_filename):
    """從原始檔名獲取基礎檔名（不含副檔名）"""
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

    # 獲取進階參數
    data = request.form
    temperature = float(data.get('temperature', 0.0))
    no_speech_threshold = float(data.get('no_speech_threshold', 0.8))
    hallucination_silence_threshold = float(data.get('hallucination_silence_threshold', 2.0))
    word_timestamps = data.get('word_timestamps', 'true').lower() == 'true'
    initial_prompt = data.get('initial_prompt', None)
    language = data.get('language', 'auto')  # 新增語言參數

    base_filename = get_base_filename(file.filename)

    try:
        # 在實際需要時才載入模型
        model = load_model_if_needed(current_model)
        print(f"已載入模型: {current_model}")  # 調試資訊

        # 創建臨時文件來保存上傳的音頻
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            file.save(temp_file.name)
            # 使用 Whisper 進行轉錄，加入可控制的參數
            result = model.transcribe(
                temp_file.name,
                temperature=temperature,
                verbose=True,
                no_speech_threshold=no_speech_threshold,
                word_timestamps=word_timestamps,
                hallucination_silence_threshold=hallucination_silence_threshold,
                initial_prompt=initial_prompt if initial_prompt else None,
                language=None if language == 'auto' else language,  # 設定語言參數
                condition_on_previous_text=False
            )
            
            # 刪除臨時文件
            os.unlink(temp_file.name)
        
            # 準備不同格式的輸出
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
                'detected_language': result.get('language', 'unknown')  # 返回檢測到的語言
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
    
    # 創建臨時文件
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=f'.{format_type}', encoding='utf-8') as temp_file:
        if format_type == 'json':
            json.dump(content, temp_file, ensure_ascii=False, indent=2)
        else:
            temp_file.write(content)
    
    # 發送文件
    return send_file(
        temp_file.name,
        as_attachment=True,
        download_name=f'{filename}.{format_type}',
        mimetype='text/plain'
    )

def get_podcast_info(url):
    """從 Apple Podcasts URL 獲取播客資訊"""
    try:
        # 從 URL 提取播客 ID 和集數 ID
        podcast_id = re.search(r'id(\d+)', url)
        episode_id = re.search(r'i=(\d+)', url)
        
        if not podcast_id or not episode_id:
            raise Exception("無效的 Apple Podcasts URL")
            
        # 首先獲取播客的所有集數資訊
        api_url = f"https://itunes.apple.com/lookup?id={podcast_id.group(1)}&entity=podcastEpisode&limit=200"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        print(f"正在獲取播客資訊: {api_url}")  # 調試資訊
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        if not data.get('results'):
            raise Exception("找不到播客資訊")
            
        # 在所有集數中尋找目標集數
        target_episode_id = episode_id.group(1)
        episode_info = None
        
        print(f"正在搜尋集數 ID: {target_episode_id}")  # 調試資訊
        for result in data['results']:
            if str(result.get('trackId')) == target_episode_id:
                episode_info = result
                break
                
        if not episode_info:
            raise Exception(f"找不到指定的集數 (ID: {target_episode_id})")
            
        print(f"找到集數資訊: {episode_info.get('trackName')}")  # 調試資訊
        return {
            'podcast_name': episode_info.get('collectionName', ''),
            'episode_name': episode_info.get('trackName', ''),
            'episode_url': episode_info.get('episodeUrl', ''),
            'duration': episode_info.get('trackTimeMillis', 0) / 1000  # 轉換為秒
        }
    except Exception as e:
        print(f"獲取播客資訊時發生錯誤: {str(e)}")  # 調試資訊
        raise Exception(f"獲取播客資訊失敗: {str(e)}")

def download_episode(episode_info):
    """下載播客單集"""
    try:
        if not episode_info.get('episode_url'):
            raise Exception("找不到音訊檔案連結")
            
        # 準備檔案名稱
        podcast_name = re.sub(r'[<>:"/\\|?*]', '_', episode_info['podcast_name'])
        episode_name = re.sub(r'[<>:"/\\|?*]', '_', episode_info['episode_name'])
        filename = f"{podcast_name} - {episode_name}.mp3"
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        
        print(f"準備下載到: {filepath}")  # 調試資訊
        print(f"音訊 URL: {episode_info['episode_url']}")  # 調試資訊
        
        # 使用 yt-dlp 下載音訊
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': filepath,
            'quiet': False,  # 顯示下載進度
            'no_warnings': False,  # 顯示警告
            'verbose': True,  # 顯示詳細資訊
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([episode_info['episode_url']])
                    
        if not os.path.exists(filepath):
            raise Exception("下載完成但找不到檔案")
            
        print(f"下載完成: {filepath}")  # 調試資訊
        return {
            'filename': filename,
            'title': f"{episode_info['podcast_name']} - {episode_info['episode_name']}",
            'duration': episode_info['duration']
        }
    except Exception as e:
        print(f"下載音訊檔案時發生錯誤: {str(e)}")  # 調試資訊
        raise Exception(f"下載音訊檔案失敗: {str(e)}")

@app.route('/download-podcast', methods=['POST'])
def download_podcast():
    if not request.is_json:
        return jsonify({'error': 'Invalid request'}), 400
    
    data = request.get_json()
    url = data.get('url')
    
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    try:
        print(f"收到下載請求: {url}")  # 調試資訊
        # 檢查是否為 Apple Podcasts URL
        if 'podcasts.apple.com' in url:
            # 獲取播客資訊
            podcast_info = get_podcast_info(url)
            print(f"獲取到播客資訊: {podcast_info}")  # 調試資訊
            # 下載音訊
            result = download_episode(podcast_info)
            return jsonify({
                'success': True,
                'filename': result['filename'],
                'title': result['title'],
                'duration': result['duration']
            })
        else:
            raise Exception("目前只支援 Apple Podcasts 連結")
            
    except Exception as e:
        print(f"處理下載請求時發生錯誤: {str(e)}")  # 調試資訊
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