from flask import Flask, request, send_file, jsonify
from flask_restx import Api, Resource, fields
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import torch
import io
import requests
import time
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
api = Api(app, version='1.0', title='TTS API', description='A simple TTS API')

ns = api.namespace('tts', description='TTS operations')

default_prompt = "Hello, how can I assist you today?"
default_description = "A neutral English female voice."

ADDRESS = 'localhost'
PORT = 7865
MAX_RETRIES = 5
RETRY_DELAY = 2  # seconds

tts_model = api.model('TTSRequest', {
    'prompt': fields.String(required=True, description='The text to be spoken', default="Hey, how are you doing today?"),
    'description': fields.String(required=True, description='Description of the speaker\'s voice', default="A neutral English female voice.")
})

# Load model and tokenizer
models = [
    ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-jenny-30H").to('cuda:0'),
    ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-jenny-30H").to('cuda:1')
]
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-jenny-30H")


def generate_audio(description, prompt, model, device_id):
    for attempt in range(MAX_RETRIES):
        try:
            input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device_id)
            prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device_id)

            generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
            logging.debug(f"Generation result (attempt {attempt + 1}, device {device_id}): {generation}, type: {type(generation)}")
            audio_arr = generation.cpu().numpy().squeeze()

            logging.debug(f"Generated audio array shape (attempt {attempt + 1}, device {device_id}): {audio_arr.shape}")

            if len(audio_arr.shape) == 1:
                audio_arr = audio_arr[:, None]  # Add a new axis to make it (samples, 1)

            if audio_arr.size == 0 or len(audio_arr.shape) != 2:
                logging.error(f"Generated audio array is empty or invalid (attempt {attempt + 1}, device {device_id}). Retrying...")
                time.sleep(RETRY_DELAY)
                continue

            buffer = io.BytesIO()
            sf.write(buffer, audio_arr, model.config.sampling_rate, format='WAV')
            buffer.seek(0)
            return buffer
        except Exception as e:
            logging.error(f"Error during TTS generation or saving (attempt {attempt + 1}, device {device_id}): {e}")
            logging.error("Full stack trace:", exc_info=True)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            raise e

@ns.route('/generate')
class TTSGenerate(Resource):
    @ns.expect(tts_model)
    def post(self):
        data = request.json
        prompt = data.get('prompt', default_prompt)
        description = data.get('description', default_description)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(generate_audio, description, prompt, models[i], f'cuda:{i}') for i in range(2)]
            for future in as_completed(futures):
                try:
                    buffer = future.result()
                    break
                except Exception as e:
                    logging.error(f"Generation failed on one GPU: {e}")
                    continue
            else:
                return jsonify({'error': 'Failed to generate audio on both GPUs'}), 500

        try:
            tts_wav_path = '/mnt/ml1_data/untitled.wav'
            with open(tts_wav_path, 'wb') as f:
                f.write(buffer.read())

            rvc_response = requests.post(
                f'http://{ADDRESS}:{PORT}/queue/join?',
                headers={'Content-Type': 'application/json'},
                json={
                    "data": [0, tts_wav_path, -2, None, "rmvpe", "", "logs/kobold/added_IVF94_Flat_nprobe_1_kobold_v2.index", 0.75, 3, 0, 0.25, 0.33],
                    "event_data": None,
                    "fn_index": 2,
                    "trigger_id": 33,
                    "session_hash": "4ivgfr7m5rr"
                }
            )

            if rvc_response.status_code != 200:
                return jsonify({'error': 'Failed to process with RVC'}), 500

            session_hash = "4ivgfr7m5rr"
            result_url = None
            for _ in range(10):
                status_response = requests.get(f'http://{ADDRESS}:{PORT}/queue/data?session_hash={session_hash}',
                                               headers={'Accept': 'text/event-stream'})
                for line in status_response.iter_lines():
                    if line:
                        data = line.decode('utf-8').replace('data: ', '')
                        event_data = json.loads(data)
                        if event_data.get('msg') == 'process_completed':
                            result_url = event_data['output']['data'][1]['url']
                            break
                if result_url:
                    break
                time.sleep(2)

            if not result_url:
                return jsonify({'error': 'Failed to retrieve processed file from RVC'}), 500

            final_wav_response = requests.get(result_url)
            if final_wav_response.status_code != 200:
                return jsonify({'error': 'Failed to download the processed WAV file'}), 500

            final_buffer = io.BytesIO(final_wav_response.content)
            return send_file(final_buffer, mimetype='audio/wav', as_attachment=True, download_name='processed_output.wav')

        except Exception as e:
            logging.error(f"Error during RVC processing or file handling: {e}")
            logging.error("Full stack trace:", exc_info=True)
            return jsonify({'error': f"Error during RVC processing or file handling: {e.__class__.__name__}, {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

