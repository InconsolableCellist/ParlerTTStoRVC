# 🗣️ TTS-RVC-Server

**TTS-RVC-Server** is a Python application that generates text-to-speech (TTS) audio using [Parler TTS](https://github.com/huggingface/parler-tts) and imitates a voice using the RVC endpoint. This version of the script is configured for two CUDA GPUs.

## 🛠️ Configuration

Your server is configurable via the script parameters and environment variables. 

**Configurable Parameters**:

- **address**: The address of the RVC server.
- **port**: The port of the RVC server.
- **max_retries**: Maximum number of retry attempts for TTS generation.
- **retry_delay**: Delay between retries in seconds.

## 🛰️ API Endpoints

### POST /tts/generate

Description: Generates TTS audio and processes it through the RVC endpoint.

Request:

```json
{
    "prompt": "I eat shinies.",
    "description": "A kobold."
}
```

Response: Returns a processed WAV file.
Example cURL Request

```sh

curl -X POST 'http://localhost:5000/tts/generate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "I eat shinies",
  "description": "A kobold"
}'
```

## 📅 Version History

0.1 - Initial check in

## ⚖️ License

This project is licensed under the MIT License. See the LICENSE file for details.
LICENSE

```sql

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
