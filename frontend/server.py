"""
TrainTrain — Vanilla HTTP server replacing Gradio.

Serves the static frontend (index.html, app.js, styles.css) and exposes
REST API endpoints that call into the existing trainer modules.
"""

import os
import sys
import json
import io
import base64
import mimetypes
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, unquote

# Ensure the project root is on sys.path so we can import trainer modules
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Import trainer modules (lazy, after path is set) ───────────

from trainer import trainer, train
from trainer.train import train_named, stop_time

# Import scripts/traintrain to populate trainer.all_configs
import scripts.traintrain

# ── Paths ───────────────────────────────────────────────────────

FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_FILES = {
    '/': 'index.html',
    '/index.html': 'index.html',
    '/app.js': 'app.js',
    '/styles.css': 'styles.css',
}

# The API base URL injected into the frontend so JS knows where to send requests.
# By default, we use the same origin (empty string = relative URLs).
# Override via TRAINTRAIN_API_BASE env var if frontend is served from a different origin.
_API_BASE = os.environ.get('TRAINTRAIN_API_BASE', '')

# ── MIME types ──────────────────────────────────────────────────

mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('text/css', '.css')

# ── HTTP Handler ────────────────────────────────────────────────


class TrainTrainHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the TrainTrain frontend + API."""

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text, status=200):
        body = text.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html, status=200):
        body = html.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, path):
        rel_path = STATIC_FILES.get(path)
        if not rel_path:
            self._send_text('Not Found', 404)
            return
        file_path = os.path.join(FRONTEND_DIR, rel_path)
        if not os.path.isfile(file_path):
            self._send_text('Not Found', 404)
            return
        with open(file_path, 'rb') as f:
            content = f.read()
        content_type, _ = mimetypes.guess_type(file_path)
        if content_type is None:
            content_type = 'application/octet-stream'

        # For index.html, inject the API base URL as a script variable
        if rel_path == 'index.html':
            api_base_script = (
                '<script>\n'
                f'window.TRAINTRAIN_API_BASE = {json.dumps(_API_BASE)};\n'
                '</script>\n'
            )
            # Insert before closing </head> or before </body>
            insert_pos = content.find(b'</head>')
            if insert_pos == -1:
                insert_pos = content.find(b'</body>')
            if insert_pos != -1:
                content = content[:insert_pos] + api_base_script.encode('utf-8') + content[insert_pos:]

        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(content)

    def _read_body(self):
        length = int(self.headers.get('Content-Length', 0))
        if length == 0:
            return b''
        return self.rfile.read(length)

    def _parse_json_body(self):
        body = self._read_body()
        if not body:
            return None
        try:
            return json.loads(body.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    # ── Routes ──────────────────────────────────────────────────

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # Static files
        if path in STATIC_FILES:
            self._send_static(path)
            return

        # API: list presets
        if path == '/api/presets':
            try:
                files = [f.replace('.json', '') for f in os.listdir(trainer.presetspath) if f.endswith('.json')]
                self._send_json(files)
            except Exception as e:
                self._send_json({'error': str(e)}, 500)
            return

        # API: load preset
        if path.startswith('/api/preset/'):
            name = unquote(path[len('/api/preset/'):])
            try:
                result = trainer.import_json(name, preset=True)
                # result = [mode, model, vae, ...all_configs, dummy]
                if len(result) >= 3:
                    data = {'mode': result[0], 'model': result[1], 'vae': result[2]}
                    for i, config in enumerate(trainer.all_configs):
                        idx = 3 + i
                        if idx < len(result):
                            data[config[0]] = result[idx]
                    self._send_json(data)
                else:
                    self._send_json({'error': 'Preset not found'}, 404)
            except Exception as e:
                self._send_json({'error': str(e)}, 500)
            return

        # API: load JSON
        if path.startswith('/api/json/'):
            name = unquote(path[len('/api/json/'):])
            try:
                result = trainer.import_json(name, preset=False)
                if len(result) >= 3:
                    data = {'mode': result[0], 'model': result[1], 'vae': result[2]}
                    for i, config in enumerate(trainer.all_configs):
                        idx = 3 + i
                        if idx < len(result):
                            data[config[0]] = result[idx]
                    self._send_json(data)
                else:
                    self._send_json({'error': 'JSON not found'}, 404)
            except Exception as e:
                self._send_json({'error': str(e)}, 500)
            return

        self._send_text('Not Found', 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # API: start training
        if path == '/api/train':
            data = self._parse_json_body()
            if data is None:
                self._send_json({'error': 'Missing body'}, 400)
                return
            try:
                mode = data.get('mode', 'LoRA')
                model = data.get('model', '')
                vae = data.get('vae', '')
                config = data.get('config', {})
                # Debug: log qwen3_path value
                qp = config.get('qwen3_path', 'NOT_IN_CONFIG')
                print(f"[DEBUG] qwen3_path = {repr(qp)}")
                orig_image = data.get('orig_image', None)
                targ_image = data.get('targ_image', None)
                images = [orig_image, targ_image] if orig_image or targ_image else None
                result = train_named(False, mode, model, vae, config, images)
                self._send_text(str(result))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_text(f'Error: {e}', 500)
            return

        # API: stop training
        if path == '/api/stop':
            data = self._parse_json_body()
            save = data.get('save', False) if data else False
            try:
                result = stop_time(save)
                self._send_text(str(result))
            except Exception as e:
                self._send_text(f'Error: {e}', 500)
            return

        # API: save preset
        if path == '/api/preset':
            data = self._parse_json_body()
            if data is None:
                self._send_json({'error': 'Missing body'}, 400)
                return
            try:
                mode = data.get('mode', 'LoRA')
                model = data.get('model', '')
                vae = data.get('vae', '')
                config = data.get('config', {})
                result = train_named(True, mode, model, vae, config)
                self._send_text(str(result))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_text(f'Error: {e}', 500)
            return

        # API: open folder
        if path == '/api/open-folder':
            try:
                os.startfile(trainer.jsonspath)
                self._send_text('OK')
            except Exception as e:
                self._send_text(f'Error: {e}', 500)
            return

        self._send_text('Not Found', 404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, format, *args):
        """Suppress default logging; use print for cleaner output."""
        print(f"[{self.log_date_time_string()}] {args[0]} {args[1]} {args[2]}")


# ── Main ────────────────────────────────────────────────────────

def main():
    port = int(os.environ.get('TRAINTRAIN_PORT', 7860))
    server = ThreadingHTTPServer(('0.0.0.0', port), TrainTrainHandler)
    print("")
    print("  +----------------------------------------------+")
    print("  |         TrainTrain -- Frontend Server        |")
    print("  +----------------------------------------------+")
    print(f"  |  URL:  http://localhost:{port}                |")
    print(f"  |  API:  http://localhost:{port}/api/           |")
    print("  +----------------------------------------------+")
    print("")
    print(f"  NOTE: If accessing the frontend from a different origin (e.g.")
    print(f"        via SD webui proxy), set TRAINTRAIN_API_BASE to")
    print(f"        http://localhost:{port}")
    print("")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()


if __name__ == '__main__':
    main()
