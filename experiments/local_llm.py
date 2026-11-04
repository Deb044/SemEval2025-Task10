"""Shared, loopback-only llama.cpp process and request helpers."""
import json
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from experiments.common import sha256


def request(port, route, body=None, timeout=600):
    payload = None if body is None else json.dumps(body).encode('utf-8')
    req = urllib.request.Request(f'http://127.0.0.1:{int(port)}{route}', data=payload,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'Local model request failed ({error.code}): {error.read().decode()[:600]}') from error


@contextmanager
def server(binary, model, out, context=2048, port=8082):
    """Own exactly one subprocess; never replace an existing service on the port."""
    import socket
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', port)) == 0:
            raise ValueError(f'Port {port} is already occupied; select a free port')
    weights = Path('work/models') / model['file']
    if not weights.exists() or sha256(weights) != model['sha256']:
        raise ValueError('Missing or mismatched model weights: ' + model['file'])
    Path(out).mkdir(parents=True, exist_ok=True)
    with open(Path(out) / (model['file'] + '.log'), 'w', encoding='utf-8') as log:
        process = subprocess.Popen([str(binary), '-m', str(weights), '--host', '127.0.0.1', '--port', str(port),
            '-c', str(context), '-t', '4', '-ngl', '0', '-np', '1', '--jinja', '--seed', '42'],
            stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            deadline = time.monotonic() + 120
            while True:
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('Model server failed; inspect its private log')
                try:
                    if request(port, '/health', timeout=1).get('status') == 'ok':
                        break
                except (OSError, RuntimeError):
                    time.sleep(.25)
            yield port
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def complete(port, messages, max_tokens=256, schema=None):
    body = {'messages': messages, 'temperature': 0, 'seed': 42, 'max_tokens': max_tokens,
            'cache_prompt': False, 'chat_template_kwargs': {'enable_thinking': False}}
    if schema:
        body['response_format'] = {'type': 'json_object', 'schema': schema}
    started = time.perf_counter()
    response = request(port, '/v1/chat/completions', body)
    choice = response['choices'][0]
    return {'text': ' '.join((choice['message'].get('content') or '').split()),
            'finish_reason': choice['finish_reason'], 'seconds': time.perf_counter() - started,
            'usage': response.get('usage', {})}
