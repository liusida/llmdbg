"""Small real-wire DAP client and private persistent host for integration tests."""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import threading
import time


class DAPClient:
    def __init__(self, port, log, timeout=30):
        self.socket = socket.create_connection(('127.0.0.1', port), timeout=timeout)
        self.socket.settimeout(timeout)
        self.buffer = bytearray()
        self.seq = 0
        self.events = []
        self.log = log
        self.timeout = timeout

    def close(self):
        self.socket.close()
        self.log.close()

    def record(self, direction, message):
        self.log.write(json.dumps({'direction': direction, 'message': message})+'\n')
        self.log.flush()

    def read(self, deadline):
        while b'\r\n\r\n' not in self.buffer:
            self.receive(deadline)
        header, _, payload = self.buffer.partition(b'\r\n\r\n')
        length = int(next(line.split(b':', 1)[1] for line in header.split(b'\r\n')
                          if line.lower().startswith(b'content-length:')))
        while len(payload) < length:
            self.receive(deadline)
            header, _, payload = self.buffer.partition(b'\r\n\r\n')
        message = json.loads(payload[:length])
        self.buffer = bytearray(payload[length:])
        self.record('receive', message)
        return message

    def receive(self, deadline):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('DAP operation exceeded its deadline')
        self.socket.settimeout(remaining)
        chunk = self.socket.recv(65536)
        if not chunk:
            raise EOFError('Debugger closed the DAP connection')
        self.buffer.extend(chunk)

    def request(self, command, **arguments):
        self.seq += 1
        request = {'seq': self.seq, 'type': 'request', 'command': command, 'arguments': arguments}
        self.record('send', request)
        payload = json.dumps(request).encode()
        self.socket.sendall(f'Content-Length: {len(payload)}\r\n\r\n'.encode()+payload)
        deadline = time.monotonic()+self.timeout
        while True:
            message = self.read(deadline)
            if message.get('type') == 'response':
                if message.get('request_seq') != self.seq:
                    raise AssertionError(f'Unexpected DAP response: {message}')
                if not message.get('success'):
                    raise AssertionError(f'{command} failed: {message}')
                return message.get('body', {})
            self.events.append(message)

    def stopped(self):
        deadline = time.monotonic()+self.timeout
        while True:
            for index, event in enumerate(self.events):
                name = event.get('event')
                if name in {'terminated', 'exited', 'llmdbg/computationFailed'}:
                    raise AssertionError(f'Debugger failed before expected stop: {event}')
                if name == 'stopped':
                    del self.events[:index+1]
                    if event.get('body', {}).get('reason') == 'exception':
                        raise AssertionError(f'Debugger exception: {event}')
                    return event['body']
            self.events.append(self.read(deadline))


class DebugHost:
    """Own only the subprocess started here; never attach to a user's host."""
    def __init__(self, logs: Path, timeout=30, env_overrides=None):
        self.logs = logs
        logs.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.stderr = (logs/'host-stderr.log').open('w')
        self.stdout = (logs/'host-stdout.log').open('w')
        env = dict(os.environ, LLMDBG_GENERATION_CACHE_DIR=str(logs/'generation-cache'))
        env.update(env_overrides or {})
        root = Path(__file__).resolve().parents[1]
        env['PYTHONPATH'] = str(root/'src')+os.pathsep+env.get('PYTHONPATH', '')
        self.process = subprocess.Popen([sys.executable, '-u', '-m', 'llmdbg.host', '--port', '0'],
            cwd=root, env=env, stdout=subprocess.PIPE, stderr=self.stderr, text=True)
        ready = queue.Queue()

        def consume():
            for line in self.process.stdout:
                self.stdout.write(line)
                self.stdout.flush()
                if line.startswith('LLMDBG_HOST_READY '):
                    ready.put(int(line.split()[-1]))
            ready.put(None)

        self.reader = threading.Thread(target=consume, daemon=True)
        self.reader.start()
        try:
            self.port = ready.get(timeout=timeout)
            if self.port is None:
                raise RuntimeError(f'Debug host exited; inspect {logs}')
        except BaseException:
            self.close()
            raise

    def client(self, name):
        return DAPClient(self.port, (self.logs/(name+'.dap.jsonl')).open('w'), self.timeout)

    def close(self):
        if self.stderr.closed:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.reader.join(timeout=10)
        self.process.stdout.close()
        self.stdout.close()
        self.stderr.close()
