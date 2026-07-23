"""
mcp_utils.py
------------
Small helper to auto-launch weather_server.py in the background so you don't
have to manually start it in a separate terminal before running the agent.
"""

import socket
import subprocess
import sys
import time


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def ensure_weather_server(server_path: str, host: str = "127.0.0.1", port: int = 8000, startup_timeout: float = 15):
    """Make sure weather_server.py (SSE) is up, starting it if needed.

    Returns:
        - a subprocess.Popen handle if this call started the server (caller
          is responsible for terminating it, e.g. via atexit)
        - None if a server was already listening on host:port (nothing to
          clean up — someone else, or an earlier call, is managing it)

    Raises:
        RuntimeError / TimeoutError if the server fails to come up.
    """
    if _port_open(host, port):
        return None  # already running

    process = subprocess.Popen(
        [sys.executable, server_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        if _port_open(host, port):
            return process
        if process.poll() is not None:
            raise RuntimeError(
                "weather_server.py exited before it started listening. "
                "Run it manually (`python weather_server.py`) to see the error."
            )
        time.sleep(0.3)

    process.terminate()
    raise TimeoutError(f"weather_server.py did not start listening on {host}:{port} within {startup_timeout}s.")