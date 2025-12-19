import threading
import time
import requests
import psutil
import subprocess
import sys
from pathlib import Path

from pystray import Icon, Menu, MenuItem
from PIL import Image, ImageDraw

CONTROLLER_URL = "http://localhost:8080"
AGENT_NAME = "agent-lite-win-1"  # should match what app.py registers as

APP_PATH = Path(__file__).resolve().parent / "app.py"


def _pythonw_exe() -> str:
    """
    Prefer pythonw.exe on Windows so we don't spawn a console window.
    If not found, fall back to sys.executable.
    """
    exe = sys.executable
    lower = exe.lower()
    if lower.endswith("python.exe"):
        candidate = exe[:-10] + "pythonw.exe"
        return candidate
    return exe


class AgentTray:
    def __init__(self):
        self.icon = Icon("agent-lite", self._create_icon(), "Agent Lite")
        self.running = True
        self.status_text = "Starting..."
        self.agent_proc: subprocess.Popen | None = None

        # Start the agent right away
        self._start_agent()

        # Start tooltip updater
        self._start_background_updater()

        self.icon.menu = Menu(
            MenuItem("Start / Restart agent", self.start_restart_agent),
            MenuItem("Stop agent", self.stop_agent),
            MenuItem("Pause agent (controller quarantine)", self.pause_agent),
            MenuItem("Resume agent (controller restore)", self.resume_agent),
            MenuItem("Exit tray", self.exit_tray),
        )

    def _create_icon(self, size=64, color_fg=(0, 255, 0), color_bg=(0, 0, 0)):
        image = Image.new("RGB", (size, size), color_bg)
        d = ImageDraw.Draw(image)
        margin = 10
        d.ellipse((margin, margin, size - margin, size - margin), fill=color_fg)
        return image

    # -------------------------
    # Agent process management
    # -------------------------

    def _start_agent(self):
        """
        Launch app.py with pythonw.exe so it runs without a console window.
        """
        try:
            if not APP_PATH.exists():
                self.status_text = f"Missing app.py at {APP_PATH}"
                self.icon.title = self.status_text
                return

            # If already running, don't start another
            if self.agent_proc and self.agent_proc.poll() is None:
                return

            exe = _pythonw_exe()
            self.agent_proc = subprocess.Popen(
                [exe, str(APP_PATH)],
                cwd=str(APP_PATH.parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            self.agent_proc = None

    def _stop_agent_proc(self):
        """
        Stop the local agent process if we launched it.
        """
        try:
            if self.agent_proc and self.agent_proc.poll() is None:
                self.agent_proc.terminate()
                try:
                    self.agent_proc.wait(timeout=3)
                except Exception:
                    try:
                        self.agent_proc.kill()
                    except Exception:
                        pass
        finally:
            self.agent_proc = None

    def start_restart_agent(self, icon, item):
        self._stop_agent_proc()
        time.sleep(0.2)
        self._start_agent()

    def stop_agent(self, icon, item):
        self._stop_agent_proc()

    # -------------------------
    # Controller actions
    # -------------------------

    def pause_agent(self, icon, item):
        try:
            requests.post(
                f"{CONTROLLER_URL}/agents/{AGENT_NAME}/quarantine",
                json={"reason": "paused_from_tray"},
                timeout=3,
            )
        except Exception:
            pass

    def resume_agent(self, icon, item):
        try:
            requests.post(
                f"{CONTROLLER_URL}/agents/{AGENT_NAME}/restore",
                json={},
                timeout=3,
            )
        except Exception:
            pass

    # -------------------------
    # Status tooltip updater
    # -------------------------

    def _start_background_updater(self):
        t = threading.Thread(target=self._update_loop, daemon=True)
        t.start()

    def _update_loop(self):
        while self.running:
            try:
                cpu = psutil.cpu_percent(interval=0.5)

                # controller agent listing (adjust if your controller uses /api/agents etc.)
                resp = requests.get(f"{CONTROLLER_URL}/agents", timeout=2)
                data = resp.json()

                agent_info = data.get(AGENT_NAME, {}) if isinstance(data, dict) else {}
                state = agent_info.get("state", "unknown")
                metrics = agent_info.get("metrics", {}) or {}
                tasks_completed = metrics.get("tasks_completed", 0)

                proc_state = "down"
                if self.agent_proc and self.agent_proc.poll() is None:
                    proc_state = "up"

                self.status_text = (
                    f"{AGENT_NAME} [{state}]  "
                    f"proc:{proc_state}  "
                    f"CPU:{cpu:.0f}%  "
                    f"Completed:{tasks_completed}"
                )
                self.icon.title = self.status_text

            except Exception:
                proc_state = "down"
                if self.agent_proc and self.agent_proc.poll() is None:
                    proc_state = "up"
                self.status_text = f"Controller unreachable | proc:{proc_state}"
                self.icon.title = self.status_text

            time.sleep(3)

    # -------------------------
    # Exit
    # -------------------------

    def exit_tray(self, icon, item):
        self.running = False
        self._stop_agent_proc()
        self.icon.stop()

    def run(self):
        self.icon.run()


if __name__ == "__main__":
    tray = AgentTray()
    tray.run()
