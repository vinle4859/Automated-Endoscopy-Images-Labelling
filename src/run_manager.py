"""
Run Manager — Timestamped output directories for reproducibility.
=================================================================
Every pipeline step (generate, yolo, diagnostics, validation) writes its
outputs into a timestamped subdirectory under ``results/runs/``.  This
prevents old outputs from being confused with current ones and keeps a
full history of every run.

Session grouping (v8):
    When ``--step all`` is used, all steps share a single **session** directory.
    The session timestamp is set once at the start of the pipeline and reused
    for all steps, so they are grouped together:

        results/runs/
            20260225_143520_session/
                generate/
                    visualizations/...
                    run_info.json
                yolo/
                    train/weights/best.pt
                    run_info.json
                session_info.json          ← session-level metadata

    When individual steps are run (``--step generate``), each step still gets
    its own timestamped directory as before:

        results/runs/
            20260225_143520_generate/

Usage in scripts:
    from run_manager import create_run_dir, get_latest_run_dir, start_session

    # For grouped runs (--step all):
    session = start_session(version='v8')
    run_dir = create_run_dir('generate', session=session, ...)

    # For single-step runs:
    run_dir = create_run_dir('generate', version='v8', params={...})
    vis_dir = os.path.join(run_dir, 'visualizations')
"""

import json
import os
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(PROJECT_ROOT, 'results', 'runs')


class Session:
    """
    Groups multiple pipeline steps under one timestamped directory.
    """
    def __init__(self, version: str = 'unknown'):
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.version = version
        self.dirname = f"{self.timestamp}_session"
        self.session_dir = os.path.join(RUNS_DIR, self.dirname)
        os.makedirs(self.session_dir, exist_ok=True)

        # Write session-level metadata
        meta = {
            'timestamp': datetime.now().isoformat(),
            'type': 'session',
            'version': version,
            'session_dir': self.session_dir,
            'steps': [],
        }
        self._meta_path = os.path.join(self.session_dir, 'session_info.json')
        with open(self._meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def _record_step(self, step_name: str, run_dir: str):
        """Append a completed step to the session metadata."""
        with open(self._meta_path, encoding='utf-8') as f:
            meta = json.load(f)
        meta['steps'].append({
            'step': step_name,
            'run_dir': run_dir,
            'timestamp': datetime.now().isoformat(),
        })
        with open(self._meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


def start_session(version: str = 'unknown') -> Session:
    """Start a new session that groups multiple pipeline steps."""
    return Session(version=version)


def create_run_dir(step_name: str, *,
                   version: str = 'unknown',
                   params: dict | None = None,
                   extra_meta: dict | None = None,
                   session: Session | None = None) -> str:
    """
    Create a timestamped run directory and write ``run_info.json``.

    Parameters
    ----------
    step_name : str
        Short identifier for the pipeline step, e.g. ``'generate'``,
        ``'yolo'``, ``'signal_diag'``, ``'param_valid'``.
    version : str
        Pipeline version tag, e.g. ``'v8'``.
    params : dict, optional
        Key configuration parameters to record.
    extra_meta : dict, optional
        Any additional metadata to store (e.g. git hash, data counts).
    session : Session, optional
        If provided, the run directory is created inside the session
        directory instead of as a standalone timestamped directory.

    Returns
    -------
    str
        Absolute path to the new run directory.
    """
    if session is not None:
        # Inside a session: use step_name as subdirectory
        run_dir = os.path.join(session.session_dir, step_name)
        os.makedirs(run_dir, exist_ok=True)
        version = session.version  # use session version
    else:
        # Standalone: timestamped directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dirname = f"{timestamp}_{step_name}"
        run_dir = os.path.join(RUNS_DIR, dirname)
        os.makedirs(run_dir, exist_ok=True)

    # Write metadata
    meta = {
        'timestamp': datetime.now().isoformat(),
        'step': step_name,
        'version': version,
        'params': params or {},
        'run_dir': run_dir,
    }
    if session is not None:
        meta['session_dir'] = session.session_dir
    if extra_meta:
        meta.update(extra_meta)

    meta_path = os.path.join(run_dir, 'run_info.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Update "latest" pointer for this step
    os.makedirs(RUNS_DIR, exist_ok=True)
    latest_path = os.path.join(RUNS_DIR, f'latest_{step_name}.txt')
    with open(latest_path, 'w', encoding='utf-8') as f:
        f.write(run_dir)

    # Record in session if applicable
    if session is not None:
        session._record_step(step_name, run_dir)

    return run_dir


def get_latest_run_dir(step_name: str) -> str | None:
    """
    Return the path to the most recent run directory for *step_name*,
    or ``None`` if no run has been recorded.
    """
    latest_path = os.path.join(RUNS_DIR, f'latest_{step_name}.txt')
    if not os.path.exists(latest_path):
        return None
    with open(latest_path, encoding='utf-8') as f:
        path = f.read().strip()
    if os.path.isdir(path):
        return path
    return None


def list_runs(step_name: str | None = None) -> list[dict]:
    """
    List all recorded runs, optionally filtered by *step_name*.

    Returns a list of dicts (parsed ``run_info.json``) sorted newest-first.
    """
    if not os.path.isdir(RUNS_DIR):
        return []

    results = []
    for entry in os.listdir(RUNS_DIR):
        entry_path = os.path.join(RUNS_DIR, entry)
        if not os.path.isdir(entry_path):
            continue
        meta_path = os.path.join(entry_path, 'run_info.json')
        if not os.path.exists(meta_path):
            continue
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)
        if step_name and meta.get('step') != step_name:
            continue
        results.append(meta)

    results.sort(key=lambda m: m.get('timestamp', ''), reverse=True)
    return results
