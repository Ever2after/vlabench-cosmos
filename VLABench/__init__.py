import os
from pathlib import Path
import sys
import logging

is_windows = sys.platform == 'win32'
is_linux = sys.platform.startswith('linux')
is_mac = sys.platform == 'darwin'

env_var_name = 'VLABENCH_ROOT'


def _resolve_vlabench_root() -> Path:
    package_dir = Path(__file__).resolve().parent
    configured_root = os.environ.get(env_var_name)

    candidates = []
    if configured_root:
        candidate = Path(configured_root).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        candidates.append(candidate)
    candidates.append(package_dir)

    for candidate in candidates:
        if (candidate / "configs" / "robot_config.json").is_file():
            return candidate

    return package_dir


vlabench_root = _resolve_vlabench_root()
os.environ[env_var_name] = str(vlabench_root)

if is_windows:
    logging.info("Detect Windows, set VLABENCH_ROOT to %s", vlabench_root)
elif is_linux:
    logging.info("Detect Linux, set VLABENCH_ROOT to %s", vlabench_root)
elif is_mac:
    logging.info("Detect macOS, set VLABENCH_ROOT to %s", vlabench_root)
