from pathlib import Path

from dotenv import load_dotenv


WORKFLOW_ROOT = Path(__file__).resolve().parent
REPO_ROOT = WORKFLOW_ROOT.parent
ANALYSIS_ROOT = WORKFLOW_ROOT / "analysis"


def analysis_config_dir(config_id: str) -> Path:
    return ANALYSIS_ROOT / config_id


def load_workflow_dotenv(override: bool = True) -> bool:
    loaded = False
    for env_path in (REPO_ROOT / ".env", WORKFLOW_ROOT / ".env"):
        if env_path.exists():
            loaded = load_dotenv(env_path, override=override) or loaded
    return loaded
