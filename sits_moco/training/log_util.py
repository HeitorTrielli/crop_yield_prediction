"""Training log helpers (flush stdout for piped / WSL runs)."""


def log_training(msg: str) -> None:
    print(msg, flush=True)
