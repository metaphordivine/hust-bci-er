from __future__ import annotations

import os


def _external_env_lookup_enabled() -> bool:
    if "PYTEST_CURRENT_TEST" not in os.environ:
        return True
    return os.environ.get("AGENT_ENABLE_EXTERNAL_ENV_LOOKUP_FOR_TESTS") == "1"


def _windows_env_value(name: str, scope: str) -> str | None:
    if os.name != "nt" or not _external_env_lookup_enabled():
        return None
    try:
        import winreg
    except ImportError:  # pragma: no cover - non-Windows fallback
        return None

    if scope == "User":
        hive = winreg.HKEY_CURRENT_USER
        key_path = "Environment"
    elif scope == "Machine":
        hive = winreg.HKEY_LOCAL_MACHINE
        key_path = r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
    else:
        raise ValueError(f"unknown environment scope: {scope}")

    try:
        with winreg.OpenKey(hive, key_path) as key:
            value, _ = winreg.QueryValueEx(key, name)
    except OSError:
        return None
    return str(value).strip() or None


def configured_env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is not None and value.strip():
        return value.strip()
    for scope in ("User", "Machine"):
        value = _windows_env_value(name, scope)
        if value:
            return value
    return default
