
import json
import os

from common.constants import JSON_PATH

def set(key, value):
    """
    key: name to store value under (e.g. "doc_type").
    value: the value to store.

    Records value under key in the shared global-values JSON file at
    JSON_PATH, preserving every other key already stored there. If
    JSON_PATH exists but is empty or not valid JSON (e.g. left over
    from a crashed previous write), it is treated the same as
    JSON_PATH not existing yet, and its entries are not preserved.
    Creates JSON_PATH's parent directory and the file itself if they
    do not exist yet. Writes to a temporary file first and atomically
    renames it over JSON_PATH, so a crash mid-write cannot leave a
    corrupted or partially written file behind.

    Returns nothing.
    """
    data = {}
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}

    data[key] = value

    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
    tmp_path = JSON_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, JSON_PATH)

def get(key):
    """
    key: name of the value to look up (e.g. "doc_type").

    Reads the shared global-values JSON file at JSON_PATH and looks
    up the value previously recorded for key.

    Raises KeyError if JSON_PATH does not exist yet, or if key has
    no entry in it.

    Returns the value recorded for key.
    """
    if not os.path.exists(JSON_PATH):
        raise KeyError(f"{JSON_PATH} does not exist")

    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    if key not in data:
        raise KeyError(f"{key} not found in {JSON_PATH}")

    value = data[key]
    return value