
import json
import os

from common.constants import OUTPUT_PATH

def set(layer_name, path):
    """
    layer_name: which pipeline layer this output path belongs to
        (e.g. "l1_doc_processing").
    path: the output path that layer just produced.

    Records path under layer_name in the shared output-path JSON
    file at OUTPUT_PATH, preserving every other layer's entry already
    stored there. If OUTPUT_PATH exists but is empty or not valid
    JSON (e.g. left over from a crashed previous write), it is
    treated the same as OUTPUT_PATH not existing yet, and its entries
    are not preserved. Creates OUTPUT_PATH's parent directory and the
    file itself if they do not exist yet. Writes to a temporary file
    first and atomically renames it over OUTPUT_PATH, so a crash
    mid-write cannot leave a corrupted or partially written file
    behind.

    Returns nothing.
    """
    data = {}
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}

    data[layer_name] = path

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    tmp_path = OUTPUT_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, OUTPUT_PATH)

def get(layer_name):
    """
    layer_name: which pipeline layer's output path to look up
        (e.g. "l1_doc_processing").

    Reads the shared output-path JSON file at OUTPUT_PATH and looks
    up the path previously recorded for layer_name.

    Raises KeyError if OUTPUT_PATH does not exist yet, or if
    layer_name has no entry in it.

    Returns the output path recorded for layer_name.
    """
    if not os.path.exists(OUTPUT_PATH):
        raise KeyError(f"{OUTPUT_PATH} does not exist")

    with open(OUTPUT_PATH, "r") as f:
        data = json.load(f)

    if layer_name not in data:
        raise KeyError(f"{layer_name} not found in {OUTPUT_PATH}")

    path = data[layer_name]
    return path