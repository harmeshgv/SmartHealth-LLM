import json


def read_json(json_path: str):
    with open(json_path, "r") as f:
        output = json.load(f)
    return output
