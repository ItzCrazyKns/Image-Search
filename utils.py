import json

def load_config(config_path="./config.json"):
    with open(config_path, "r") as file:
        config = json.load(file)
    return config