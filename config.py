import json
import os

if os.path.exists('config.json') is False:
    raise RuntimeError('No config.json file present.')

with open('config.json', 'r') as file:
    CONFIG = json.load(file)


def get(key):
    return CONFIG[key]
