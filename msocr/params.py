"""The pipeline's shared parameters, from msocr/pipeline.json: read by the Python reference here and by the JavaScript
port in Manuscribe, so that neither codebase holds its own copy of a threshold, a band, or a learned weight.

    from .params import P
    P["segmenter"]["img_h"]
"""
import json
from pathlib import Path

P = json.loads(Path(__file__).with_name("pipeline.json").read_text(encoding="utf-8"))
VERSION = P["version"]
