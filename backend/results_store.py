"""Tiny in-memory result store for the prototype (latest 5 results)."""
import uuid
from collections import OrderedDict

_MAX_RESULTS = 5
_results: "OrderedDict[str, dict]" = OrderedDict()


def save(volume, mask) -> str:
    rid = uuid.uuid4().hex
    _results[rid] = {"volume": volume, "mask": mask}
    while len(_results) > _MAX_RESULTS:
        _results.popitem(last=False)
    return rid


def get(rid: str):
    return _results.get(rid)