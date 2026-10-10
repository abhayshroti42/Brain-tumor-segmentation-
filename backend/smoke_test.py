"""Smoke test for a RUNNING backend.
Usage: python backend\\smoke_test.py [path\\to\\sample.npz]"""
import io, json, sys, urllib.error, urllib.request, uuid
import numpy as np

BASE = "http://127.0.0.1:8000"
passed = []


def call(method, path, files=None):
    data, headers = None, {}
    if files:
        b = uuid.uuid4().hex
        body = b""
        for name, (fname, content) in files.items():
            body += (f'--{b}\r\nContent-Disposition: form-data; name="{name}"; '
                     f'filename="{fname}"\r\nContent-Type: application/octet-stream\r\n\r\n'
                     ).encode() + content + b"\r\n"
        data = body + f"--{b}--\r\n".encode()
        headers["Content-Type"] = f"multipart/form-data; boundary={b}"
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def npz(**arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def check(name, ok, detail=""):
    passed.append(ok)
    print("PASS" if ok else "FAIL", "-", name, detail)


def upload(fname, content):
    return call("POST", "/predict/npz", {"file": (fname, content)})


s, b = call("GET", "/health")
check("health: model ready", s == 200 and json.loads(b)["model_ready"])
good = (4, 128, 128, 128)
cases = {
    "wrong shape": ("a.npz", npz(image=np.zeros((4, 64, 64, 64), np.float32))),
    "missing 'image' key": ("a.npz", npz(foo=np.zeros(3))),
    "non-float dtype": ("a.npz", npz(image=np.zeros(good, np.int16))),
    "NaN values": ("a.npz", npz(image=np.full(good, np.nan, np.float32))),
    "wrong extension": ("a.txt", b"hello"),
    "not a real npz": ("a.npz", b"not a zip file"),
}
for name, (fname, content) in cases.items():
    s, b = upload(fname, content)
    check(f"invalid upload ({name}) -> 400", s == 400, b.decode()[:70])

s, _ = call("POST", "/predict/nifti")
check("nifti disabled -> 501", s == 501)
s, _ = call("GET", "/results/nope/mask")
check("unknown result id -> 404", s == 404)

if len(sys.argv) > 1:
    s, b = upload("sample.npz", open(sys.argv[1], "rb").read())
    ok = s == 200
    check("valid sample -> 200", ok, b.decode()[:120] if not ok else "")
    if ok:
        rid = json.loads(b)["id"]
        for path in (f"/results/{rid}/mask", f"/results/{rid}/image/0"):
            s, b = call("GET", path)
            check(f"{path.split(rid)[1]} is 2097152 bytes", s == 200 and len(b) == 128 ** 3)
        s, _ = call("GET", f"/results/{rid}/image/9")
        check("channel 9 -> 400", s == 400)
        s, b = call("GET", f"/results/{rid}/mask.npz")
        check("mask.npz loads as (128,128,128)",
              s == 200 and np.load(io.BytesIO(b))["mask"].shape == (128, 128, 128))

print(f"\n{sum(passed)}/{len(passed)} passed")
sys.exit(0 if all(passed) else 1)