"""Sealed contributions: the key pair that seals contributed lines for the maintainer, and their opening.

Manuscribe seals a package's lines (a zip: each line's image, its corrected text, what was proposed) with AES-256-GCM
under a key of its own, and wraps that key with RSA-OAEP (SHA-256) to the maintainer's public key, which it fetches
from the models' site (contribute.json, written by `python -m msocr.site`); only the private key, kept on the
maintainer's computer, opens it. The same is done by manuscribe-contributions/tools/inbox.py (where the key is a
secret of that repository) and by Manuscribe's recogview.js (seal).

    .venv/bin/python -m msocr.seal keygen       make the key pair (once): KEY, and KEY with .pub beside it
    .venv/bin/python -m msocr.seal show         the public key's id and what contribute.json carries

KEY: ~/.config/manuscribe/contribution-key.pem, or MANUSCRIBE_CONTRIBUTION_KEY. Keep a copy of it somewhere safe: the
lines sealed to it open with nothing else.
"""
import base64
import hashlib
import json
import os
import sys
from pathlib import Path

ALG = "RSA-OAEP-256+A256GCM"
KEY = Path(os.environ.get("MANUSCRIBE_CONTRIBUTION_KEY") or Path.home() / ".config" / "manuscribe" / "contribution-key.pem")


def _ser():
    from cryptography.hazmat.primitives import serialization
    return serialization


def spki(public_key):
    s = _ser()
    return public_key.public_bytes(s.Encoding.DER, s.PublicFormat.SubjectPublicKeyInfo)


def key_id(public_key):
    return hashlib.sha256(spki(public_key)).hexdigest()[:16]


def public_info(public_key):
    """-> what contribute.json carries of the key"""
    return {"id": key_id(public_key), "alg": ALG, "spki": base64.b64encode(spki(public_key)).decode()}


def keygen(path=KEY, bits=4096):
    from cryptography.hazmat.primitives.asymmetric import rsa
    path = Path(path)
    if path.exists():
        raise SystemExit(f"[seal] {path} is there already: a new key would leave the lines sealed to it unopenable")
    priv = rsa.generate_private_key(public_exponent=65537, key_size=bits)
    s = _ser()
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(priv.private_bytes(s.Encoding.PEM, s.PrivateFormat.PKCS8, s.NoEncryption()))
    path.with_suffix(".pub").write_bytes(priv.public_key().public_bytes(s.Encoding.PEM, s.PublicFormat.SubjectPublicKeyInfo))
    return priv


def load_private(path=KEY):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"[seal] no private key at {path} (python -m msocr.seal keygen makes one; MANUSCRIBE_CONTRIBUTION_KEY names another)")
    return _ser().load_pem_private_key(path.read_bytes(), password=None)


def load_public(path=None):
    path = Path(path) if path else KEY.with_suffix(".pub")
    if path.exists():
        return _ser().load_pem_public_key(path.read_bytes())
    return load_private(KEY).public_key()


def seal(inner, public_key):
    """-> (seal, sealed bytes): as Manuscribe seals (for the tests)"""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    k, iv = AESGCM.generate_key(bit_length=256), os.urandom(12)
    sealed = AESGCM(k).encrypt(iv, inner, None)
    wrapped = public_key.encrypt(k, padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None))
    return ({"alg": ALG, "key": key_id(public_key), "wrapped": base64.b64encode(wrapped).decode(), "iv": base64.b64encode(iv).decode(),
             "file": "sealed.bin", "bytes": len(sealed), "sha256": hashlib.sha256(sealed).hexdigest()}, sealed)


def open_sealed(s, sealed, priv):
    """-> the lines' zip; raises where the key, the checksum or the bytes are not right"""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    if s.get("alg") != ALG:
        raise ValueError(f"sealed with {s.get('alg')!r}, not {ALG}")
    if key_id(priv.public_key()) != s.get("key"):
        raise ValueError(f"sealed to key {s.get('key')}, not to this one ({key_id(priv.public_key())})")
    if hashlib.sha256(sealed).hexdigest() != s.get("sha256"):
        raise ValueError("the sealed lines are not what the package says (checksum)")
    k = priv.decrypt(base64.b64decode(s["wrapped"]), padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None))
    return AESGCM(k).decrypt(base64.b64decode(s["iv"]), sealed, None)


def package_id(about, sha):
    return "c" + str(about.get("made", ""))[:10].replace("-", "") + "-" + sha[:12]


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "keygen":
        priv = keygen()
        print(f"[seal] key {key_id(priv.public_key())} made: {KEY} (private: keep a copy somewhere safe) and {KEY.with_suffix('.pub')}")
    elif cmd == "show":
        print(json.dumps(public_info(load_public()), indent=1))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
