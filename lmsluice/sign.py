"""Request signing for object stores, with nothing installed.

Every one of the three clouds this package reads from authenticates an HTTPS
request with **HMAC-SHA256 over a canonical form of that request**. They differ
in what goes into the canonical form and where the result is put, and in
nothing else. `hmac` and `hashlib` are in the standard library, so the whole of
`--load-format lmsluice` reaching S3, GCS and Azure is this file plus a URL
parser -- no boto3, no google-cloud-storage, no azure-storage-blob.

That matters more here than it would elsewhere. The one row in
`docs/competition.md` §7 that only this project ticks is **no install
required**, and the row beside it that only competitors tick is object storage.
Closing the second by giving up the first would be a trade, not a gain: on the
machines this exists for -- a phone, an iGPU laptop, a CI container with no
wheels cached -- `pip install boto3` is frequently the whole obstacle.

## What is here, and what is deliberately not

Signing only: pure functions from a request description to a header value. No
sockets, no credentials discovery, no retries. That split is what makes this
file testable against published known answers rather than against a cloud
account, and `test_lmsluice.py` checks it against AWS's own worked example down
to the signature hex.

**Nothing here invents cryptography.** HMAC-SHA256 and SHA-256 from the
standard library, composed exactly as each vendor's specification states.

## The three, and why two of them are one

- **AWS Signature V4** (`sigv4_headers`). Canonical request → string to sign →
  a signing key derived by four chained HMACs → `Authorization`. Used for S3
  and for every S3-compatible store, which is the same protocol at a different
  endpoint: MinIO, Cloudflare R2, Backblaze B2, Ceph RGW, Wasabi.
- **Google Cloud Storage** is the same function. Its XML API accepts SigV4 with
  HMAC keys, so `gs://` with an HMAC pair is `sigv4_headers` with
  `service="s3"` and a different host. GCS's own JSON API wants an OAuth
  bearer token instead, which needs no signing at all -- one header.
- **Azure Blob Shared Key** (`shared_key_header`) is a different canonical form
  and one HMAC rather than four. A SAS token needs no signing either: it *is* a
  signature, already made, carried in the query string.

## One detail that is not a detail: S3 does not normalise its paths

SigV4 has two path rules. Most services normalise the URI and encode it twice;
**S3 does neither**, because an object key may legitimately contain `.`, `..`
or a double slash and normalising would sign a different object than the one
requested. `normalise=False` is therefore the setting for S3, GCS's XML API and
every S3-compatible, and the parameter exists at all so the generic rule can be
checked against the published test vectors that exercise it.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import hmac
import urllib.parse

ALGORITHM = "AWS4-HMAC-SHA256"
# SHA-256 of the empty string. Every read this package makes has an empty body,
# and S3 requires the header whether or not there is a payload to hash.
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

# Unreserved characters per RFC 3986, which is what AWS says to preserve.
_UNRESERVED = ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
               "0123456789-._~")


def now() -> datetime.datetime:
    """UTC, injectable so a known-answer test can pin the timestamp."""
    return datetime.datetime.now(datetime.timezone.utc)


def _quote(s: str, safe: str = "") -> str:
    return urllib.parse.quote(s, safe=_UNRESERVED + safe)


def canonical_uri(path: str, *, normalise: bool = False) -> str:
    """The path as SigV4 wants to sign it.

    With `normalise`, `.` and `..` segments are resolved and the result encoded
    a second time, which is the rule for most AWS services. Without it the path
    is encoded once and left alone, which is the rule for S3 -- and the right
    one, because `a//b` and `a/./b` are three different object keys and a
    signature over a tidied path would authorise a request for a different one.
    """
    if not path:
        return "/"
    if normalise:
        out: list[str] = []
        for seg in path.split("/"):
            if seg == ".":
                continue
            if seg == "..":
                if out:
                    out.pop()
                continue
            out.append(seg)
        path = "/".join(out) or "/"
        if not path.startswith("/"):
            path = "/" + path
        return _quote(_quote(path, safe="/"), safe="/")
    return _quote(path, safe="/")


def canonical_query(query: str) -> str:
    """Sorted, each name and value encoded once, `=` even when the value is
    empty -- which is what trips people, because `?acl` signs as `acl=`."""
    if not query:
        return ""
    pairs = []
    for part in query.split("&"):
        if not part:
            continue
        name, _, value = part.partition("=")
        pairs.append((_quote(urllib.parse.unquote(name)),
                      _quote(urllib.parse.unquote(value))))
    pairs.sort()
    return "&".join(f"{n}={v}" for n, v in pairs)


def canonical_headers(headers: dict) -> tuple[str, str]:
    """(canonical headers block, signed header list).

    Names lowercased, values stripped, sorted by name. Sequential inner spaces
    should also collapse for a strict reading of the specification; no header
    this package sends contains any, and collapsing them inside a quoted string
    would corrupt values that legitimately do, so it is left alone deliberately
    rather than by omission.
    """
    items = sorted((k.lower(), " ".join(str(v).split()))
                   for k, v in headers.items())
    block = "".join(f"{k}:{v}\n" for k, v in items)
    return block, ";".join(k for k, _ in items)


def canonical_request(method: str, path: str, query: str, headers: dict,
                      payload_sha256: str, *, normalise: bool = False) -> str:
    block, signed = canonical_headers(headers)
    return "\n".join([method.upper(), canonical_uri(path, normalise=normalise),
                      canonical_query(query), block, signed, payload_sha256])


def signing_key(secret: str, date: str, region: str, service: str) -> bytes:
    """Four chained HMACs. The key is scoped to a day, a region and a service,
    which is why a leaked signature is worth so much less than a leaked key."""
    k = ("AWS4" + secret).encode()
    for part in (date, region, service, "aws4_request"):
        k = hmac.new(k, part.encode(), hashlib.sha256).digest()
    return k


def sigv4_headers(*, method: str, url: str, region: str, service: str = "s3",
                  access_key: str, secret_key: str, token: str | None = None,
                  headers: dict | None = None,
                  payload_sha256: str = EMPTY_SHA256,
                  when: datetime.datetime | None = None,
                  normalise: bool = False) -> dict:
    """The headers that authorise one request. Never mutates its input.

    Returns the caller's headers plus `Host`, `x-amz-date`,
    `x-amz-content-sha256`, an `Authorization`, and `x-amz-security-token`
    where a session token is in play. The secret is used and dropped; it is
    never stored on anything this returns.
    """
    parsed = urllib.parse.urlparse(url)
    stamp = (when or now()).strftime("%Y%m%dT%H%M%SZ")
    date = stamp[:8]
    signed_headers = {**(headers or {}),
                      "host": parsed.netloc,
                      "x-amz-content-sha256": payload_sha256,
                      "x-amz-date": stamp}
    if token:
        signed_headers["x-amz-security-token"] = token
    creq = canonical_request(method, parsed.path, parsed.query, signed_headers,
                             payload_sha256, normalise=normalise)
    scope = f"{date}/{region}/{service}/aws4_request"
    to_sign = "\n".join([ALGORITHM, stamp, scope,
                         hashlib.sha256(creq.encode()).hexdigest()])
    signature = hmac.new(signing_key(secret_key, date, region, service),
                         to_sign.encode(), hashlib.sha256).hexdigest()
    _, signed = canonical_headers(signed_headers)
    return {**signed_headers,
            "Authorization": (f"{ALGORITHM} Credential={access_key}/{scope}, "
                              f"SignedHeaders={signed}, Signature={signature}")}


# -- Azure Blob ------------------------------------------------------------

def canonical_azure_resource(account: str, path: str, query: str) -> str:
    """`/account/path`, then every query parameter, lowercased and sorted.

    A parameter appearing more than once has its values sorted and joined by
    commas, which is the specification's rule and the part an implementation
    written from memory usually gets wrong.
    """
    out = f"/{account}{path or '/'}"
    if query:
        params: dict[str, list[str]] = {}
        for name, values in urllib.parse.parse_qs(
                query, keep_blank_values=True).items():
            params.setdefault(name.lower(), []).extend(values)
        for name in sorted(params):
            out += "\n" + name + ":" + ",".join(sorted(params[name]))
    return out


def azure_string_to_sign(method: str, account: str, path: str, query: str,
                         headers: dict) -> str:
    """Azure's canonical form: thirteen fixed lines, then `x-ms-*`, then the
    resource. The fixed lines are positional -- a missing one is an empty line,
    not an omitted line -- and `Content-Length` is empty rather than `0`, which
    changed in the 2015-02-21 API version and is the single most common cause
    of a 403 against a correct key.
    """
    lower = {k.lower(): " ".join(str(v).split()) for k, v in headers.items()}
    length = lower.get("content-length", "")
    if length == "0":
        length = ""
    fixed = [method.upper(), lower.get("content-encoding", ""),
             lower.get("content-language", ""), length,
             lower.get("content-md5", ""), lower.get("content-type", ""),
             lower.get("date", ""), lower.get("if-modified-since", ""),
             lower.get("if-match", ""), lower.get("if-none-match", ""),
             lower.get("if-unmodified-since", ""), lower.get("range", "")]
    ms = "".join(f"{k}:{v}\n" for k, v in sorted(lower.items())
                 if k.startswith("x-ms-"))
    return "\n".join(fixed) + "\n" + ms + canonical_azure_resource(
        account, path, query)


def shared_key_header(*, method: str, url: str, account: str, key: str,
                      headers: dict | None = None,
                      when: datetime.datetime | None = None) -> dict:
    """Azure Shared Key authorisation for one request.

    The account key is base64 in every place Azure publishes it, and it is the
    *decoded* bytes that key the HMAC -- signing with the base64 text produces
    a well-formed header that is always rejected.
    """
    parsed = urllib.parse.urlparse(url)
    stamp = (when or now()).strftime("%a, %d %b %Y %H:%M:%S GMT")
    out = {**(headers or {}), "x-ms-date": stamp, "x-ms-version": "2021-08-06"}
    to_sign = azure_string_to_sign(method, account, parsed.path, parsed.query,
                                   out)
    signature = base64.b64encode(hmac.new(
        base64.b64decode(key), to_sign.encode("utf-8"),
        hashlib.sha256).digest()).decode()
    return {**out, "Authorization": f"SharedKey {account}:{signature}"}
