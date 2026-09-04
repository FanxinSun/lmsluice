"""S3, Google Cloud Storage and Azure Blob, as three more implementations of `pread`.

`source.py` states the shape this follows: the routing arithmetic never asks
what a source *is*, only how fast it delivers. A local file and a web server
are two implementations of one interface; an object store is a third, and the
pipeline above it does not change. `lmsluice get`, `load`, `stream`, `plan` and
`bench` work on a bucket URL because nothing in them was ever told what a
source is -- which is the test of the abstraction rather than a claim about it.

## Why there is no SDK here

All three clouds authorise an HTTPS range GET with HMAC-SHA256 over a canonical
form of the request. `sign.py` does that with `hmac` and `hashlib`. So the
whole of reaching object storage is a URL parser, a credential lookup and a
signature -- and `docs/competition.md` §7's **no install required** row, the
one only this project ticks, survives closing the object-storage row beside it.

That is not purism. `pip install boto3` pulls botocore, s3transfer, jmespath,
python-dateutil and urllib3; on the machines this exists for -- a phone, a CI
container with no wheel cache, an air-gapped box -- it is frequently the whole
obstacle. Keeping it optional means the common case needs nothing and the
uncommon case is still served.

**The SDKs remain, as named modes for what the standard library cannot do.**
Instance-metadata and IAM role credentials, OAuth service-account flows that
need RS256, and Azure managed identity are credential *sources*, not protocols;
where one is needed, `boto3`, `google-auth` or `azure-identity` supplies the
credential and the request is still signed and sent by this file. Each source
reports `mode` as `stdlib` or `sdk`, the way `Archive.route` and the crypto
backend name themselves, so a benchmark and a user both know which ran.

## Credentials

The same rules as the encryption key in `crypt.py`, for the same reason: never
on the command line, where `/proc` makes it readable by every process on the
machine; never logged; never stored by us. They are read from the places each
cloud already defines, so a machine already configured for `aws s3` or `gcloud`
or `az` needs nothing new — and a **public bucket works with nothing
configured at all**, which is the case a first-time reader is most likely to
try.

Every credential-bearing header is redacted from every error this module
raises. `redact()` is applied at the boundary rather than at each raise site,
because the failure mode is a header reaching a log through a path nobody
thought about.

## URL forms

    s3://bucket/key                      AWS, or any S3-compatible with
                                         LMSLUICE_S3_ENDPOINT set
    gs://bucket/key                      GCS, XML API with HMAC keys, or the
                                         JSON API with a bearer token
    az://account/container/blob          Azure Blob
    https://host/...                     an S3-compatible by explicit endpoint,
                                         signed when credentials are present

MinIO, Cloudflare R2, Backblaze B2, Ceph RGW and Wasabi are S3 at a different
endpoint and need no code of their own.
"""

from __future__ import annotations

import json
import os
import urllib.parse

from . import sign
from .source import HttpSource, Source

# Header names whose values authorise a request. Never printed, never logged.
SECRET_HEADERS = frozenset({
    "authorization", "x-amz-security-token", "x-ms-copy-source-authorization",
    "x-goog-encryption-key", "cookie", "proxy-authorization",
})

# Query parameters that are themselves a credential: a SAS token is a
# signature in the URL, so a URL can be a secret and must be redacted too.
SECRET_PARAMS = frozenset({"sig", "signature", "x-amz-signature",
                           "x-goog-signature", "access_token"})


class CloudError(OSError):
    """A cloud request failed. Never carries a credential."""


def redact(text: str) -> str:
    """Remove anything that would authorise a request, from anywhere.

    Applied to every message this module raises. A signature in a traceback is
    a signature in a log file, a CI transcript and a bug report, and unlike a
    key it is often still valid when it gets there.
    """
    # Both loops carry a cursor and only ever move forward. Rewriting in place
    # and re-searching from zero does not terminate -- the marker is still
    # there after the value behind it is replaced, so it matches again, and
    # again. It hung the suite before it was written this way.
    for name in SECRET_PARAMS:
        for sep in ("?", "&"):
            marker = f"{sep}{name}="
            at = 0
            while True:
                i = text.lower().find(marker, at)
                if i < 0:
                    break
                start = i + len(marker)
                stop = len(text)
                for ch in ("&", " ", "'", '"', ")", "\n"):
                    k = text.find(ch, start)
                    if 0 <= k < stop:
                        stop = k
                text = text[:start] + "REDACTED" + text[stop:]
                at = start + len("REDACTED")
    for name in SECRET_HEADERS:
        marker = name + ":"
        at = 0
        while True:
            i = text.lower().find(marker, at)
            if i < 0:
                break
            start = i + len(marker)
            stop = text.find("\n", start)
            stop = len(text) if stop < 0 else stop
            text = text[:start] + " REDACTED" + text[stop:]
            at = start + len(" REDACTED")
    return text


def redact_env(text: str) -> str:
    """Also strip anything that matches a credential currently in the
    environment, which catches the values `redact` cannot recognise by name."""
    text = redact(text)
    for var in ("AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
                "AZURE_STORAGE_KEY", "AZURE_STORAGE_SAS_TOKEN",
                "GOOGLE_OAUTH_ACCESS_TOKEN", "LMSLUICE_S3_SECRET_KEY",
                "GCS_HMAC_SECRET", "AWS_ACCESS_KEY_ID", "GCS_HMAC_ACCESS_KEY",
                "AZURE_STORAGE_CONNECTION_STRING"):
        value = os.environ.get(var)
        if value and len(value) >= 8 and value in text:
            text = text.replace(value, "REDACTED")
    return text


# -- credentials -----------------------------------------------------------

def _ini(path: str) -> dict:
    """A minimal INI reader, because `configparser` chokes on the nested
    indented blocks AWS writes into `~/.aws/config` for SSO and roles."""
    out: dict[str, dict[str, str]] = {}
    section = None
    try:
        with open(os.path.expanduser(path)) as fh:
            for line in fh:
                line = line.split("#", 1)[0].split(";", 1)[0].strip()
                if not line:
                    continue
                if line.startswith("[") and line.endswith("]"):
                    section = line[1:-1].strip()
                    out.setdefault(section, {})
                elif "=" in line and section is not None:
                    k, _, v = line.partition("=")
                    out[section][k.strip().lower()] = v.strip()
    except OSError:
        pass
    return out


def aws_credentials(profile: str | None = None) -> dict:
    """AWS credentials from the places AWS itself defines, in its own order.

    Environment first, then `~/.aws/credentials` and `~/.aws/config` for the
    named profile. Returns `{}` for a public bucket, which is a success and not
    a failure -- anonymous access must work with nothing configured.
    """
    env = {
        "access_key": os.environ.get("AWS_ACCESS_KEY_ID"),
        "secret_key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "token": os.environ.get("AWS_SESSION_TOKEN"),
        "region": (os.environ.get("AWS_REGION")
                   or os.environ.get("AWS_DEFAULT_REGION")),
    }
    if env["access_key"] and env["secret_key"]:
        return {k: v for k, v in env.items() if v}

    name = profile or os.environ.get("AWS_PROFILE") or "default"
    creds = _ini(os.environ.get("AWS_SHARED_CREDENTIALS_FILE",
                                "~/.aws/credentials")).get(name, {})
    conf = _ini(os.environ.get("AWS_CONFIG_FILE", "~/.aws/config"))
    conf = conf.get(name if name == "default" else f"profile {name}", {})
    out = {
        "access_key": creds.get("aws_access_key_id"),
        "secret_key": creds.get("aws_secret_access_key"),
        "token": creds.get("aws_session_token"),
        "region": (env["region"] or creds.get("region") or conf.get("region")),
    }
    return {k: v for k, v in out.items() if v}


def gcs_credentials() -> dict:
    """GCS credentials, preferring what the standard library can actually use.

    Two stdlib modes. An **HMAC key pair** (`gcloud storage hmac create`) makes
    the XML API byte-identical to S3 and is the simplest thing that works. A
    **bearer token** -- `GOOGLE_OAUTH_ACCESS_TOKEN`, or one minted from the
    refresh token `gcloud auth application-default login` leaves behind -- needs
    no signing at all.

    A **service-account JSON key** is the one mode this cannot do alone: it
    requires an RS256-signed JWT, and while `crypt.py` can reach OpenSSL by
    ctypes, RSA signing is a larger surface than a range GET justifies. It is
    reported as needing the SDK rather than silently ignored.
    """
    out: dict = {}
    access = os.environ.get("GCS_HMAC_ACCESS_KEY")
    secret = os.environ.get("GCS_HMAC_SECRET")
    if access and secret:
        return {"access_key": access, "secret_key": secret, "mode": "hmac"}
    token = os.environ.get("GOOGLE_OAUTH_ACCESS_TOKEN")
    if token:
        return {"token": token, "mode": "bearer"}
    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    default = os.path.expanduser(
        "~/.config/gcloud/application_default_credentials.json")
    for candidate in (path, default):
        if not candidate or not os.path.exists(candidate):
            continue
        try:
            with open(candidate) as fh:
                blob = json.load(fh)
        except (OSError, ValueError):
            continue
        if blob.get("type") == "authorized_user" and blob.get("refresh_token"):
            return {"refresh": blob, "mode": "refresh", "path": candidate}
        if blob.get("type") == "service_account":
            out = {"mode": "needs-sdk", "path": candidate,
                   "why": ("a service-account key signs a JWT with RS256, "
                           "which this package does not implement. Use an HMAC "
                           "key pair (gcloud storage hmac create), a bearer "
                           "token in GOOGLE_OAUTH_ACCESS_TOKEN, or install "
                           "lmsluice[gcs]")}
    return out


def azure_credentials() -> dict:
    """Azure credentials: a connection string, an account plus key, or a SAS."""
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if conn:
        parts = {}
        for piece in conn.split(";"):
            k, _, v = piece.partition("=")
            if k:
                parts[k.strip().lower()] = v.strip()
        if parts.get("accountname") and parts.get("accountkey"):
            return {"account": parts["accountname"],
                    "key": parts["accountkey"], "mode": "shared-key"}
        if parts.get("sharedaccesssignature"):
            return {"sas": parts["sharedaccesssignature"], "mode": "sas"}
    account = os.environ.get("AZURE_STORAGE_ACCOUNT")
    key = os.environ.get("AZURE_STORAGE_KEY")
    if account and key:
        return {"account": account, "key": key, "mode": "shared-key"}
    sas = os.environ.get("AZURE_STORAGE_SAS_TOKEN")
    if sas:
        return {"sas": sas.lstrip("?"), "mode": "sas"}
    return {}


# -- URLs ------------------------------------------------------------------

def _endpoint(var: str, default: str) -> str:
    host = os.environ.get(var) or default
    if "://" not in host:
        host = "https://" + host
    return host.rstrip("/")


def s3_url(bucket: str, key: str, *, region: str) -> tuple[str, str]:
    """(https URL, region). Virtual-hosted style on AWS, path style elsewhere.

    Path style for a custom endpoint because MinIO, Ceph and a great many
    S3-compatibles are deployed on a bare host or an IP where a bucket
    subdomain does not resolve, and it is the form that works in both places.
    """
    custom = os.environ.get("LMSLUICE_S3_ENDPOINT")
    if custom:
        return f"{_endpoint('LMSLUICE_S3_ENDPOINT', '')}/{bucket}/{key}", region
    host = (f"{bucket}.s3.amazonaws.com" if region in ("us-east-1", "", None)
            else f"{bucket}.s3.{region}.amazonaws.com")
    return f"https://{host}/{key}", region or "us-east-1"


# -- sources ---------------------------------------------------------------

class S3Source(HttpSource):
    """An object in S3, or in anything that speaks S3.

    A thin subclass and not a rewrite: `HttpSource` already owns the parts that
    are hard -- one kept-alive connection per thread, the retry on a connection
    the far end closed while idle, and settling whether ranges work by asking
    for one byte rather than trusting `Accept-Ranges`. All this adds is a
    signature per request.
    """

    cloud = "s3"

    def __init__(self, url: str, *, profile: str | None = None, **kw):
        parsed = urllib.parse.urlparse(url)
        bucket, key = parsed.netloc, parsed.path.lstrip("/")
        if not bucket or not key:
            raise CloudError(f"{url}: expected s3://bucket/key")
        creds = aws_credentials(profile)
        self.region = creds.get("region") or "us-east-1"
        https, self.region = s3_url(bucket, key, region=self.region)
        self.bucket, self.key = bucket, key
        self.anonymous = not (creds.get("access_key") and creds.get("secret_key"))
        self.mode = "stdlib"
        signer = None
        if not self.anonymous:
            def signer(method, target, headers, _c=creds):
                return sign.sigv4_headers(
                    method=method, url=target, region=self.region,
                    service="s3", access_key=_c["access_key"],
                    secret_key=_c["secret_key"], token=_c.get("token"),
                    headers=headers)
        super().__init__(https, signer=signer, **kw)
        self.name = url


class GcsSource(HttpSource):
    """An object in Google Cloud Storage.

    Two stdlib modes, and the first is the surprise: GCS's **XML API accepts
    SigV4**, so an HMAC key pair makes this the S3 path with a different host
    and no separate protocol at all. The second is a bearer token on the same
    XML endpoint, which needs no signing.
    """

    cloud = "gcs"

    def __init__(self, url: str, **kw):
        parsed = urllib.parse.urlparse(url)
        bucket, key = parsed.netloc, parsed.path.lstrip("/")
        if not bucket or not key:
            raise CloudError(f"{url}: expected gs://bucket/key")
        creds = gcs_credentials()
        self.bucket, self.key = bucket, key
        host = _endpoint("LMSLUICE_GCS_ENDPOINT", "https://storage.googleapis.com")
        https = f"{host}/{bucket}/{key}"
        self.mode = "stdlib"
        self.auth = creds.get("mode", "anonymous")
        self.anonymous = self.auth == "anonymous"
        signer, headers = None, dict(kw.pop("headers", None) or {})
        if self.auth == "hmac":
            def signer(method, target, hdrs, _c=creds):
                return sign.sigv4_headers(
                    method=method, url=target, region="auto", service="s3",
                    access_key=_c["access_key"], secret_key=_c["secret_key"],
                    headers=hdrs)
        elif self.auth == "bearer":
            headers["Authorization"] = f"Bearer {creds['token']}"
        elif self.auth == "refresh":
            headers["Authorization"] = f"Bearer {_refresh_token(creds['refresh'])}"
            self.auth = "bearer (refreshed)"
        elif self.auth == "needs-sdk":
            raise CloudError(f"{url}: {creds['why']}")
        super().__init__(https, signer=signer, headers=headers, **kw)
        self.name = url


class AzureSource(HttpSource):
    """A blob in Azure Storage.

    Shared Key or a SAS. The range header is `x-ms-range` rather than `Range`,
    which is the only structural difference from the other two and the reason
    `HttpSource` takes the name as a parameter.
    """

    cloud = "azure"

    def __init__(self, url: str, **kw):
        parsed = urllib.parse.urlparse(url)
        account = parsed.netloc
        path = parsed.path.lstrip("/")
        if not account or "/" not in path:
            raise CloudError(f"{url}: expected az://account/container/blob")
        creds = azure_credentials()
        if creds.get("account") and creds["account"] != account:
            # The URL names the account; a connection string for a different
            # one is a configuration mistake worth saying out loud rather than
            # a 403 twenty seconds later.
            creds = {}
        self.account, self.blob = account, path
        host = _endpoint("LMSLUICE_AZURE_ENDPOINT",
                         f"https://{account}.blob.core.windows.net")
        https = f"{host}/{path}"
        self.mode = "stdlib"
        self.auth = creds.get("mode", "anonymous")
        self.anonymous = self.auth == "anonymous"
        signer = None
        if self.auth == "shared-key":
            def signer(method, target, hdrs, _c=creds):
                return sign.shared_key_header(
                    method=method, url=target, account=_c["account"],
                    key=_c["key"], headers=hdrs)
        elif self.auth == "sas":
            https = f"{https}?{creds['sas']}"
        super().__init__(https, range_header="x-ms-range", signer=signer, **kw)
        self.name = url


def _refresh_token(blob: dict) -> str:
    """Mint an access token from a `gcloud auth application-default` refresh
    token. An HTTPS POST and a JSON field -- no SDK, no RSA, no JWT."""
    import urllib.error
    import urllib.request

    body = urllib.parse.urlencode({
        "client_id": blob["client_id"], "client_secret": blob["client_secret"],
        "refresh_token": blob["refresh_token"], "grant_type": "refresh_token",
    }).encode()
    req = urllib.request.Request(
        blob.get("token_uri", "https://oauth2.googleapis.com/token"),
        data=body, headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["access_token"]
    except Exception as exc:              # noqa: BLE001
        raise CloudError(redact_env(
            f"could not refresh the Google credential: {exc}")) from None


SCHEMES = {"s3": S3Source, "gs": GcsSource, "gcs": GcsSource, "az": AzureSource,
           "azure": AzureSource, "abfs": AzureSource, "abfss": AzureSource}


def is_cloud(target: str) -> bool:
    return urllib.parse.urlparse(str(target)).scheme in SCHEMES


def open_cloud(target: str, **kw) -> Source:
    """A `Source` for a bucket URL, wrapped if the store will not serve ranges."""
    from .source import CachedSource

    scheme = urllib.parse.urlparse(target).scheme
    cls = SCHEMES.get(scheme)
    if cls is None:
        raise CloudError(f"{target}: not an object-store URL")
    try:
        src = cls(target, **kw)
    except CloudError:
        raise
    except Exception as exc:              # noqa: BLE001
        raise CloudError(redact_env(f"{target}: {type(exc).__name__}: {exc}")) \
            from None
    if getattr(src, "head_error", None) is not None:
        # Neither the HEAD nor the one-byte GET reached the object. A source of
        # size zero would be returned here and fail on the first read with a
        # message about the archive, so the connection error is raised where it
        # happened -- with the credentials taken out of it.
        why = redact_env(f"{type(src.head_error).__name__}: {src.head_error}")
        auth = describe(src)
        src.close()
        raise CloudError(
            f"{target}: could not read the object ({why}). Authenticated as "
            f"{auth}." + ("" if not getattr(src, "anonymous", False) else
                          " No credentials were found, so the request was "
                          "anonymous; a private object needs them."))
    return src if src.random_access else CachedSource(src)


def describe(src) -> str:
    """One line naming the store, the mode and how it authenticated."""
    cloud = getattr(src, "cloud", None)
    if cloud is None:
        return "not an object store"
    auth = getattr(src, "auth", "anonymous" if getattr(src, "anonymous", True)
                   else "signed")
    return f"{cloud} · {getattr(src, 'mode', 'stdlib')} · {auth}"


# -- writing ---------------------------------------------------------------
#
# The upload side, and it is deliberately last: the download is the case
# `strategy.md` §2b calls the best arithmetic in the repository -- a model
# arriving over a link slower than the decoder arrives faster compressed, with
# no measurement needed to know the sign. Uploading is the mirror and rarer.
#
# All three stores chunk a large upload the same way: begin, send numbered
# parts, then commit a manifest naming them. S3 calls it a multipart upload;
# GCS's XML API implements the same one, so it is again the same code; Azure
# calls the parts blocks and the manifest a block list. What actually differs
# is two URLs and one XML vocabulary.

# 8 MiB. Above S3's 5 MiB floor for every part but the last, and a size where
# the per-part round trip is amortised without holding much memory. Overridable
# because the right value is a property of the link, not of this file.
PART_BYTES = 8 << 20


def _xml_text(body: bytes, tag: str) -> str:
    import xml.etree.ElementTree as ET

    root = ET.fromstring(body)
    for el in root.iter():
        if el.tag.rsplit("}", 1)[-1] == tag:
            return (el.text or "").strip()
    raise CloudError(f"no <{tag}> in the store's reply")


def _send(src, method: str, url: str, *, body: bytes = b"",
          headers: dict | None = None, sha: str | None = None) -> tuple:
    """One signed request with a body. Returns (status, headers, body).

    Separate from `HttpSource.pread` because a write is not a read with a
    payload: SigV4 signs the SHA-256 of the body, so an upload cannot reuse the
    empty-payload hash every read uses, and getting that wrong is a 403 that
    looks like a credential problem.
    """
    import hashlib
    import http.client

    parsed = urllib.parse.urlparse(url)
    digest = sha if sha is not None else hashlib.sha256(body).hexdigest()
    out = {**(headers or {}), "Content-Length": str(len(body))}
    signer = getattr(src, "_signer", None)
    if signer is not None:
        try:
            out = signer(method, url, out, digest)
        except TypeError:                 # a signer that takes no payload hash
            out = signer(method, url, out)
    else:
        out = {**src.headers, **out}
    cls = (http.client.HTTPSConnection if parsed.scheme == "https"
           else http.client.HTTPConnection)
    conn = cls(parsed.netloc, timeout=src.timeout)
    try:
        path = urllib.parse.urlunparse(
            ("", "", parsed.path or "/", parsed.params, parsed.query, ""))
        conn.request(method, path, body=body, headers=out)
        r = conn.getresponse()
        return r.status, dict(r.getheaders()), r.read()
    finally:
        conn.close()


def _expect(status: int, body: bytes, what: str, url: str) -> None:
    if 200 <= status < 300:
        return
    raise CloudError(redact_env(
        f"{what} failed with HTTP {status} for {url}: "
        f"{body[:400].decode('utf-8', 'replace')}"))


def put(local_path: str, target: str, *, part_bytes: int = PART_BYTES,
        progress=None, **kw) -> dict:
    """Upload a local file to a bucket URL. Returns a small report.

    Chunked for every store, single-request for a file that fits in one part --
    a 4 MB archive should not pay three round trips to say so.
    """
    scheme = urllib.parse.urlparse(target).scheme
    if scheme not in SCHEMES:
        raise CloudError(f"{target}: not an object-store URL")
    size = os.path.getsize(local_path)
    src = _writer(target, **kw)
    try:
        if src.cloud == "azure":
            report = _put_azure(src, local_path, size, part_bytes, progress)
        else:
            report = _put_s3(src, local_path, size, part_bytes, progress)
    finally:
        src.close()
    report.update(bytes=size, target=target, store=describe(src))
    return report


class _Writer:
    """A destination: the signed URL and the credentials, with no HEAD.

    `S3Source` and friends probe the object at construction to learn its size
    and whether ranges work, which is exactly wrong for a target that does not
    exist yet. This shares their signing and nothing else.
    """

    def __init__(self, cloud, url, signer, headers, timeout=60.0):
        self.cloud, self.url, self._signer = cloud, url, signer
        self.headers, self.timeout = dict(headers or {}), timeout
        self.mode, self.auth = "stdlib", "signed" if signer else "anonymous"
        self.anonymous = signer is None and not headers

    def close(self):
        pass


def _writer(target: str, *, profile: str | None = None, **_kw) -> _Writer:
    parsed = urllib.parse.urlparse(target)
    scheme = parsed.scheme
    if scheme == "s3":
        creds = aws_credentials(profile)
        region = creds.get("region") or "us-east-1"
        url, region = s3_url(parsed.netloc, parsed.path.lstrip("/"),
                             region=region)
        if not (creds.get("access_key") and creds.get("secret_key")):
            raise CloudError(f"{target}: writing needs credentials; none found")

        def signer(method, u, hdrs, sha=sign.EMPTY_SHA256, _c=creds):
            return sign.sigv4_headers(
                method=method, url=u, region=region, service="s3",
                access_key=_c["access_key"], secret_key=_c["secret_key"],
                token=_c.get("token"), headers=hdrs, payload_sha256=sha)

        return _Writer("s3", url, signer, None)
    if scheme in ("gs", "gcs"):
        creds = gcs_credentials()
        host = _endpoint("LMSLUICE_GCS_ENDPOINT", "https://storage.googleapis.com")
        url = f"{host}/{parsed.netloc}/{parsed.path.lstrip('/')}"
        if creds.get("mode") == "hmac":
            def signer(method, u, hdrs, sha=sign.EMPTY_SHA256, _c=creds):
                return sign.sigv4_headers(
                    method=method, url=u, region="auto", service="s3",
                    access_key=_c["access_key"], secret_key=_c["secret_key"],
                    headers=hdrs, payload_sha256=sha)

            return _Writer("gcs", url, signer, None)
        token = creds.get("token") or (
            _refresh_token(creds["refresh"]) if creds.get("mode") == "refresh"
            else None)
        if token:
            return _Writer("gcs", url, None, {"Authorization": f"Bearer {token}"})
        raise CloudError(f"{target}: writing needs an HMAC pair or a bearer token")
    creds = azure_credentials()
    account = parsed.netloc
    host = _endpoint("LMSLUICE_AZURE_ENDPOINT",
                     f"https://{account}.blob.core.windows.net")
    url = f"{host}/{parsed.path.lstrip('/')}"
    if creds.get("mode") == "shared-key":
        def signer(method, u, hdrs, _sha=None, _c=creds):
            return sign.shared_key_header(method=method, url=u,
                                          account=_c["account"], key=_c["key"],
                                          headers=hdrs)

        return _Writer("azure", url, signer, None)
    if creds.get("mode") == "sas":
        return _Writer("azure", f"{url}?{creds['sas']}", None, None)
    raise CloudError(f"{target}: writing needs a Shared Key or a SAS")


def _parts(path: str, size: int, part_bytes: int):
    with open(path, "rb") as fh:
        n = 0
        while True:
            block = fh.read(part_bytes)
            if not block:
                return
            n += 1
            yield n, block


def _put_s3(src, path: str, size: int, part_bytes: int, progress) -> dict:
    """Single PUT under one part; S3/GCS multipart above it.

    An aborted multipart leaves parts billed and invisible, so a failure part
    way through cancels the upload rather than leaving it for the user to find
    on a bill. That is the one thing this does beyond the happy path, and it is
    the one that costs money if omitted.
    """
    import hashlib

    if size <= part_bytes:
        with open(path, "rb") as fh:
            body = fh.read()
        status, _, reply = _send(src, "PUT", src.url, body=body,
                                 sha=hashlib.sha256(body).hexdigest())
        _expect(status, reply, "PUT", src.url)
        return {"parts": 1, "method": "single PUT"}

    status, _, reply = _send(src, "POST", f"{src.url}?uploads")
    _expect(status, reply, "CreateMultipartUpload", src.url)
    upload = _xml_text(reply, "UploadId")
    tags = []
    try:
        for number, block in _parts(path, size, part_bytes):
            url = f"{src.url}?partNumber={number}&uploadId={urllib.parse.quote(upload)}"
            status, headers, reply = _send(
                src, "PUT", url, body=block,
                sha=hashlib.sha256(block).hexdigest())
            _expect(status, reply, f"UploadPart {number}", src.url)
            etag = headers.get("ETag") or headers.get("etag")
            if not etag:
                raise CloudError(f"part {number} came back with no ETag")
            tags.append((number, etag))
            if progress:
                progress(min(number * part_bytes, size), size)
        body = ("<CompleteMultipartUpload>"
                + "".join(f"<Part><PartNumber>{n}</PartNumber>"
                          f"<ETag>{t}</ETag></Part>" for n, t in tags)
                + "</CompleteMultipartUpload>").encode()
        url = f"{src.url}?uploadId={urllib.parse.quote(upload)}"
        status, _, reply = _send(src, "POST", url, body=body,
                                 sha=hashlib.sha256(body).hexdigest())
        _expect(status, reply, "CompleteMultipartUpload", src.url)
    except Exception:
        # Parts of an abandoned upload are stored and billed and do not appear
        # in a listing. Cancelling on the way out is the difference between a
        # failed upload and a charge nobody can find.
        try:
            _send(src, "DELETE",
                  f"{src.url}?uploadId={urllib.parse.quote(upload)}")
        except Exception:                 # noqa: BLE001 -- the first error wins
            pass
        raise
    return {"parts": len(tags), "method": "multipart"}


def _put_azure(src, path: str, size: int, part_bytes: int, progress) -> dict:
    """One PUT under 256 MiB; staged blocks and a block list above it.

    Azure needs no cancel: staged blocks that are never committed expire on
    their own, which is the one place its model is kinder than S3's.
    """
    import base64

    base, _, query = src.url.partition("?")
    def at(extra):
        joined = "&".join(x for x in (query, extra) if x)
        return f"{base}?{joined}" if joined else base

    if size <= part_bytes:
        with open(path, "rb") as fh:
            body = fh.read()
        status, _, reply = _send(src, "PUT", at(""), body=body,
                                 headers={"x-ms-blob-type": "BlockBlob"})
        _expect(status, reply, "PUT Blob", base)
        return {"parts": 1, "method": "single PUT"}

    ids = []
    for number, block in _parts(path, size, part_bytes):
        # Block ids must be the same length for every block in one blob, so
        # they are zero-padded before base64 rather than after.
        bid = base64.b64encode(f"{number:08d}".encode()).decode()
        status, _, reply = _send(
            src, "PUT", at(f"comp=block&blockid={urllib.parse.quote(bid)}"),
            body=block)
        _expect(status, reply, f"Put Block {number}", base)
        ids.append(bid)
        if progress:
            progress(min(number * part_bytes, size), size)
    body = ("<?xml version='1.0' encoding='utf-8'?><BlockList>"
            + "".join(f"<Latest>{b}</Latest>" for b in ids)
            + "</BlockList>").encode()
    status, _, reply = _send(src, "PUT", at("comp=blocklist"), body=body,
                             headers={"Content-Type": "application/xml"})
    _expect(status, reply, "Put Block List", base)
    return {"parts": len(ids), "method": "block blob"}
