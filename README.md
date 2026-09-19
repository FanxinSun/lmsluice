# lmsluice — the model transport facilitator

Moves model weights from wherever they are to wherever they are needed, over
the route that is actually faster **on the machine it is running on**. It
decides by measurement rather than assumption, and it says what it decided and
on what numbers. Pure standard library, and nothing about the bytes changes:
every weight arrives byte-identical.

Development priorities: [current plan](docs/strategy.md) — complete multimodal
bundle delivery and consumer readiness, with affordability and niche positioning
assessed independently. These are next-development priorities, not claims of
universal model/engine support. The codec-independent/plain path remains core.

## Who this is for

The decision it makes turns on one comparison — is the decoder faster than the
link it is feeding from — and the two sides of that vary very differently. A
decoder varies by more than a factor of ten once you count what it actually
depends on — codec choice alone is 6.1× on one CPU, and thread count another 8×.
**The link varies by a factor of a hundred**, which is still the bigger term and
still the one that decides.

| link | GB/s | verdict | |
|---|---|---|---|
| NVMe, warm in the host cache | 6.2 | 0.74× — a modest tax | **measured** |
| **9p / network mount** | **0.26** | **1.45× — pays** | **measured** |
| NVMe, first touch | 2.4 | pays | *projected* |
| UFS 4.0 (phone flash) | 4.0 | pays | *projected* |
| SATA SSD | 0.55 | pays, ratio-capped | *projected* |
| eMMC / SD / USB | 0.30 | pays, ratio-capped | *projected* |
| 1 Gb/s network | 0.125 | pays, ratio-capped | *projected* |
| a download | 0.01 – 0.1 | pays, see below | *projected* |

**Two rows are measured end to end; the rest are the gate's arithmetic and are
marked as projections.** They are not measurements and should not be read as
any. The projection is `min(1/f, decode/link)` with decode = 5.95 GB/s measured
through this package's own transport, which puts the crossover at **5.95 GB/s of
link** — every projected row sits below it. The one measured row above the
crossover behaves as predicted in direction and is worse than predicted in
degree: 0.74× where the arithmetic says 0.95×, so about 20% is unaccounted and
is being chased.

So the value is very nearly a function of one variable: **how slow the link
is** — and every link below the crossover above is one where compression is
free or better.

**A correction, because this page used to say otherwise.** It claimed an archive
"costs you 6× on load time" on a fast NVMe. That figure paired the fastest plain
measurement with the slowest coded one, and the coded one used a chunk size that
reaches lmz's conditioned codec — which decodes **6.1× slower** than its field
split on the same data. With the chunk size this package now writes, the same
machine measures **0.74×**: still a tax on a warm local NVMe, but a modest one,
and it still saves 33% of the disk. `MEASURED.md` has both numbers and the
comparison that was wrong.

Note that 8.3 GB/s appears in `MEASURED.md` as the decoder's rate **from RAM**,
with no I/O in it. It is not a crossover and this page does not use it as one:
the crossover is 5.95, measured through the transport that actually does the
work.

**A second correction, still settling.** The destination buffer a load decodes
into cost more than the transport that filled it: `bytearray(n)` zero-fills,
which faults every page eagerly and writes a full pass of zeroes the decoder
immediately overwrites — 0.55 s against 0.31 s of transport on a 1.87 GB
checkpoint. It is now a private anonymous mapping advised for huge pages, which
is 4.5× cheaper and needs nothing installed. **The end-to-end figures on this
page predate that change and are pessimistic by up to ~2× on this box.** They
are left as they are rather than restated, because the replacements have only
been taken page-cache-warm and this page's numbers are cold-protocol ones; the
two are not interchangeable and quoting across them is how the 6× above
happened. No gate verdict moves either way — both routes paid the allocation
equally. `MEASURED.md` has the fault counts and the arithmetic.

**Measured, not asserted** — a 1 GiB BF16 model over a 9p mount, cache dropped
before every run, byte-identical. That mount is one sample of the sub-1-GB/s
class and not a target; the finding is the link rate, not the filesystem:

| | plain | coded | | runs |
|---|---|---|---|---|
| loading | 0.447 GB/s | **0.646** | **1.45×**, against a 1.49× ceiling | 5 |
| saving | 4.50 s | **3.55 s** | **1.27×** | 3 |

The win grows with model size and saturates by about a gigabyte; below a
hundred megabytes fixed costs eat most of it. `MEASURED.md` has the sizes, the
spread, and the two measurement traps that would have inflated both numbers.

## Downloading a model

The largest case, and the only one where the answer needs no measurement.

Over a 10–100 MB/s link the decoder is two orders of magnitude faster, so it
disappears entirely under the transfer and **you simply move fewer bytes** —
arithmetic, not a gate. It also pays twice, in bytes not transferred *and* in
load time not spent, at lmz's directory-level ratio rather than its shard-level
one, because a checkpoint directory ships the same tensors more than once:

    lmsluice get https://host/model.lmz ./model.safetensors

Ranged GETs with a connection kept alive per thread, decoded as they land, so
the archive never touches the disk in coded form. A server that refuses ranges
is fetched once rather than refused.

Exercised against a local range-serving server and a range-refusing one, both
byte-identical. **Not yet run over a real wide-area link** — the arithmetic
above is the link speed and the ratio, not a measurement of this path at that
speed.

## Compress on first fetch

The reason you do not have to convert anything first.

    lmsluice cache ./model.safetensors            # would it pay here?
    lmsluice cache ./model.safetensors --build    # then build it

The first load reads the plain file at its normal speed. Afterwards the model is
compressed into a local cache, and every later load takes the coded route —
measured at **1.80× from compression alone** on a slow link, **3.85×** with the
cache on local disk, byte-identical, with nothing published and nothing
installed.

**Nothing installed** is literal: `lmsluice/zstdcodec.py` uses the standard
library's zstd, so a machine with no lmz still gets a cache. On Python 3.10-3.13,
where `compression.zstd` does not exist yet, it falls back to deflate — which
works, round-trips byte-identically and compresses a little less well (r=0.657
against 0.624 on the same BF16 fixture). `pip install lmsluice[zstd]` buys the
ratio back; nothing needs it to function, and an archive written either way
opens on either. lmz is preferred
where present because it compresses better; it is an upgrade rather than a
prerequisite.

**And it is a decision, not a policy.** On a fast NVMe the right cache entry is
the plain file, and it will tell you so rather than making every later load
slower. Nothing is cached unless asked, nothing is deleted, and an entry bound to
a file that has changed misses rather than serving stale weights.

## Buckets

    lmsluice get s3://my-bucket/model.lmz  ./model.safetensors
    lmsluice get gs://my-bucket/model.lmz  ./model.safetensors
    lmsluice get az://account/container/model.lmz ./model.safetensors

S3, Google Cloud Storage and Azure Blob, **with no SDK**. All three authorise an
HTTPS range GET with HMAC-SHA256 over a canonical form of the request, and
`hmac` and `hashlib` are in the standard library, so the whole of it is
`lmsluice/sign.py` — AWS Signature V4, Azure Shared Key, SAS tokens and bearer
tokens. `boto3` pulls botocore, s3transfer, jmespath, python-dateutil and
urllib3; on a phone, a CI container with no wheel cache or an air-gapped box
that is frequently the whole obstacle.

**All three have been read from for real**, anonymously, from public buckets —
S3, GCS and Azure, with concurrent ranges checked byte-for-byte against the
same bytes read contiguously (`MEASURED.md` has the objects and the 4.72 MB it
cost). Signed requests against a real store are the one thing still unverified:
they need a bucket and credentials.

**Signing is checked against each vendor's own published example**, not against
a fake I also wrote. A signature is byte-exact or worthless, and a wrong
canonical form produces a well-formed header that is always rejected with no
clue which of a dozen rules was misread — so the tests pin AWS's worked
`GET Object` signature to the hex digit and Microsoft's documented
string-to-sign byte for byte. Round trips then go through a local fake that
re-derives every signature it receives.

**Every command works on a bucket URL and none of them was changed.** `get`,
`info`, `load`, `stream`, `plan` and `bench` never asked what a source is —
they ask how fast it delivers — so an object store is a third implementation of
`pread` beside a local file and a web server. That is the test of the
abstraction rather than a claim about it. S3-compatible stores (MinIO,
Cloudflare R2, Backblaze B2, Ceph RGW, Wasabi) need no code of their own: set
`LMSLUICE_S3_ENDPOINT`.

**Credentials come from where each cloud already puts them** — environment,
`~/.aws/credentials` with profiles, a connection string, a SAS, `gcloud`'s
application-default refresh token — so a configured machine needs nothing new,
and **a public bucket needs nothing at all**. No flag anywhere takes a secret,
because argv is readable by every process on the machine out of `/proc`, and
every credential-bearing header and every signature in a URL is redacted from
every error. A SAS token *is* a signature carried in the query string, which
makes a URL itself a credential and error messages the place they leak from.

**The SDKs are extras, and narrower than they look.** `lmsluice[s3]`,
`[gcs]`, `[azure]` are for credential *sources* the standard library cannot
reach — instance-metadata and IAM roles, service-account JWTs needing RS256,
managed identity. The request is signed and sent by this package either way,
and each source reports its mode as `stdlib` or `sdk`.

**Uploading works too**, and is deliberately the second half:

    lmsluice put ./model.lmz s3://my-bucket/model.lmz

All three stores chunk a large upload the same way — begin, send numbered
parts, commit a manifest naming them. S3 calls it a multipart upload, GCS's XML
API implements the same one, and Azure calls the parts blocks and the manifest
a block list; what differs is two URLs and one XML vocabulary. A part-way
failure **cancels the upload** rather than leaving it, because the parts of an
abandoned multipart are stored, billed, and absent from a listing — a charge
whose cause is hard to find later.

Signing costs 12.7 µs per request for SigV4 and 5.9 µs for Shared Key
(9800X3D, Python 3.14). At the 4 MiB default chunk that is a ceiling of ~330
GB/s, which is to say free against any link a bucket is reached over — and the
number says where it would not be: very small chunks over a very fast one.

## Encryption at rest

A checkpoint on a shared filesystem, in a bucket, or on a laptop that leaves the
building is readable by anyone who can read the file.

    lmsluice seal --make-key key.bin              # 32 random bytes, mode 600
    lmsluice seal model.lmz model.sealed --key-file key.bin
    LMSLUICE_KEY_FILE=key.bin lmsluice get model.sealed out.safetensors

AES-256-GCM, authenticated, **with nothing installed**: OpenSSL's libcrypto is
reached through `ctypes`, the same way `cuda.py` reaches the CUDA driver. Any
Python that can open an HTTPS connection has already loaded it. Nothing here
invents cryptography — no keystream from `hashlib`, no XOR, no home-made
construction — and where no library can be reached, encryption reports `none`
and refuses to write a file that would look protected and not be.

**The key is always a path, never a value.** Not a flag holding a key and not an
environment variable holding one: on Linux any process can read another's
command line out of `/proc`, and an environment is inherited by children. The
key is never logged, never printed, and never written into the archive.

**It wraps the archive rather than living inside a codec.** Encryption changes
when the threat model changes, not when the format does, so it sits in the
transport layer as an envelope — which is also the only way it could cover an
lmz archive, whose container lmz writes and this project does not edit. One
mechanism, every codec, and no codec that mentions it; a test enforces that.

**Structure stays readable, and is still protected.** Tensor names, shapes and
the chunk index are in the clear, so `lmsluice info` works for someone who
cannot read the weights and a loader can plan a partial read before it can
decrypt one — tensorizer makes the same trade. They are covered by an HMAC
under a separate subkey, because an attacker who cannot read a weight could
otherwise still edit the index that says where the weights are. A wrong key is
refused when the file is opened; a tampered unit is refused by its tag and
named; nothing half-authentic is ever returned.

**What it costs, measured rather than assumed** (9800X3D, 16 logical cores,
OpenSSL 3.5.5, 403 MB BF16 checkpoint lmz-coded, warm page cache, best of 5):
**3% of load throughput on one fetch thread, 21% on sixteen.** The stage is
priced in `plan.py` like any other and `limited_by` will name `decrypt` where
it binds. The thread number is the interesting one and it is not about AES:
the same work run across processes instead of threads scales from 15 to 99
GB/s, so what caps a threaded fetch pool is the interpreter, not the cipher.
A sealed archive therefore asks for a *small* fetch depth where an unsealed one
wants a large one, which is the opposite of the usual advice and is why the
default is chosen after the file has said whether it is sealed.

The one thing encryption takes away is `mmap`: there is no arrangement of page
tables that decrypts, so a sealed archive can never take the zero-copy route
however fast the disk is.

**Writing one costs a second pass, and `lmsluice write --seal` says so first.**
The envelope wraps a finished archive, so the file is written and then read
back and rewritten sealed: both copies exist at once. That is the price of
covering an lmz archive without editing lmz, and the plan states both terms —
the extra wall clock as its own phase, and the disk high-water as a column of
its own — so a 70 GB checkpoint sees the cost before it starts rather than when
the volume fills.

## The one ratio

Compression is free on a path when the decoder is faster than the link it is
feeding from. Everything here follows from that. For N plain bytes and an
archive whose coded size is r times its plain size, with the stages overlapped:

    plain    N / link
    coded    max(rN / link, N / decode)
    speedup  min(1/r, decode / link)

The gate is the second term and **it does not mention r at all**. Compression
pays exactly when the codec beats the link; the ratio then decides how much,
up to a ceiling of 1/r. Writing is the same statement with `encode` in place
of `decode`, which is why the answer is often the opposite on the way out.

Three consequences, each of which is easy to get wrong:

- **A fast disk is a reason not to compress.** Above the gate the coded route
  loses, and by more the faster the disk is. A loader that always takes the
  compressed path is choosing to be slower on the machines that would
  otherwise be fastest.
- **The ratio cannot rescue a slow codec.** `min` is a floor, not a sum.
  Saving 40% of the bytes buys nothing where the decoder is already the
  bottleneck.
- **Expanding first is a third route, and it is far worse than both.** Without
  a tool like this, a user with a compressed checkpoint runs `decompress` and
  then opens the result — paying the link twice and a write in between. It is
  priced here so the comparison people actually face is visible.

## Buy me a coffee

lmsluice is free and MIT-licensed. If it has been useful to you, you are warmly
welcome to support its continued development with a coffee. There is no
obligation at all—your interest, feedback, and use of the project already mean
a great deal. Thank you.

### [☕ **Buy me a coffee**](https://buymeacoffee.com/fanxinsun)

If Buy Me a Coffee is not convenient, [Alipay](assets/alipay.jpg) is also
available (打开支付宝，扫一扫). Thank you for helping this work continue.
