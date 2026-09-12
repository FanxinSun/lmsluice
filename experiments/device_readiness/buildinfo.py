"""Git archive metadata used by the dependency-free readiness campaign."""

# ``export-subst`` expands this marker in a Git archive.  A normal checkout
# uses the repository command instead, so the value never becomes a stale
# hardcoded release identifier.
SOURCE_COMMIT = "$Format:%H$"
