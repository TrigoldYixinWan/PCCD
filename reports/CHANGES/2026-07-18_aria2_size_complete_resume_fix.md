# Guard download resume-state fix

**Date:** 2026-07-18
**Level:** Green implementation correction; no scientific outcome inspected

## Problem

`prepare_guard_aria2.py` treated a weight shard as complete whenever its file
size matched the pinned LFS size.  aria2 creates a target-size sparse file while
retaining unfinished byte ranges in a sibling `.aria2` control file.  When a
one-hour Xet signed URL expired, the next invocation therefore skipped an
incomplete shard even though resumable range state remained.  The downstream
SHA-256 verifier correctly failed closed.

## Correction

An exact-size shard is now skipped only when its SHA-256 matches the official
pinned LFS digest.  A hash mismatch with an existing `.aria2` control file is
queued with a refreshed official signed URL so aria2 resumes the missing byte
ranges.  A mismatch without resume state raises an error rather than silently
overwriting or accepting the file.  Stale control state is removed only after
the payload hash is verified.

No repository, revision, model, prompt, label, score, threshold, or hypothesis
changed.  Previously completed shards remain reusable after hash verification.
