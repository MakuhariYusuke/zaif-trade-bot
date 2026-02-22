LFS Migration — 2025-12-13
=========================

Short summary
-------------

We completed a Git LFS migration for the `main` branch on 2025-12-13. This moves a set of large artifacts (model zip files, results, datasets, and checkpoints) into Git LFS, and rewrites the `main` branch history accordingly.

Key points
----------
- Effects: `main` was rewritten and force-pushed to origin. A backup branch and tag were created: `backup-before-lfs-migrate-main-20251213`.
- LFS patterns were added to `.gitattributes`. Files matching those patterns will now be stored in Git LFS on new commits.
- LFS objects were uploaded to the remote and are available for download.

Developer action required
-------------------------
1. Install `git lfs` locally if you don't already have it:
   - Windows / PowerShell
```
git lfs install
```

2. Re-clone the repository (recommended) or reset your local main to match origin:
```
git fetch origin
git checkout main
git reset --hard origin/main
git lfs pull --all
```

Notes and references
--------------------
- Backup branches/tags were created before the rewrite (e.g., `backup-before-lfs-migrate-main-20251213`).
- We updated `.gitignore` to prevent common generated artifacts from being re-added. Please confirm your developer workflows don't write large artifacts to tracked directories.
- If you have CI runners referencing stateful clones, ensure `git lfs` is installed in the runner/agent and `git lfs pull` is executed during checkout.

If you find any important artifacts missing after migration, raise an issue and we'll add them into LFS or move to external storage as needed.

Thank you!

<!-- ci-trigger: whitespace edit -->
