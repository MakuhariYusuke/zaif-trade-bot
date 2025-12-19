"""Find exact and similar duplicate functions/classes across the repo.

Generates a JSON report and a brief Markdown summary.

Usage:
    python tools/find_duplicates.py --root . --out reports/duplicate_report.json --md reports/duplicate_report.md
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import os
import tempfile
import time
import threading
import gc
from collections import defaultdict
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple
import itertools
import multiprocessing
import psutil



@dataclass
class Occurrence:
    path: str
    start: int
    end: int
    name: str
    kind: str  # function or class


def _read_file(path: Path) -> str:
    # Read with fallback encodings and size limits handled by caller
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        try:
            return path.read_text(encoding="latin-1")
        except Exception:
            logging.debug("Failed to read %s with utf-8/latin-1", path)
            return ""


def _normalize_code(src: str) -> str:
    # Remove leading/trailing whitespace, collapse internal whitespace
    lines = [ln.strip() for ln in src.splitlines() if ln.strip()]
    return "\n".join(lines)


def _compare_pair(pair, normalized, threshold):
    hi, hj = pair
    ci = normalized.get(hi, "")
    cj = normalized.get(hj, "")
    if not ci or not cj:
        return None
    if abs(len(ci) - len(cj)) / max(1, min(len(ci), len(cj))) > 0.3:
        return None
    ratio = SequenceMatcher(None, ci, cj).ratio()
    if ratio >= threshold:
        return (hi, hj, ratio)
    return None


def _compute_sig(item, shingle_k, num_perm, timeout_reached):
    h, code = item
    if timeout_reached:
        return h, []
    shingles = _shingle(code, k=shingle_k)
    sig = _minhash_signature(shingles, num_perm=num_perm)
    return h, sig


def _minhash_signature(shingles: set, num_perm: int = 32):
    sig = []
    for i in range(num_perm):
        minv = None
        for sh in shingles:
            h = hashlib.sha1(f"{i}:{sh}".encode("utf-8")).digest()
            v = int.from_bytes(h, "big")
            if minv is None or v < minv:
                minv = v
        sig.append(minv if minv is not None else 0)
    return sig


def _shingle(text: str, k: int = 5):
    # character k-grams
    s = text
    return {s[i : i + k] for i in range(max(0, len(s) - k + 1))}


def _extract_node_source(source: str, node: ast.AST) -> str:
    # ast nodes have lineno and end_lineno in Python3.8+
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None)
    if start is None or end is None:
        return ""
    lines = source.splitlines()
    seg = lines[start - 1 : end]
    # strip possible leading docstring expression
    if seg:
        try:
            first = seg[0].lstrip()
            if first.startswith(('"""', "'''") ):
                # remove up to closing triple quote
                joined = "\n".join(seg)
                # naive strip of first string literal
                # find the first closing triple quote
                for quote in ('"""', "'''"):
                    if joined.startswith(quote):
                        idx = joined.find(quote, len(quote))
                        if idx != -1:
                            joined = joined[idx + len(quote) :]
                        break
                seg = joined.splitlines()
        except Exception:
            pass
    return "\n".join(seg)


def analyze(
    root: Path,
    exts: Tuple[str, ...],
    max_files: int = 0,
    max_bytes: int = 200_000,
    progress_every: int = 200,
    min_lines: int = 1,
) -> Tuple[Dict[str, List[Occurrence]], Dict[str, str]]:
    """Scan Python files under `root` with safety checks.

    - max_files: if >0, stop after scanning that many files (for quick tests)
    - max_bytes: skip files larger than this size (bytes)
    - progress_every: log progress every N files
    """
    functions_by_hash: Dict[str, List[Occurrence]] = defaultdict(list)
    normalized_by_hash: Dict[str, str] = {}


    files_seen = 0
    files_scanned = 0
    start_time = time.time()

    for p in root.rglob("*.py"):
        files_seen += 1
        # skip venv, .git, build artifacts and hidden dirs
        if any(part.startswith(".") for part in p.parts):
            logging.debug("Skipping hidden path %s", p)
            continue
        if "venv" in p.parts or "build" in p.parts or "dist" in p.parts:
            logging.debug("Skipping virtualenv/build path %s", p)
            continue

        try:
            stat = p.stat()
            if stat.st_size > max_bytes:
                logging.info("Skipping large file %s (%d bytes)", p, stat.st_size)
                continue
        except Exception:
            logging.debug("Could not stat file %s", p)

        try:
            src = _read_file(p)
        except Exception:
            logging.exception("Error reading file %s, skipping", p)
            continue

        if not src:
            continue

        try:
            mod = ast.parse(src)
        except SyntaxError:
            logging.debug("SyntaxError parsing %s, skipping", p)
            continue
        except Exception:
            logging.exception("Unexpected parse error for %s", p)
            continue

        files_scanned += 1
        if files_scanned % progress_every == 0:
            logging.info("Scanned %d files, elapsed %.1fs", files_scanned, time.time() - start_time)

        for node in ast.walk(mod):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                try:
                    code = _extract_node_source(src, node)
                    norm = _normalize_code(code)
                    if not norm:
                        continue
                    # Skip trivially short functions/classes to reduce noise
                    lines_count = norm.count("\n") + 1
                    if lines_count < min_lines:
                        logging.debug("Skipping short node %s in %s (%d lines)", getattr(node, "name", "<anon>"), p, lines_count)
                        continue
                    h = hashlib.sha1(norm.encode("utf-8")).hexdigest()
                    normalized_by_hash[h] = norm
                    occ = Occurrence(
                        path=str(p.relative_to(root)),
                        start=node.lineno,
                        end=getattr(node, "end_lineno", node.lineno),
                        name=getattr(node, "name", "<anon>"),
                        kind=type(node).__name__,
                    )
                    functions_by_hash[h].append(occ)
                except Exception:
                    logging.exception("Error extracting node from %s", p)

        if 0 < max_files <= files_scanned:
            logging.info("Reached max_files=%d, stopping scan", max_files)
            break

    logging.info("Finished scanning: seen=%d scanned=%d groups=%d", files_seen, files_scanned, len(functions_by_hash))
    return functions_by_hash, normalized_by_hash


def find_similar(
    normalized: Dict[str, str],
    threshold: float = 0.95,
    max_comparisons: int = 2_000_000,
    use_lsh: bool = False,
    lsh_perm: int = 16,
    lsh_bands: int = 16,
    shingle_k: int = 3,
    timeout_seconds: int = 600,  # 10 minutes default timeout
) -> List[Tuple[str, str, float]]:
    """Compare pairs heuristically by length buckets.

    This can be expensive (O(n^2) within buckets). To avoid long, unbounded
    runs on large repositories we stop after `max_comparisons` pairwise
    comparisons and log progress.
    
    Added timeout to prevent hangs.
    """
    similar = []
    comparisons = 0
    timeout_reached = False
    
    def timeout_handler():
        nonlocal timeout_reached
        timeout_reached = True
        logging.warning("Similarity search timed out after %d seconds", timeout_seconds)
    
    timer = threading.Timer(timeout_seconds, timeout_handler)
    timer.start()
    
    # New LSH and MinHash candidate generation

    def _lsh_candidates(normalized: Dict[str, str], num_perm: int = 32, bands: int = 32, shingle_k: int = 5):
        # Build MinHash signatures and LSH buckets, return set of candidate (h1,h2)
        items = list(normalized.items())
        sigs = {}
        logging.info("Computing MinHash signatures: num_items=%d num_perm=%d", len(items), num_perm)
        process = psutil.Process()
        memory_limit_gb = 4
        
        with multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), 4)) as pool:  # Limit processes
            results = pool.starmap(_compute_sig, [(item, shingle_k, num_perm, timeout_reached) for item in items])
        for h, sig in results:
            if sig:  # not empty
                sigs[h] = sig
                # Check memory periodically
                if len(sigs) % 1000 == 0:
                    mem_gb = process.memory_info().rss / (1024 ** 3)
                    if mem_gb > memory_limit_gb:
                        logging.warning("Memory usage %.2f GB during signature computation, stopping", mem_gb)
                        return set()

        # banding
        rows = max(1, num_perm // bands)
        actual_bands = num_perm // rows
        logging.info("Using LSH bands=%d rows=%d (actual_bands=%d)", bands, rows, actual_bands)
        buckets = defaultdict(list)
        for h, sig in sigs.items():
            if timeout_reached:
                return set()
            for b in range(actual_bands):
                start = b * rows
                end = start + rows
                band_hash = hashlib.sha1(
                    ("|".join(str(x) for x in sig[start:end])).encode("utf-8")
                ).hexdigest()
                buckets[(b, band_hash)].append(h)

        candidates = set()
        for key, lst in buckets.items():
            if timeout_reached:
                return set()
            if len(lst) < 2:
                continue
            for a, b in itertools.combinations(lst, 2):
                if a == b:
                    continue
                pair = tuple(sorted((a, b)))
                candidates.add(pair)
                # Limit candidates to prevent memory explosion
                if len(candidates) > max_comparisons:
                    logging.warning("Too many candidates (%d), limiting to %d", len(candidates), max_comparisons)
                    break
            if len(candidates) > max_comparisons:
                break
        logging.info("LSH produced %d candidate pairs", len(candidates))
        return candidates

    try:
        if use_lsh:
            # generate candidate pairs via LSH
            candidates = _lsh_candidates(normalized, num_perm=lsh_perm, bands=lsh_bands, shingle_k=shingle_k)
            candidate_list = list(candidates)[:max_comparisons]  # Limit to max_comparisons
            
            # Batch processing to reduce memory usage
            batch_size = 10000  # Process in batches of 10k pairs
            similar = []
            process = psutil.Process()
            memory_limit_gb = 4  # Stop if memory usage exceeds 4GB
            
            for i in range(0, len(candidate_list), batch_size):
                if timeout_reached:
                    break
                batch = candidate_list[i:i + batch_size]
                # Check memory usage
                mem_gb = process.memory_info().rss / (1024 ** 3)
                if mem_gb > memory_limit_gb:
                    logging.warning("Memory usage %.2f GB exceeds limit %.2f GB, stopping", mem_gb, memory_limit_gb)
                    break
                logging.info("Processing batch %d-%d (%d pairs), memory: %.2f GB", i, i+len(batch), len(batch), mem_gb)
                
                # Parallel comparison using multiprocessing
                with multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), 4)) as pool:  # Limit processes
                    results = pool.starmap(_compare_pair, [(pair, normalized, threshold) for pair in batch])
                
                batch_similar = [r for r in results if r is not None]
                similar.extend(batch_similar)
                comparisons += len(batch)
                
                # Force garbage collection
                gc.collect()
                
                if comparisons % 100000 == 0:
                    logging.info("Compared %d pairs so far, %d similar found", comparisons, len(similar))
            
            logging.info("Finished similar comparison via LSH: %d candidates compared, %d similar found", comparisons, len(similar))
            return similar

        # Fallback: bucketed pairwise comparisons
        items = list(normalized.items())
        # Use a finer-grained bucket key that mixes character length and line-count
        buckets: Dict[Tuple[int, int], List[Tuple[str, str]]] = defaultdict(list)
        for h, code in items:
            char_bin = len(code) // 40
            lines_bin = (code.count("\n") + 1) // 5
            buckets[(char_bin, lines_bin)].append((h, code))

        for bidx, bucket in enumerate(buckets.values()):
            if timeout_reached:
                logging.warning("Timeout reached during bucket comparison, returning partial results")
                break
            n = len(bucket)
            if n < 2:
                continue
            logging.debug("Processing bucket %d with %d items", bidx, n)
            for i in range(n):
                if timeout_reached:
                    break
                hi, ci = bucket[i]
                for j in range(i + 1, n):
                    if timeout_reached:
                        break
                    comparisons += 1
                    if comparisons % 100000 == 0:
                        logging.info("Compared %d pairs so far", comparisons)
                    if comparisons > max_comparisons:
                        logging.warning("Reached max_comparisons=%d, stopping similar search", max_comparisons)
                        return similar
                    hj, cj = bucket[j]
                    # quick length filter
                    if abs(len(ci) - len(cj)) / max(1, min(len(ci), len(cj))) > 0.3:
                        continue
                    ratio = SequenceMatcher(None, ci, cj).ratio()
                    if ratio >= threshold:
                        similar.append((hi, hj, ratio))
        logging.info("Finished similar comparison: %d pairs compared, %d similar found", comparisons, len(similar))
    finally:
        timer.cancel()
    return similar



def write_reports(out_json: Path, out_md: Path, groups: Dict[str, List[Occurrence]], normalized: Dict[str, str], similar: List[Tuple[str, str, float]]):
    out_json.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "exact_groups": {
            h: [asdict(o) for o in occs] for h, occs in groups.items() if len(occs) > 1
        },
        "similar_pairs": [
            {"h1": h1, "h2": h2, "score": score, "code1_preview": normalized[h1][:400], "code2_preview": normalized[h2][:400]} for (h1, h2, score) in similar
        ],
    }

    # Write JSON atomically
    tmp_json = Path(str(out_json) + ".tmp")
    try:
        tmp_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(str(tmp_json), str(out_json))
        logging.info("Wrote JSON report %s", out_json)
    except Exception:
        logging.exception("Failed to write JSON report to %s", out_json)

    lines = ["# Duplicate Report", "", "## Exact duplicate groups", ""]
    for h, occs in report["exact_groups"].items():
        lines.append(f"- Group {h}: {len(occs)} occurrences")
        for o in occs:
            lines.append(f"  - {o['path']}:{o['start']}-{o['end']} ({o['name']} {o['kind']})")
        lines.append("")

    lines.append("## Similar pairs (>= threshold)")
    lines.append("")
    for p in report["similar_pairs"]:
        lines.append(f"- {p['h1']} ~ {p['h2']} | score={p['score']:.3f}")
        lines.append(f"  - {p['code1_preview'].splitlines()[0] if p['code1_preview'] else ''}")
        lines.append(f"  - {p['code2_preview'].splitlines()[0] if p['code2_preview'] else ''}")
        lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    tmp_md = Path(str(out_md) + ".tmp")
    try:
        tmp_md.write_text("\n".join(lines), encoding="utf-8")
        os.replace(str(tmp_md), str(out_md))
        logging.info("Wrote Markdown report %s", out_md)
    except Exception:
        logging.exception("Failed to write Markdown report to %s", out_md)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Repository root to scan")
    parser.add_argument("--out", default="reports/duplicate_report.json", help="Output JSON report path")
    parser.add_argument("--md", default="reports/duplicate_report.md", help="Output Markdown summary path")
    parser.add_argument("--similarity", type=float, default=0.95, help="Similarity threshold for near-duplicates")
    parser.add_argument(
        "--max-comparisons",
        type=int,
        default=2000000,
        help="Maximum pairwise comparisons in similarity search (safety limit)",
    )
    parser.add_argument("--min-lines", type=int, default=3, help="Skip definitions shorter than this many lines (default: 3)")
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of files scanned (for quick/test runs)")
    parser.add_argument("--max-bytes", type=int, default=200000, help="Skip files larger than this size in bytes")
    parser.add_argument("--progress-every", type=int, default=200, help="Log progress every N files scanned")
    parser.add_argument("--use-lsh", action="store_true", help="Use MinHash+LSH to pre-filter candidate pairs")
    parser.add_argument("--lsh-perm", type=int, default=16, help="Number of MinHash permutations (signature length)")
    parser.add_argument("--lsh-bands", type=int, default=16, help="Number of LSH bands")
    parser.add_argument("--shingle-k", type=int, default=3, help="k size for character shingling used by LSH")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds for similarity search (default: 300)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    root = Path(args.root).resolve()
    try:
        groups, normalized = analyze(
            root,
            (".py",),
            max_files=args.max_files,
            max_bytes=args.max_bytes,
            progress_every=args.progress_every,
            min_lines=args.min_lines,
        )
        similar = find_similar(
            normalized,
            threshold=args.similarity,
            max_comparisons=args.max_comparisons,
            use_lsh=args.use_lsh,
            lsh_perm=args.lsh_perm,
            lsh_bands=args.lsh_bands,
            shingle_k=args.shingle_k,
            timeout_seconds=args.timeout,
        )
        write_reports(Path(args.out), Path(args.md), groups, normalized, similar)
        logging.info("Wrote %s and %s", args.out, args.md)
        logging.info("All done")
        logging.shutdown()
        raise SystemExit(0)
    except KeyboardInterrupt:
        logging.warning("Interrupted by user, partial results may not be written")
    except Exception:
        logging.exception("Unhandled error while running duplicate scanner")
        logging.shutdown()
        raise


if __name__ == "__main__":
    main()
