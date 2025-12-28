#!/usr/bin/env python3
"""
Hash-Bucket Reorganizer - O(n) with no sorting.

Strategy:
  1. Each shard writes to N bucket files based on hash(fen) % N
  2. Merge phase: concatenate bucket partials (no sorting needed!)

Why this is fast:
  - No sorting (O(n) vs O(n log n))
  - No per-FEN file I/O
  - Merge is pure concatenation
  - Embarrassingly parallel

Output guarantee: All records for a given FEN are in the SAME bucket file.
(Multiple FENs share a bucket, but no FEN is split across buckets)

Distributed workflow:
  Machine 1: python reorganize_bucket.py phase1 --shards 0-500 --worker-id m1
  Machine 2: python reorganize_bucket.py phase1 --shards 500-1000 --worker-id m2
  ...
  Final:     python reorganize_bucket.py phase2 --intermediate-dir combined/
"""

import sys
from pathlib import Path
import argparse
import time
import hashlib

sys.path.insert(0, str(Path(__file__).parent))

from apache_beam import coders
from src.bagz import BagFileReader, BagWriter

# =============================================================================
# Configuration
# =============================================================================

# Output includes FEN for simpler processing (trade storage for speed)
# Format: (fen, move, win_prob) - same as input
RECORD_CODER = coders.TupleCoder([
    coders.StrUtf8Coder(),  # fen
    coders.StrUtf8Coder(),  # move
    coders.FloatCoder(),  # win_prob
])


def fen_to_bucket(fen: str, num_buckets: int) -> int:
    """Deterministic hash of FEN to bucket ID."""
    # Using first 8 bytes of MD5 for good distribution
    return int(hashlib.md5(fen.encode()).hexdigest()[:8], 16) % num_buckets


# =============================================================================
# Phase 1: Shard → Bucket Partials (STREAMING - low memory)
# =============================================================================

def run_phase1(
        input_dir: Path,
        output_dir: Path,
        num_buckets: int,
        worker_id: str,
        shard_range: tuple[int, int] | None = None,
        pattern: str = '*.bag',
        compress: bool = False,
        delete_after: bool = False,
):
    """Run phase 1 on a range of shards."""

    input_files = sorted(input_dir.glob(pattern))

    if shard_range:
        start_idx, end_idx = shard_range
        input_files = input_files[start_idx:end_idx]

    print(f"Phase 1: Processing {len(input_files)} shards")
    print(f"  Output: {output_dir}")
    print(f"  Buckets: {num_buckets}")
    print(f"  Worker ID: {worker_id}")
    print(f"  Compress: {compress}")
    print(f"  Delete after: {delete_after}")
    print("=" * 60)

    total_records = 0
    initial_start = time.time()

    ext = '.bagz' if compress else '.bag'
    # Open writers ONCE for all shards
    writers: dict[int, BagWriter] = {}

    def get_writer(bucket_id: int) -> BagWriter:
        if bucket_id not in writers:
            filepath = output_dir / f"bucket_{bucket_id:05d}_{worker_id}{ext}"
            writers[bucket_id] = BagWriter(str(filepath), compress=compress)
        return writers[bucket_id]

    try:
        for idx, input_file in enumerate(input_files):
            print(f"[{idx + 1}/{len(input_files)}] {input_file.name}")

            stats = {'records': 0, 'time': 0.0}
            start = time.time()

            output_dir.mkdir(parents=True, exist_ok=True)

            file_success = False
            try:
                """
                Process a single shard into bucket partials using streaming writes.

                Memory usage: O(num_buckets) for file handles, NOT O(records).
                Each record is written immediately, no buffering.
                """
                reader = BagFileReader(str(input_file))
                num_records = len(reader)

                for i in range(num_records):
                    raw = reader[i]
                    fen, move, win_prob = RECORD_CODER.decode(raw)

                    bucket_id = fen_to_bucket(fen, num_buckets)
                    writer = get_writer(bucket_id)
                    writer.write(raw)  # Write original bytes (already encoded)

                    if (i + 1) % 1_000_000 == 0:
                        print(f"    {i + 1:,}/{num_records:,} records...")

                stats['records'] = num_records
                file_success = True

            except Exception as e:
                print(f"Failure processing {idx}, {e}")

            stats['time'] = time.time() - start
            stats['buckets_written'] = len(writers)

            total_records += stats['records']
            rate = stats['records'] / stats['time'] if stats['time'] > 0 else 0
            print(f"  {stats['records']:,} records in {stats['time']:.1f}s ({rate:,.0f}/s)")
            print(f"  Wrote to {stats['buckets_written']} buckets")

            # Delete source file after successful processing
            if delete_after and file_success:
                input_file.unlink()
                print(f"  Deleted: {input_file.name}")

            # ETA
            elapsed = time.time() - initial_start
            avg_time = elapsed / (idx + 1)
            remaining = len(input_files) - idx - 1
            eta = avg_time * remaining
            print(f"  ETA: {eta / 60:.1f} min remaining")

    # Close all writers
    finally:
        for w in writers.values():
            w.close()

    elapsed = time.time() - initial_start
    print("\n" + "=" * 60)
    print(f"Phase 1 complete: {total_records:,} records in {elapsed / 60:.1f} min")
    print(f"Throughput: {total_records / elapsed:,.0f} records/sec")


# =============================================================================
# Phase 2: Merge Bucket Partials (pure concatenation)
# =============================================================================

def merge_bucket(
        bucket_id: int,
        intermediate_dir: Path,
        output_dir: Path,
        compress_output: bool = True,
) -> dict:
    """
    Merge all partials for a single bucket.

    This is pure concatenation - no sorting needed!
    """
    stats = {'records': 0, 'partials': 0}

    # Find all partials for this bucket (from any worker)
    pattern = f"bucket_{bucket_id:05d}_*"
    partials = list(intermediate_dir.glob(pattern + ".bag")) + \
               list(intermediate_dir.glob(pattern + ".bagz"))

    if not partials:
        return stats

    stats['partials'] = len(partials)

    # Output path
    ext = '.bagz' if compress_output else '.bag'
    output_path = output_dir / f"bucket_{bucket_id:05d}{ext}"

    # Concatenate all partials (just copy bytes, no decode/encode!)
    with BagWriter(str(output_path), compress=compress_output) as writer:
        for partial_path in partials:
            reader = BagFileReader(str(partial_path))
            n = len(reader)
            for i in range(n):
                # Direct byte copy (reader already decompresses if needed)
                writer.write(reader[i])
            stats['records'] += n

    return stats


def run_phase2(
        intermediate_dir: Path,
        output_dir: Path,
        num_buckets: int,
        compress_output: bool = True,
        delete_intermediates: bool = False,
):
    """Merge all bucket partials into final bucket files."""

    print(f"Phase 2: Merging {num_buckets} buckets")
    print(f"  Input: {intermediate_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Compress: {compress_output}")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    total_records = 0
    total_partials = 0
    nonempty_buckets = 0
    start = time.time()

    for bucket_id in range(num_buckets):
        stats = merge_bucket(
            bucket_id=bucket_id,
            intermediate_dir=intermediate_dir,
            output_dir=output_dir,
            compress_output=compress_output,
        )

        if stats['records'] > 0:
            nonempty_buckets += 1
            total_records += stats['records']
            total_partials += stats['partials']

        # Progress every 100 buckets
        if (bucket_id + 1) % 100 == 0:
            elapsed = time.time() - start
            pct = (bucket_id + 1) / num_buckets * 100
            rate = total_records / elapsed if elapsed > 0 else 0
            eta = elapsed / (bucket_id + 1) * (num_buckets - bucket_id - 1)
            print(f"  {bucket_id + 1}/{num_buckets} ({pct:.0f}%) | "
                  f"{total_records:,} records | {rate:,.0f}/s | ETA: {eta:.0f}s")

    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print(f"Phase 2 complete:")
    print(f"  Total records: {total_records:,}")
    print(f"  Non-empty buckets: {nonempty_buckets}/{num_buckets}")
    print(f"  Partials merged: {total_partials}")
    print(f"  Time: {elapsed:.1f}s ({total_records / elapsed:,.0f} records/sec)")

    # Cleanup intermediates
    if delete_intermediates:
        print("\nCleaning up intermediates...")
        count = 0
        for f in intermediate_dir.glob("bucket_*"):
            f.unlink()
            count += 1
        print(f"  Deleted {count} intermediate files")


# =============================================================================
# Single-machine mode (both phases)
# =============================================================================

def run_single_machine(
        input_dir: Path,
        output_dir: Path,
        num_buckets: int,
        pattern: str = '*.bag',
        compress: bool = True,
        delete_after: bool = False,
):
    """Run both phases on a single machine."""

    intermediate_dir = output_dir / '_intermediates'
    final_dir = output_dir / 'buckets'

    # Phase 1
    run_phase1(
        input_dir=input_dir,
        output_dir=intermediate_dir,
        num_buckets=num_buckets,
        worker_id='local',
        pattern=pattern,
        compress=False,  # Don't compress intermediates (speed)
        delete_after=delete_after,
    )

    print("\n")

    # Phase 2
    run_phase2(
        intermediate_dir=intermediate_dir,
        output_dir=final_dir,
        num_buckets=num_buckets,
        compress_output=compress,
        delete_intermediates=True,
    )


# =============================================================================
# Main
# =============================================================================

def parse_range(s: str) -> tuple[int, int]:
    """Parse 'start-end' into (start, end)."""
    parts = s.split('-')
    return int(parts[0]), int(parts[1])


def main():
    parser = argparse.ArgumentParser(
        description='Hash-bucket reorganizer (no sorting!)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single machine (both phases):
  python reorganize_bucket.py single --input-dir ./data --output-dir ./output --buckets 1000

  # Distributed - Phase 1 on multiple machines:
  python reorganize_bucket.py phase1 --input-dir ./data --output-dir ./intermediate \\
      --buckets 1000 --worker-id machine1 --shards 0-500

  # Distributed - Phase 2 (merge) on one machine:
  python reorganize_bucket.py phase2 --intermediate-dir ./all_intermediates \\
      --output-dir ./final --buckets 1000
        """
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    # Single machine
    p_single = subparsers.add_parser('single', help='Run both phases on single machine')
    p_single.add_argument('--input-dir', type=Path, required=True)
    p_single.add_argument('--output-dir', type=Path, required=True)
    p_single.add_argument('--buckets', type=int, default=1000,
                          help='Number of output buckets (default: 1000)')
    p_single.add_argument('--pattern', default='*.bag')
    p_single.add_argument('--compress', action='store_true',
                          help='Compress final output')
    p_single.add_argument('--delete-after', action='store_true',
                          help='Delete source files after processing')

    # Phase 1
    p_phase1 = subparsers.add_parser('phase1', help='Shard → bucket partials')
    p_phase1.add_argument('--input-dir', type=Path, required=True)
    p_phase1.add_argument('--output-dir', type=Path, required=True)
    p_phase1.add_argument('--buckets', type=int, default=1000)
    p_phase1.add_argument('--worker-id', type=str, required=True,
                          help='Unique ID for this worker (e.g., machine1)')
    p_phase1.add_argument('--shards', type=str, default=None,
                          help='Shard range to process, e.g., "0-500"')
    p_phase1.add_argument('--pattern', default='*.bag')
    p_phase1.add_argument('--compress', action='store_true',
                          help='Compress intermediate outputs')
    p_phase1.add_argument('--delete-after', action='store_true',
                          help='Delete source files after processing')

    # Phase 2
    p_phase2 = subparsers.add_parser('phase2', help='Merge bucket partials')
    p_phase2.add_argument('--intermediate-dir', type=Path, required=True)
    p_phase2.add_argument('--output-dir', type=Path, required=True)
    p_phase2.add_argument('--buckets', type=int, default=1000)
    p_phase2.add_argument('--compress', action='store_true',
                          help='Compress final output')
    p_phase2.add_argument('--delete-intermediates', action='store_true')

    args = parser.parse_args()

    if args.command == 'single':
        run_single_machine(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            num_buckets=args.buckets,
            pattern=args.pattern,
            compress=args.compress,
            delete_after=args.delete_after,
        )

    elif args.command == 'phase1':
        shard_range = parse_range(args.shards) if args.shards else None
        run_phase1(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            num_buckets=args.buckets,
            worker_id=args.worker_id,
            shard_range=shard_range,
            pattern=args.pattern,
            compress=args.compress,
            delete_after=args.delete_after,
        )

    elif args.command == 'phase2':
        run_phase2(
            intermediate_dir=args.intermediate_dir,
            output_dir=args.output_dir,
            num_buckets=args.buckets,
            compress_output=args.compress,
            delete_intermediates=args.delete_intermediates,
        )


if __name__ == '__main__':
    main()
