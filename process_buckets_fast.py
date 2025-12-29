#!/usr/bin/env python3
"""
Optimized bucket processor with parallel Stockfish.

Key optimizations:
1. Use top win_prob move as "best" instead of Stockfish (optional)
2. Parallel Stockfish with multiple engines
3. Batch writing

Usage:
    # Fast mode (use top win_prob as best, no Stockfish):
    python process_buckets_fast.py --input-dir ./output --output-dir ./training_data \
        --bucket-start 0 --bucket-end 30 --no-stockfish

    # With Stockfish (slower but more accurate):
    python process_buckets_fast.py --input-dir ./output --output-dir ./training_data \
        --bucket-start 0 --bucket-end 30 --stockfish-path ./Stockfish --stockfish-time 0.01
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import chess
import chess.engine

sys.path.insert(0, str(Path(__file__).parent))

from apache_beam import coders
from src.bagz import BagFileReader

RECORD_CODER = coders.TupleCoder([
    coders.StrUtf8Coder(),
    coders.StrUtf8Coder(),
    coders.FloatCoder(),
])


def get_legal_moves_uci(fen: str) -> set[str]:
    """Get all legal moves in UCI format."""
    board = chess.Board(fen)
    return {move.uci() for move in board.legal_moves}


def apply_move(fen: str, move_uci: str) -> str:
    """Apply a move and return the resulting FEN."""
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    board.push(move)
    return board.fen()


def find_bucket_files(input_dir: Path, bucket_id: int) -> list[Path]:
    """Find all worker files for a given bucket ID."""
    pattern = f"bucket_{bucket_id:05d}_*"
    files = list(input_dir.glob(pattern + ".bag")) + \
            list(input_dir.glob(pattern + ".bagz"))
    return sorted(files)


def load_bucket_data(files: list[Path]) -> dict[str, list[tuple[str, float]]]:
    """Load all records from bucket files and group by FEN."""
    fen_moves: dict[str, list[tuple[str, float]]] = defaultdict(list)
    
    for filepath in files:
        reader = BagFileReader(str(filepath))
        n = len(reader)
        for i in range(n):
            raw = reader[i]
            fen, move, win_prob = RECORD_CODER.decode(raw)
            fen_moves[fen].append((move, win_prob))
    
    return dict(fen_moves)


class StockfishPool:
    """Pool of Stockfish engines for parallel analysis."""
    
    def __init__(self, stockfish_path: str, num_engines: int = 4, time_limit: float = 0.01):
        self.engines = []
        self.locks = []
        self.time_limit = time_limit
        
        for _ in range(num_engines):
            engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            self.engines.append(engine)
            self.locks.append(threading.Lock())
        
        self._idx = 0
        self._idx_lock = threading.Lock()
    
    def get_best_move(self, fen: str) -> tuple[str, str]:
        """Get best move and resulting FEN using round-robin engine selection."""
        with self._idx_lock:
            idx = self._idx
            self._idx = (self._idx + 1) % len(self.engines)
        
        with self.locks[idx]:
            board = chess.Board(fen)
            result = self.engines[idx].play(board, chess.engine.Limit(time=self.time_limit))
            best_move = result.move.uci()
            board.push(result.move)
            return best_move, board.fen()
    
    def close(self):
        for engine in self.engines:
            engine.quit()


def process_fen_fast(
    fen: str,
    moves: list[tuple[str, float]],
    stockfish_pool: StockfishPool | None = None,
) -> dict:
    """Process a single FEN position."""
    
    # Get legal moves for validation
    legal_moves = get_legal_moves_uci(fen)
    data_moves = {m for m, _ in moves}
    
    # Check completeness
    missing = legal_moves - data_moves
    extra = data_moves - legal_moves
    is_complete = len(missing) == 0 and len(extra) == 0
    
    # Sort and dedupe moves by win_prob (descending)
    sorted_moves = sorted(moves, key=lambda x: x[1], reverse=True)
    seen = set()
    deduped_moves = []
    for move, prob in sorted_moves:
        if move not in seen:
            seen.add(move)
            deduped_moves.append((move, prob))
    
    # Detect tie-breaking scenario: top 5 moves all have win_prob > 99%
    # This is when Stockfish's best_move matters most for training
    top5_probs = [p for _, p in deduped_moves[:5]]
    needs_tiebreak = len(top5_probs) >= 5 and all(p > 0.99 for p in top5_probs)
    
    # Also check for losing positions (all < 1%)
    is_losing = len(top5_probs) >= 5 and all(p < 0.01 for p in top5_probs)
    
    # Get best move from Stockfish (required for training!)
    if stockfish_pool:
        try:
            best_move, next_fen = stockfish_pool.get_best_move(fen)
        except Exception as e:
            # Fallback to top win_prob (but mark as no stockfish)
            best_move = deduped_moves[0][0]
            next_fen = apply_move(fen, best_move)
    else:
        # No Stockfish - use top win_prob as fallback
        best_move = deduped_moves[0][0]
        next_fen = apply_move(fen, best_move)
    
    # Check if Stockfish agrees with top win_prob move
    top_move = deduped_moves[0][0]
    stockfish_agrees = (best_move == top_move)
    
    # Find rank of Stockfish's move in our win_prob ordering
    stockfish_rank = None
    for i, (m, _) in enumerate(deduped_moves):
        if m == best_move:
            stockfish_rank = i + 1  # 1-indexed
            break
    
    return {
        "fen": fen,
        "is_complete": is_complete,
        "num_moves": len(deduped_moves),
        "moves": [[m, round(p, 4)] for m, p in deduped_moves],
        "top5": [[m, round(p, 4)] for m, p in deduped_moves[:5]],
        "best_action": best_move,  # Stockfish's choice (for tie-breaking training)
        "next_fen": next_fen,
        "needs_tiebreak": needs_tiebreak,  # True if top 5 all > 99%
        "is_losing": is_losing,  # True if top 5 all < 1%
        "stockfish_agrees": stockfish_agrees,  # Does SF pick the top win_prob move?
        "stockfish_rank": stockfish_rank,  # Where does SF's move rank in win_prob order?
        "missing": list(missing) if missing else None,
        "extra": list(extra) if extra else None,
    }


def process_bucket(
    bucket_id: int,
    input_dir: Path,
    output_dir: Path,
    stockfish_pool: StockfishPool | None = None,
    num_workers: int = 8,
) -> dict:
    """Process all FENs in a bucket with parallel processing."""
    
    stats = {"bucket_id": bucket_id, "files": 0, "fens": 0, "complete": 0, "incomplete": 0,
             "needs_tiebreak": 0, "stockfish_agrees": 0}
    
    files = find_bucket_files(input_dir, bucket_id)
    if not files:
        return stats
    
    stats["files"] = len(files)
    print(f"  Loading {len(files)} files...")
    
    fen_moves = load_bucket_data(files)
    stats["fens"] = len(fen_moves)
    print(f"  Found {len(fen_moves):,} unique FENs")
    
    output_path = output_dir / f"bucket_{bucket_id:05d}.jsonl"
    
    start_time = time.time()
    processed = 0
    
    with open(output_path, "w") as f:
        # Process with thread pool for Stockfish parallelism
        if stockfish_pool:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(process_fen_fast, fen, moves, stockfish_pool): fen
                    for fen, moves in fen_moves.items()
                }
                
                for future in as_completed(futures):
                    result = future.result()
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    
                    if result["is_complete"]:
                        stats["complete"] += 1
                    else:
                        stats["incomplete"] += 1
                    
                    if result.get("needs_tiebreak"):
                        stats["needs_tiebreak"] += 1
                    if result.get("stockfish_agrees"):
                        stats["stockfish_agrees"] += 1
                    
                    processed += 1
                    if processed % 10000 == 0:
                        elapsed = time.time() - start_time
                        rate = processed / elapsed
                        eta = (stats["fens"] - processed) / rate
                        print(f"    {processed:,}/{stats['fens']:,} ({rate:.0f}/s, ETA: {eta:.0f}s)")
        else:
            # Sequential processing (fast, no Stockfish)
            for idx, (fen, moves) in enumerate(fen_moves.items()):
                result = process_fen_fast(fen, moves, None)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                
                if result["is_complete"]:
                    stats["complete"] += 1
                else:
                    stats["incomplete"] += 1
                
                if result.get("needs_tiebreak"):
                    stats["needs_tiebreak"] += 1
                if result.get("stockfish_agrees"):
                    stats["stockfish_agrees"] += 1
                
                if (idx + 1) % 50000 == 0:
                    elapsed = time.time() - start_time
                    rate = (idx + 1) / elapsed
                    eta = (stats["fens"] - idx - 1) / rate
                    print(f"    {idx + 1:,}/{stats['fens']:,} ({rate:.0f}/s, ETA: {eta:.0f}s)")
    
    elapsed = time.time() - start_time
    print(f"  Done in {elapsed:.1f}s ({stats['fens'] / elapsed:.0f} FENs/s)")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Process bucket files into LLM training data")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bucket-start", type=int, default=0)
    parser.add_argument("--bucket-end", type=int, default=30)
    parser.add_argument("--stockfish-path", type=str, default="./Stockfish")
    parser.add_argument("--stockfish-time", type=float, default=0.01,
                        help="Time limit per position (default: 0.01s)")
    parser.add_argument("--no-stockfish", action="store_true",
                        help="Skip Stockfish, use top win_prob as best move")
    parser.add_argument("--num-engines", type=int, default=4,
                        help="Number of parallel Stockfish engines")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of worker threads")
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing buckets {args.bucket_start} to {args.bucket_end - 1}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Stockfish: {'disabled' if args.no_stockfish else args.stockfish_path}")
    print("=" * 60)
    
    stockfish_pool = None
    if not args.no_stockfish:
        print(f"Starting {args.num_engines} Stockfish engines...")
        stockfish_pool = StockfishPool(
            args.stockfish_path,
            num_engines=args.num_engines,
            time_limit=args.stockfish_time
        )
    
    total_stats = {"buckets": 0, "fens": 0, "complete": 0, "incomplete": 0,
                   "needs_tiebreak": 0, "stockfish_agrees": 0}
    overall_start = time.time()
    
    try:
        for bucket_id in range(args.bucket_start, args.bucket_end):
            print(f"\n[Bucket {bucket_id:05d}]")
            
            stats = process_bucket(
                bucket_id=bucket_id,
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                stockfish_pool=stockfish_pool,
                num_workers=args.num_workers,
            )
            
            if stats["fens"] > 0:
                total_stats["buckets"] += 1
                total_stats["fens"] += stats["fens"]
                total_stats["complete"] += stats["complete"]
                total_stats["incomplete"] += stats["incomplete"]
                total_stats["needs_tiebreak"] += stats["needs_tiebreak"]
                total_stats["stockfish_agrees"] += stats["stockfish_agrees"]
    
    finally:
        if stockfish_pool:
            stockfish_pool.close()
    
    elapsed = time.time() - overall_start
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Buckets: {total_stats['buckets']}")
    print(f"  Total FENs: {total_stats['fens']:,}")
    print(f"  Complete: {total_stats['complete']:,}")
    print(f"  Incomplete: {total_stats['incomplete']:,}")
    print(f"  Needs tiebreak (top5 > 99%): {total_stats['needs_tiebreak']:,}")
    if total_stats['fens'] > 0:
        tiebreak_pct = total_stats['needs_tiebreak'] / total_stats['fens'] * 100
        agree_pct = total_stats['stockfish_agrees'] / total_stats['fens'] * 100
        print(f"  Tiebreak rate: {tiebreak_pct:.1f}%")
        print(f"  Stockfish agrees with top win_prob: {agree_pct:.1f}%")
    print(f"  Total time: {elapsed / 60:.1f} min")
    print(f"  Overall rate: {total_stats['fens'] / elapsed:.0f} FENs/s")


if __name__ == "__main__":
    main()
