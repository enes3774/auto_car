#!/usr/bin/env python3
"""
Process bucket files into LLM training data.

For each FEN:
1. Collect all (move, win_prob) pairs from all worker files
2. Validate: do we have ALL legal moves? (using python-chess)
3. Run Stockfish → get best_action
4. Apply best_action → get resulting FEN
5. Output structured record

Usage:
    python process_buckets.py --input-dir ./output --output-dir ./training_data \
        --bucket-start 0 --bucket-end 30 --stockfish-path /usr/bin/stockfish
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import NamedTuple
import chess
import chess.engine

sys.path.insert(0, str(Path(__file__).parent))

from apache_beam import coders
from src.bagz import BagFileReader

# Same coder as reorganize_bucket.py
RECORD_CODER = coders.TupleCoder([
    coders.StrUtf8Coder(),  # fen
    coders.StrUtf8Coder(),  # move
    coders.FloatCoder(),    # win_prob
])


class FENData(NamedTuple):
    """Processed data for a single FEN position."""
    fen: str
    moves: list[tuple[str, float]]  # (move_uci, win_prob) sorted by win_prob desc
    is_complete: bool               # True if all legal moves have values
    missing_moves: list[str]        # Legal moves without values
    extra_moves: list[str]          # Moves in data that aren't legal (errors)
    best_move_stockfish: str        # Stockfish's recommendation
    next_fen: str                   # Board after best_move_stockfish


def get_legal_moves_uci(fen: str) -> set[str]:
    """Get all legal moves in UCI format for a FEN position."""
    board = chess.Board(fen)
    return {move.uci() for move in board.legal_moves}


def apply_move(fen: str, move_uci: str) -> str:
    """Apply a move and return the resulting FEN."""
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    board.push(move)
    return board.fen()


def get_stockfish_best_move(
    fen: str, 
    engine: chess.engine.SimpleEngine,
    time_limit: float = 0.1
) -> str:
    """Get Stockfish's best move for a position."""
    board = chess.Board(fen)
    result = engine.play(board, chess.engine.Limit(time=time_limit))
    return result.move.uci()


def find_bucket_files(input_dir: Path, bucket_id: int) -> list[Path]:
    """Find all worker files for a given bucket ID."""
    pattern = f"bucket_{bucket_id:05d}_*"
    files = list(input_dir.glob(pattern + ".bag")) + \
            list(input_dir.glob(pattern + ".bagz"))
    return sorted(files)


def load_bucket_data(files: list[Path]) -> dict[str, list[tuple[str, float]]]:
    """
    Load all records from bucket files and group by FEN.
    
    Returns: {fen: [(move, win_prob), ...]}
    """
    fen_moves: dict[str, list[tuple[str, float]]] = defaultdict(list)
    
    for filepath in files:
        reader = BagFileReader(str(filepath))
        n = len(reader)
        
        for i in range(n):
            raw = reader[i]
            fen, move, win_prob = RECORD_CODER.decode(raw)
            fen_moves[fen].append((move, win_prob))
    
    return dict(fen_moves)


def process_fen(
    fen: str,
    moves: list[tuple[str, float]],
    engine: chess.engine.SimpleEngine,
    stockfish_time: float = 0.1
) -> FENData:
    """Process a single FEN position."""
    
    # Get legal moves
    legal_moves = get_legal_moves_uci(fen)
    data_moves = {m for m, _ in moves}
    
    # Check completeness
    missing = legal_moves - data_moves
    extra = data_moves - legal_moves
    is_complete = len(missing) == 0 and len(extra) == 0
    
    # Sort moves by win_prob (descending)
    sorted_moves = sorted(moves, key=lambda x: x[1], reverse=True)
    
    # Deduplicate (keep highest win_prob for each move)
    seen = set()
    deduped_moves = []
    for move, prob in sorted_moves:
        if move not in seen:
            seen.add(move)
            deduped_moves.append((move, prob))
    
    # Get Stockfish best move
    try:
        best_move = get_stockfish_best_move(fen, engine, stockfish_time)
        next_fen = apply_move(fen, best_move)
    except Exception as e:
        print(f"  Warning: Stockfish failed for {fen[:30]}...: {e}")
        # Fallback to highest win_prob move
        best_move = deduped_moves[0][0] if deduped_moves else ""
        next_fen = apply_move(fen, best_move) if best_move else ""
    
    return FENData(
        fen=fen,
        moves=deduped_moves,
        is_complete=is_complete,
        missing_moves=sorted(missing),
        extra_moves=sorted(extra),
        best_move_stockfish=best_move,
        next_fen=next_fen,
    )


def format_jsonl(data: FENData) -> str:
    """Format as JSON line."""
    return json.dumps({
        "fen": data.fen,
        "is_complete": data.is_complete,
        "num_moves": len(data.moves),
        "moves": [[m, round(p, 4)] for m, p in data.moves],
        "top5": [[m, round(p, 4)] for m, p in data.moves[:5]],
        "best_move": data.best_move_stockfish,
        "next_fen": data.next_fen,
        "missing": data.missing_moves if data.missing_moves else None,
        "extra": data.extra_moves if data.extra_moves else None,
    }, ensure_ascii=False)


def format_text(data: FENData) -> str:
    """Format as human-readable text block for LLM training."""
    moves_str = " ".join(f"{m}:{p:.3f}" for m, p in data.moves)
    top5_str = " ".join(f"{m}:{p:.3f}" for m, p in data.moves[:5])
    
    lines = [
        f"FEN: {data.fen}",
        f"LEGAL_MOVES: {len(data.moves)}",
        f"TOP5: {top5_str}",
        f"ALL_MOVES: {moves_str}",
        f"BEST: {data.best_move_stockfish}",
        f"NEXT_FEN: {data.next_fen}",
        f"COMPLETE: {data.is_complete}",
    ]
    
    if data.missing_moves:
        lines.append(f"MISSING: {' '.join(data.missing_moves)}")
    if data.extra_moves:
        lines.append(f"EXTRA: {' '.join(data.extra_moves)}")
    
    return "\n".join(lines) + "\n---\n"


def process_bucket(
    bucket_id: int,
    input_dir: Path,
    output_dir: Path,
    engine: chess.engine.SimpleEngine,
    output_format: str = "jsonl",
    stockfish_time: float = 0.1,
) -> dict:
    """Process all FENs in a bucket and write output."""
    
    stats = {
        "bucket_id": bucket_id,
        "files": 0,
        "fens": 0,
        "complete": 0,
        "incomplete": 0,
        "total_moves": 0,
    }
    
    # Find all files for this bucket
    files = find_bucket_files(input_dir, bucket_id)
    if not files:
        return stats
    
    stats["files"] = len(files)
    print(f"  Loading {len(files)} files...")
    
    # Load and merge data
    fen_moves = load_bucket_data(files)
    stats["fens"] = len(fen_moves)
    print(f"  Found {len(fen_moves)} unique FENs")
    
    # Output file
    ext = ".jsonl" if output_format == "jsonl" else ".txt"
    output_path = output_dir / f"bucket_{bucket_id:05d}{ext}"
    
    # Process each FEN
    with open(output_path, "w") as f:
        for idx, (fen, moves) in enumerate(fen_moves.items()):
            data = process_fen(fen, moves, engine, stockfish_time)
            
            stats["total_moves"] += len(data.moves)
            if data.is_complete:
                stats["complete"] += 1
            else:
                stats["incomplete"] += 1
            
            # Write output
            if output_format == "jsonl":
                f.write(format_jsonl(data) + "\n")
            else:
                f.write(format_text(data))
            
            # Progress
            if (idx + 1) % 1000 == 0:
                print(f"    Processed {idx + 1}/{len(fen_moves)} FENs...")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Process bucket files into LLM training data")
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory containing bucket_*.bag files")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for training data")
    parser.add_argument("--bucket-start", type=int, default=0,
                        help="First bucket ID to process (inclusive)")
    parser.add_argument("--bucket-end", type=int, default=30,
                        help="Last bucket ID to process (exclusive)")
    parser.add_argument("--stockfish-path", type=str, default="/usr/bin/stockfish",
                        help="Path to Stockfish executable")
    parser.add_argument("--stockfish-time", type=float, default=0.1,
                        help="Time limit for Stockfish per position (seconds)")
    parser.add_argument("--format", choices=["jsonl", "text"], default="jsonl",
                        help="Output format")
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing buckets {args.bucket_start} to {args.bucket_end - 1}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Format: {args.format}")
    print("=" * 60)
    
    # Start Stockfish engine (reuse for all positions)
    print(f"Starting Stockfish: {args.stockfish_path}")
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
    
    total_stats = {
        "buckets": 0,
        "fens": 0,
        "complete": 0,
        "incomplete": 0,
    }
    
    try:
        for bucket_id in range(args.bucket_start, args.bucket_end):
            print(f"\n[Bucket {bucket_id:05d}]")
            
            stats = process_bucket(
                bucket_id=bucket_id,
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                engine=engine,
                output_format=args.format,
                stockfish_time=args.stockfish_time,
            )
            
            if stats["fens"] > 0:
                total_stats["buckets"] += 1
                total_stats["fens"] += stats["fens"]
                total_stats["complete"] += stats["complete"]
                total_stats["incomplete"] += stats["incomplete"]
                
                pct = stats["complete"] / stats["fens"] * 100
                print(f"  Complete: {stats['complete']}/{stats['fens']} ({pct:.1f}%)")
    
    finally:
        engine.quit()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Buckets processed: {total_stats['buckets']}")
    print(f"  Total FENs: {total_stats['fens']}")
    print(f"  Complete: {total_stats['complete']}")
    print(f"  Incomplete: {total_stats['incomplete']}")
    if total_stats['fens'] > 0:
        pct = total_stats['complete'] / total_stats['fens'] * 100
        print(f"  Completeness: {pct:.1f}%")


if __name__ == "__main__":
    main()
