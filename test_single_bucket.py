#!/usr/bin/env python3
"""
Test script - process just ONE bucket to verify everything works.

Usage:
    python test_single_bucket.py --input-dir ./output --bucket-id 0
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

# Check dependencies
try:
    import chess
    import chess.engine
except ImportError:
    print("Install python-chess: pip install python-chess")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))

from apache_beam import coders
from src.bagz import BagFileReader

RECORD_CODER = coders.TupleCoder([
    coders.StrUtf8Coder(),
    coders.StrUtf8Coder(),
    coders.FloatCoder(),
])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--bucket-id", type=int, default=0)
    parser.add_argument("--stockfish-path", type=str, default="/usr/bin/stockfish")
    parser.add_argument("--limit", type=int, default=5, help="Max FENs to process")
    args = parser.parse_args()
    
    # Find bucket files
    pattern = f"bucket_{args.bucket_id:05d}_*"
    files = list(args.input_dir.glob(pattern + ".bag")) + \
            list(args.input_dir.glob(pattern + ".bagz"))
    
    print(f"Found {len(files)} files for bucket {args.bucket_id:05d}:")
    for f in files:
        print(f"  {f.name}")
    
    if not files:
        print("No files found!")
        return
    
    # Load data
    print("\nLoading data...")
    fen_moves = defaultdict(list)
    total_records = 0
    
    for filepath in files:
        reader = BagFileReader(str(filepath))
        n = len(reader)
        print(f"  {filepath.name}: {n} records")
        total_records += n
        
        for i in range(n):
            raw = reader[i]
            fen, move, win_prob = RECORD_CODER.decode(raw)
            fen_moves[fen].append((move, win_prob))
    
    print(f"\nTotal: {total_records} records → {len(fen_moves)} unique FENs")
    
    # Start Stockfish
    print(f"\nStarting Stockfish: {args.stockfish_path}")
    try:
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
    except Exception as e:
        print(f"Failed to start Stockfish: {e}")
        print("Continuing without Stockfish...")
        engine = None
    
    # Process a few FENs
    print(f"\n{'='*60}")
    print(f"Processing first {args.limit} FENs:")
    print('='*60)
    
    for idx, (fen, moves) in enumerate(list(fen_moves.items())[:args.limit]):
        print(f"\n[FEN {idx + 1}]")
        print(f"FEN: {fen}")
        
        # Get legal moves
        board = chess.Board(fen)
        legal_moves = {m.uci() for m in board.legal_moves}
        data_moves = {m for m, _ in moves}
        
        # Sort by win_prob
        sorted_moves = sorted(moves, key=lambda x: x[1], reverse=True)
        
        # Dedupe
        seen = set()
        deduped = []
        for m, p in sorted_moves:
            if m not in seen:
                seen.add(m)
                deduped.append((m, p))
        
        print(f"Legal moves: {len(legal_moves)}")
        print(f"Data moves (unique): {len(deduped)}")
        
        missing = legal_moves - data_moves
        extra = data_moves - legal_moves
        
        if missing:
            print(f"MISSING: {missing}")
        if extra:
            print(f"EXTRA (invalid?): {extra}")
        
        is_complete = len(missing) == 0 and len(extra) == 0
        print(f"Complete: {is_complete}")
        
        # Top 5
        print(f"Top 5: {deduped[:5]}")
        
        # Stockfish
        if engine:
            result = engine.play(board, chess.engine.Limit(time=0.1))
            best_move = result.move.uci()
            board.push(result.move)
            next_fen = board.fen()
            print(f"Stockfish best: {best_move}")
            print(f"Next FEN: {next_fen}")
        
        # Output format preview
        print("\n--- JSONL Output ---")
        output = {
            "fen": fen,
            "is_complete": is_complete,
            "num_moves": len(deduped),
            "moves": [[m, round(p, 4)] for m, p in deduped],
            "top5": [[m, round(p, 4)] for m, p in deduped[:5]],
        }
        if engine:
            output["best_move"] = best_move
            output["next_fen"] = next_fen
        print(json.dumps(output, indent=2))
    
    if engine:
        engine.quit()
    
    print("\n" + "="*60)
    print("Test complete!")


if __name__ == "__main__":
    main()
