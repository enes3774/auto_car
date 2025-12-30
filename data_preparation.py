"""
Chess MAV Model - Data Preparation Pipeline
Converts JSON chess data to MAV format for training Qwen3 1.7B
"""

import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np

# ============================================================================
# CONSTANTS
# ============================================================================

NUM_VALUE_BUCKETS = 64

# Valid top_n choices (randomly selected during training)
VALID_TOP_N = [3, 5, 10, 20, 30]

# UCI moves - load from file or use provided list
def load_uci_moves(filepath: Optional[str] = None) -> List[str]:
    """
    Load UCI moves from JSON file
    
    Expected format: JSON array of move strings
    ["a1h8", "a1a8", "a1g7", ...]
    """
    if filepath and Path(filepath).exists():
        with open(filepath, 'r') as f:
            moves = f.read().splitlines()
        print(f"Loaded {len(moves)} UCI moves from {filepath}")
        return moves
    else:
        raise FileNotFoundError(
            f"UCI moves file not found: {filepath}\n"
            "Please provide a JSON file with all UCI moves."
        )

# Placeholder - will be set when tokenizer is initialized
ALL_UCI_MOVES: List[str] = []
MOVE_TO_IDX: Dict[str, int] = {}

# ============================================================================
# FEN CONVERSION
# ============================================================================

def fen_to_fixed_length(fen: str) -> str:
    """
    Convert FEN to DeepMind's fixed-length format (77 chars)
    
    Format: {board_64} {active_1} {castling_4} {en_passant_2} {halfmove_2} {fullmove_3}
    Total: 64 + 1 + 4 + 2 + 2 + 3 = 76 chars + 5 spaces = 81 chars with spaces
    """
    parts = fen.split()
    
    # 1. Board: expand digits to dots (64 chars)
    board = parts[0]
    expanded_board = ""
    for char in board:
        if char.isdigit():
            expanded_board += "." * int(char)
        else:
            expanded_board += char
    # Remove slashes, should be 64 chars
    expanded_board = expanded_board.replace("/", "")
    
    # 2. Active player (1 char): w or b
    active = parts[1]
    
    # 3. Castling (4 chars): pad with '.'
    castling = parts[2] if parts[2] != "-" else ""
    castling = castling.ljust(4, ".")
    
    # 4. En passant (2 chars): '-' becomes '-.'
    en_passant = parts[3]
    if en_passant == "-":
        en_passant = "-."
    
    # 5. Halfmove clock (2 chars): pad with '.'
    halfmove = parts[4].rjust(2, ".")
    
    # 6. Fullmove number (3 chars): pad with '.'
    fullmove = parts[5].rjust(3, ".")
    
    return f"{expanded_board} {active} {castling} {en_passant} {halfmove} {fullmove}"


def win_prob_to_bucket(win_prob: float, num_buckets: int = 64) -> int:
    """
    Convert win probability [0, 1] to bucket index [0, num_buckets-1]
    
    Bucket 0 = 0% win, Bucket 63 = ~100% win
    """
    # Clamp to valid range
    win_prob = max(0.0, min(1.0, win_prob))
    # Convert to bucket (bucket 63 for win_prob = 1.0)
    bucket = int(win_prob * (num_buckets - 1) + 0.5)  # Round to nearest
    return min(bucket, num_buckets - 1)


# ============================================================================
# TOKENIZER SETUP
# ============================================================================

class ChessTokenizer:
    """
    Wrapper around base tokenizer with chess-specific tokens
    
    Move tokens use raw UCI format (e.g., "e2e4" not "<m_e2e4>")
    for direct output at inference time.
    """
    
    def __init__(
        self, 
        base_model_name: str = "Qwen/Qwen3-1.7B",
        uci_moves_file: Optional[str] = None,
        uci_moves_list: Optional[List[str]] = None
    ):
        """
        Args:
            base_model_name: HuggingFace model name
            uci_moves_file: Path to JSON file with UCI moves
            uci_moves_list: List of UCI moves (alternative to file)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        
        # Load UCI moves
        if uci_moves_list:
            self.uci_moves = uci_moves_list
        elif uci_moves_file:
            self.uci_moves = load_uci_moves(uci_moves_file)
        else:
            raise ValueError("Must provide either uci_moves_file or uci_moves_list")
        
        # Update global for backward compatibility
        global ALL_UCI_MOVES, MOVE_TO_IDX
        ALL_UCI_MOVES = self.uci_moves
        MOVE_TO_IDX = {move: idx for idx, move in enumerate(self.uci_moves)}
        
        # Store original vocab size
        self.original_vocab_size = len(self.tokenizer)
        
        # Add special tokens
        self._add_chess_tokens()
        
    def _add_chess_tokens(self):
        """Add all chess-specific tokens"""
        from tokenizers import AddedToken
        
        special_tokens = []
        
        # 1. Control tokens for format
        control_tokens = [
            "<|chess|>",      # Game identifier
            "%prev_FEN",      # Previous FEN marker
            "%FEN",           # Current FEN marker  
            "%best_action",   # Best action marker
        ]
        
        # Add %top_n for valid choices only: 3, 5, 10, 20, 30
        for n in VALID_TOP_N:
            control_tokens.append(f"%top_{n}")
        
        for token in control_tokens:
            special_tokens.append(
                AddedToken(token, single_word=True, normalized=False, special=True)
            )
        
        # 2. Value bucket tokens (64 tokens) - keep prefix to avoid number conflicts
        self.value_token_start = len(self.tokenizer) + len(special_tokens)
        for i in range(NUM_VALUE_BUCKETS):
            special_tokens.append(
                AddedToken(f"<val_{i}>", single_word=True, normalized=False, special=True)
            )
        self.value_token_end = self.value_token_start + NUM_VALUE_BUCKETS
        
        # 3. Move tokens - RAW UCI format (no prefix!)
        #    "e2e4", "Nf3", "O-O", "e7e8q" etc.
        #    This allows direct output at inference without post-processing
        self.move_token_start = self.value_token_end
        for move in self.uci_moves:
            special_tokens.append(
                AddedToken(move, single_word=True, normalized=False, special=True)
            )
        self.move_token_end = self.move_token_start + len(self.uci_moves)
        
        # Add all tokens
        self.tokenizer.add_tokens(special_tokens)
        
        # Create lookup dictionaries - moves map to themselves (no prefix)
        self.value_tokens = {i: f"<val_{i}>" for i in range(NUM_VALUE_BUCKETS)}
        self.move_tokens = {move: move for move in self.uci_moves}  # Direct mapping
        
        # Reverse lookups
        self.token_to_value = {f"<val_{i}>": i for i in range(NUM_VALUE_BUCKETS)}
        self.token_to_move = {move: move for move in self.uci_moves}  # Direct mapping
        
        # Token ID lookups for moves
        self.move_to_token_id = {}
        for move in self.uci_moves:
            token_id = self.tokenizer.convert_tokens_to_ids(move)
            self.move_to_token_id[move] = token_id
        
        print(f"Added {len(special_tokens)} special tokens")
        print(f"  Control tokens: {len(control_tokens)} (including %top_n for {VALID_TOP_N})")
        print(f"  Value tokens: {self.value_token_start} - {self.value_token_end} ({NUM_VALUE_BUCKETS} tokens)")
        print(f"  Move tokens: {self.move_token_start} - {self.move_token_end} ({len(self.uci_moves)} tokens)")
        print(f"  Move format: RAW UCI (e.g., 'e2e4', 'Nf3', 'O-O')")
        print(f"Total vocab size: {len(self.tokenizer)}")
    
    def get_value_token(self, bucket: int) -> str:
        """Get value token string for bucket index"""
        return self.value_tokens[bucket]
    
    def get_move_token(self, move: str) -> str:
        """Get move token string for UCI move"""
        return self.move_tokens.get(move, f"<m_{move}>")
    
    def is_value_token(self, token_id: int) -> bool:
        """Check if token ID is a value bucket token"""
        return self.value_token_start <= token_id < self.value_token_end
    
    def is_move_token(self, token_id: int) -> bool:
        """Check if token ID is a move token"""
        return self.move_token_start <= token_id < self.move_token_end
    
    def encode(self, text: str, **kwargs) -> List[int]:
        return self.tokenizer.encode(text, **kwargs)
    
    def decode(self, token_ids: List[int], **kwargs) -> str:
        return self.tokenizer.decode(token_ids, **kwargs)
    
    def __len__(self):
        return len(self.tokenizer)
    
    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)


# ============================================================================
# DATA FORMATTING
# ============================================================================

def format_chess_example(
    data: Dict,
    tokenizer: ChessTokenizer,
    top_n: Optional[int] = None,
    include_next_fen: bool = True,
    shuffle_moves: bool = True
) -> Tuple[str, str, int]:
    """
    Convert JSON data to MAV format strings with RANDOMIZED move order
    
    Key design decisions (from DeepMind paper):
    - Moves are shuffled (not sorted by value) to encourage permutation symmetry
    - If top_n > available moves, we output all available moves
    - This prevents model from learning position-based biases
    
    Args:
        data: JSON dict with fen, moves, top5, best_action, next_fen
        tokenizer: ChessTokenizer instance
        top_n: Number of top moves to include (random from [3,5,10,20,30] if None)
        include_next_fen: Whether to include next FEN in output
        shuffle_moves: Whether to randomize move order (default True, as per paper)
    
    Returns:
        (input_text, output_text, actual_top_n) tuple
    """
    # Get FEN in fixed format
    prev_fen = fen_to_fixed_length(data["fen"])
    
    # Get all available moves with their values
    all_moves = data["moves"]  # List of [move, win_prob]
    num_available_moves = len(all_moves)
    
    # Select top_n from valid choices
    # Key: we can request MORE than available - model learns to stop at legal moves
    if top_n is None:
        top_n = random.choice(VALID_TOP_N)
    
    # Actual number of moves to output (capped by available)
    actual_move_count = min(top_n, num_available_moves)
    
    # Select moves to include (take top by value, then shuffle order)
    selected_moves = all_moves[:actual_move_count]
    best_move =selected_moves[0][0]
    # IMPORTANT: Randomize order to prevent positional bias
    # Paper: "Using randomised ordering of moves in the training data, 
    #         the model is steered towards approximate permutation symmetry"
    if shuffle_moves:
        selected_moves = selected_moves.copy()  # Don't modify original
        random.shuffle(selected_moves)
    
    # Build input - always use the REQUESTED top_n (not actual count)
    # This teaches model that %top_20 might only have 15 moves if that's all legal
    input_text = f"<|chess|> %prev_FEN %top_{top_n} %best_action"
    if include_next_fen:
        input_text += " %FEN"
    input_text += f"\n[%prev_FEN {prev_fen}]"
    
    # Build output - moves in RANDOM order with values
    # Format: "e2e4:<val_57> d2d4:<val_55> Nf3:<val_52>"
    moves_output = []
    for move, win_prob in selected_moves:
        bucket = win_prob_to_bucket(win_prob)
        move_token = tokenizer.get_move_token(move)  # Now returns raw "e2e4"
        value_token = tokenizer.get_value_token(bucket)
        moves_output.append(f"{move_token}:{value_token}")
    
    output_text = f"\n[%top_{top_n} {' '.join(moves_output)}]"
    
    # Best action (from original data, not affected by shuffle)
    # Handle both formats: "g3d6" or ["g3d6", 0.9]
    
    
    best_move_token = tokenizer.get_move_token(best_move)
    output_text += f"\n[%best_action {best_move_token}]"
    
    # Next FEN (if available and requested)
    if include_next_fen and data.get("next_fen"):
        next_fen = fen_to_fixed_length(data["next_fen"])
        output_text += f"\n[%FEN {next_fen}]"
    
    return input_text, output_text, top_n


def format_to_tokens(
    input_text: str,
    output_text: str,
    tokenizer: ChessTokenizer,
    max_length: int = 2048,
    top_n: int = 5
) -> Dict[str, torch.Tensor]:
    """
    Tokenize input/output and create training tensors with proper masking
    
    Returns dict with:
        - input_ids: token IDs
        - attention_mask: attention mask
        - labels: labels with -100 for input tokens
        - value_token_mask: boolean mask for value tokens
        - top_n: the top_n value for this sample (for batching)
    """
    # Tokenize separately
    input_ids = tokenizer.encode(input_text, add_special_tokens=False)
    output_ids = tokenizer.encode(output_text, add_special_tokens=False)
    
    # Combine
    full_ids = input_ids + output_ids
    
    # Truncate if needed
    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
        input_len = len(input_ids)
    else:
        input_len = len(input_ids)
    
    # Create labels: -100 for input, actual IDs for output
    labels = [-100] * input_len + output_ids
    labels = labels[:max_length]
    
    # Create value token mask (for HL-Gauss loss)
    value_token_mask = torch.zeros(len(full_ids), dtype=torch.bool)
    for i, token_id in enumerate(full_ids):
        if tokenizer.is_value_token(token_id):
            value_token_mask[i] = True
    
    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "value_token_mask": value_token_mask,
        "top_n": top_n,  # For batching by top_n
    }


# ============================================================================
# STREAMING DATASET (for large datasets that don't fit in memory)
# ============================================================================

class ChessMAVIterableDataset(torch.utils.data.IterableDataset):
    """
    Streaming Dataset for Chess MAV training - loads one file at a time
    
    Use this when your dataset is too large to fit in memory.
    Each file is loaded, processed, and released before loading the next.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: ChessTokenizer,
        max_length: int = 2048,
        include_next_fen: bool = True,
        shuffle_moves: bool = True,
        shuffle_files: bool = True,
        file_extensions: List[str] = [".jsonl", ".json"],
        samples_per_file: Optional[int] = None,  # Limit samples per file for testing
    ):
        """
        Args:
            data_path: Path to folder containing JSONL/JSON files
            tokenizer: ChessTokenizer instance
            max_length: Maximum sequence length
            include_next_fen: Whether to include next FEN prediction
            shuffle_moves: Randomize move order (recommended)
            shuffle_files: Randomize order of files each epoch
            file_extensions: Extensions to look for
            samples_per_file: Limit samples per file (for testing)
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_next_fen = include_next_fen
        self.shuffle_moves = shuffle_moves
        self.shuffle_files = shuffle_files
        self.samples_per_file = samples_per_file
        
        # Collect all files
        self.files = []
        for ext in file_extensions:
            self.files.extend(self.data_path.glob(f"*{ext}"))
        self.files = sorted(set(self.files))
        
        if not self.files:
            raise FileNotFoundError(f"No files found in {data_path}")
        
        print(f"Found {len(self.files)} files (streaming mode - one at a time)")
        
        # Estimate total samples (optional - for progress tracking)
        self._estimated_samples = None
    
    def _load_file_items(self, filepath: Path):
        """Generator that yields items from a single file"""
        with open(filepath, 'r') as f:
            first_char = f.read(1)
            f.seek(0)
            
            if first_char == '[':
                # JSON array
                items = json.load(f)
                for item in items:
                    if all(k in item for k in ["fen", "moves", "best_action"]):
                        yield item
            else:
                # JSONL
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        if all(k in item for k in ["fen", "moves", "best_action"]):
                            yield item
                    except json.JSONDecodeError:
                        continue
    
    def _process_item(self, item: Dict) -> Dict[str, torch.Tensor]:
        """Process a single item to training format"""
        # Random top_n
        top_n = random.choice(VALID_TOP_N)
        
        # Format to text
        input_text, output_text, actual_top_n = format_chess_example(
            item,
            self.tokenizer,
            top_n=top_n,
            include_next_fen=self.include_next_fen,
            shuffle_moves=self.shuffle_moves
        )
        
        # Tokenize
        return format_to_tokens(
            input_text,
            output_text,
            self.tokenizer,
            self.max_length,
            top_n=top_n
        )
    
    def __iter__(self):
        """Iterate through all files, yielding processed samples"""
        # Get worker info for distributed data loading
        worker_info = torch.utils.data.get_worker_info()
        
        files = self.files.copy()
        
        # Shuffle files each epoch
        if self.shuffle_files:
            random.shuffle(files)
        
        # Split files among workers if using multiple workers
        if worker_info is not None:
            # Split files among workers
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            files = files[worker_id::num_workers]
        
        for filepath in files:
            count = 0
            for item in self._load_file_items(filepath):
                if self.samples_per_file and count >= self.samples_per_file:
                    break
                
                yield self._process_item(item)
                count += 1


def create_streaming_dataloader(
    data_path: str,
    tokenizer: ChessTokenizer,
    batch_size: int = 4,
    max_length: int = 2048,
    num_workers: int = 2,
    prefetch_factor: int = 2,
    samples_per_file: Optional[int] = None,
) -> DataLoader:
    """
    Create a streaming DataLoader for large datasets
    
    Args:
        data_path: Path to folder with JSONL/JSON files
        tokenizer: ChessTokenizer instance
        batch_size: Samples per batch
        max_length: Max sequence length
        num_workers: DataLoader workers (each worker handles different files)
        prefetch_factor: Batches to prefetch per worker
        samples_per_file: Limit samples per file (for testing)
    
    Example:
        # For 93 files × 500MB each
        loader = create_streaming_dataloader(
            "./training_data/",
            tokenizer,
            batch_size=4,
            num_workers=2  # 2 workers load different files in parallel
        )
    """
    dataset = ChessMAVIterableDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        shuffle_moves=True,
        shuffle_files=True,
        samples_per_file=samples_per_file,
    )
    
    collate_fn = lambda batch: chess_collate_fn(
        batch,
        pad_token_id=tokenizer.tokenizer.pad_token_id or 0
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )


# ============================================================================
# DATASET (loads all into memory - use for smaller datasets)
# ============================================================================

class ChessMAVDataset(Dataset):
    """
    PyTorch Dataset for Chess MAV training
    
    Features:
    - LAZY LOADING: Only loads file index upfront, reads data on-demand
    - Supports single JSONL file OR folder with multiple JSONL/JSON files
    - Pre-assigns top_n to each sample for efficient batching
    - Randomizes move order (not sorted by value)
    - Supports k > num_legal_moves (model learns to stop at available moves)
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: ChessTokenizer,
        max_length: int = 2048,
        max_samples: Optional[int] = None,
        include_next_fen: bool = True,
        shuffle_moves: bool = True,
        preassign_top_n: bool = True,
        file_extensions: List[str] = [".jsonl", ".json"],
        lazy_loading: bool = True
    ):
        """
        Args:
            data_path: Path to JSONL file OR folder containing JSONL/JSON files
            tokenizer: ChessTokenizer instance
            max_length: Maximum sequence length
            max_samples: Limit number of samples (for debugging)
            include_next_fen: Whether to include next FEN prediction
            shuffle_moves: Randomize move order (recommended, as per paper)
            preassign_top_n: Pre-assign top_n values for efficient batching
            file_extensions: Extensions to look for when data_path is a folder
            lazy_loading: If True, only index files upfront, load data on-demand
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_next_fen = include_next_fen
        self.shuffle_moves = shuffle_moves
        self.lazy_loading = lazy_loading
        
        self.data = []  # Only used if lazy_loading=False
        self.sample_top_n = []
        
        # For lazy loading: store file info and sample indices
        self.file_index = []  # List of (filepath, start_idx, count, is_json_array)
        self.total_samples = 0
        
        data_path = Path(data_path)
        
        if data_path.is_file():
            print(f"Loading data from single file: {data_path}")
            self._index_file(data_path, max_samples)
        elif data_path.is_dir():
            print(f"Loading data from folder: {data_path}")
            self._index_folder(data_path, file_extensions, max_samples)
        else:
            raise FileNotFoundError(f"Data path not found: {data_path}")
        
        # If not lazy loading, load all data now
        if not lazy_loading:
            print("Loading all data into memory...")
            self._load_all_data()
        
        # Pre-assign top_n values
        if preassign_top_n:
            self._preassign_top_n()
        
        print(f"Total samples: {self.total_samples}")
        print(f"Files indexed: {len(self.file_index)}")
        if lazy_loading:
            print(f"Lazy loading: ON (caches up to 10 files in memory)")
        else:
            print(f"Lazy loading: OFF (all data in memory)")
        print(f"Random top_n from: {VALID_TOP_N}")
        print(f"Move order: {'SHUFFLED (recommended)' if shuffle_moves else 'SORTED by value'}")
        
        if self.sample_top_n:
            from collections import Counter
            dist = Counter(self.sample_top_n)
            print(f"Top_n distribution: {dict(sorted(dist.items()))}")
    
    def _index_file(self, filepath: Path, max_samples: Optional[int] = None):
        """Index a single file (count samples without loading all data)"""
        with open(filepath, 'r') as f:
            first_char = f.read(1)
            f.seek(0)
            
            is_json_array = (first_char == '[')
            
            if is_json_array:
                # JSON array - need to parse to count
                items = json.load(f)
                count = sum(1 for item in items 
                           if all(k in item for k in ["fen", "moves", "best_action"]))
                if max_samples:
                    count = min(count, max_samples - self.total_samples)
            else:
                # JSONL - count lines
                count = 0
                for line in f:
                    if max_samples and self.total_samples + count >= max_samples:
                        break
                    try:
                        item = json.loads(line.strip())
                        if all(k in item for k in ["fen", "moves", "best_action"]):
                            count += 1
                    except:
                        pass
        
        if count > 0:
            self.file_index.append({
                'path': filepath,
                'start_idx': self.total_samples,
                'count': count,
                'is_json_array': is_json_array
            })
            self.total_samples += count
            print(f"  {filepath.name}: {count} samples")
    
    def _index_folder(self, folder_path: Path, file_extensions: List[str], max_samples: Optional[int] = None):
        """Index all files in folder"""
        files = []
        for ext in file_extensions:
            files.extend(folder_path.glob(f"*{ext}"))
        files = sorted(set(files))
        
        if not files:
            raise FileNotFoundError(f"No files found in {folder_path}")
        
        print(f"Found {len(files)} files")
        
        for filepath in files:
            if max_samples and self.total_samples >= max_samples:
                print(f"  Reached max_samples ({max_samples}), stopping indexing.")
                break
            self._index_file(filepath, max_samples)
    
    def _load_all_data(self):
        """Load all data into memory (when lazy_loading=False)"""
        for file_info in self.file_index:
            filepath = file_info['path']
            is_json_array = file_info['is_json_array']
            
            with open(filepath, 'r') as f:
                if is_json_array:
                    items = json.load(f)
                    for item in items:
                        if all(k in item for k in ["fen", "moves", "best_action"]):
                            self.data.append(item)
                else:
                    for line in f:
                        try:
                            item = json.loads(line.strip())
                            if all(k in item for k in ["fen", "moves", "best_action"]):
                                self.data.append(item)
                        except:
                            pass
    
    def _get_item_lazy(self, idx: int) -> Dict:
        """Get item using lazy loading with file caching"""
        # Find which file contains this index
        for file_info in self.file_index:
            if idx < file_info['start_idx'] + file_info['count']:
                local_idx = idx - file_info['start_idx']
                filepath = file_info['path']
                
                # Check if file is cached
                if not hasattr(self, '_file_cache'):
                    self._file_cache = {}
                    self._cache_order = []
                    self._max_cache_size = 10  # Keep max 10 files in memory
                
                cache_key = str(filepath)
                
                if cache_key not in self._file_cache:
                    # Load file into cache
                    items = self._load_file_data(filepath, file_info['is_json_array'])
                    
                    # Manage cache size
                    if len(self._cache_order) >= self._max_cache_size:
                        # Remove oldest cached file
                        oldest = self._cache_order.pop(0)
                        del self._file_cache[oldest]
                    
                    self._file_cache[cache_key] = items
                    self._cache_order.append(cache_key)
                
                return self._file_cache[cache_key][local_idx]
        
        raise IndexError(f"Index {idx} out of range")
    
    def _load_file_data(self, filepath: Path, is_json_array: bool) -> List[Dict]:
        """Load all valid items from a file"""
        items = []
        with open(filepath, 'r') as f:
            if is_json_array:
                all_items = json.load(f)
                for item in all_items:
                    if all(k in item for k in ["fen", "moves", "best_action"]):
                        items.append(item)
            else:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        if all(k in item for k in ["fen", "moves", "best_action"]):
                            items.append(item)
                    except:
                        pass
        return items
    
    def _preassign_top_n(self):
        """Pre-assign top_n values for efficient batching"""
        n_samples = self.total_samples
        pattern = VALID_TOP_N * (n_samples // len(VALID_TOP_N) + 1)
        self.sample_top_n = pattern[:n_samples]
        random.shuffle(self.sample_top_n)
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get raw item
        if self.lazy_loading:
            item = self._get_item_lazy(idx)
        else:
            item = self.data[idx]
        
        # Get pre-assigned top_n
        top_n = self.sample_top_n[idx] if self.sample_top_n else random.choice(VALID_TOP_N)
        
        # Format to text with randomized move order
        input_text, output_text, actual_top_n = format_chess_example(
            item, 
            self.tokenizer,
            top_n=top_n,
            include_next_fen=self.include_next_fen,
            shuffle_moves=self.shuffle_moves
        )
        
        # Tokenize with masks
        return format_to_tokens(
            input_text,
            output_text,
            self.tokenizer,
            self.max_length,
            top_n=top_n
        )
    
    def get_top_n_indices(self) -> Dict[int, List[int]]:
        """Get indices grouped by top_n for batch sampling"""
        from collections import defaultdict
        indices_by_top_n = defaultdict(list)
        
        for idx, top_n in enumerate(self.sample_top_n):
            indices_by_top_n[top_n].append(idx)
        
        return dict(indices_by_top_n)


# ============================================================================
# BATCH SAMPLER (Group by top_n for minimal padding)
# ============================================================================

class TopNBatchSampler:
    """
    Custom batch sampler that groups samples by top_n value
    
    This minimizes padding since samples with same top_n have similar lengths:
    - top_3: ~100 tokens
    - top_30: ~180 tokens
    
    Grouping by top_n means batches have consistent sequence lengths.
    """
    
    def __init__(
        self,
        dataset: ChessMAVDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        """
        Args:
            dataset: ChessMAVDataset with pre-assigned top_n
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle within each top_n group
            drop_last: Whether to drop incomplete batches
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Get indices grouped by top_n
        self.indices_by_top_n = dataset.get_top_n_indices()
        self.top_n_values = list(self.indices_by_top_n.keys())
        
        # Calculate total batches
        self.num_batches = 0
        for top_n, indices in self.indices_by_top_n.items():
            n_batches = len(indices) // batch_size
            if not drop_last and len(indices) % batch_size != 0:
                n_batches += 1
            self.num_batches += n_batches
    
    def __iter__(self):
        # Shuffle order of top_n groups
        top_n_order = self.top_n_values.copy()
        if self.shuffle:
            random.shuffle(top_n_order)
        
        all_batches = []
        
        for top_n in top_n_order:
            indices = self.indices_by_top_n[top_n].copy()
            
            # Shuffle indices within this top_n group
            if self.shuffle:
                random.shuffle(indices)
            
            # Create batches
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)
        
        # Shuffle batches (optional - keeps similar lengths together if False)
        if self.shuffle:
            random.shuffle(all_batches)
        
        for batch in all_batches:
            yield batch
    
    def __len__(self):
        return self.num_batches


# ============================================================================
# COLLATE FUNCTION
# ============================================================================

def chess_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int = 0,
    label_pad_id: int = -100
) -> Dict[str, torch.Tensor]:
    """
    Custom collate function with proper padding
    
    Pads sequences to max length in batch, creates proper masks.
    When batched by top_n, sequences have similar lengths -> minimal padding!
    """
    # Find max length in batch
    max_len = max(item["input_ids"].size(0) for item in batch)
    
    batch_size = len(batch)
    
    # Initialize padded tensors
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), label_pad_id, dtype=torch.long)
    value_token_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    
    # Fill in actual values (right-padding)
    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]
        labels[i, :seq_len] = item["labels"]
        value_token_mask[i, :seq_len] = item["value_token_mask"]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "value_token_mask": value_token_mask,
    }


# ============================================================================
# DATA LOADER FACTORY
# ============================================================================

def create_dataloader(
    data_path: str,
    tokenizer: ChessTokenizer,
    batch_size: int = 4,
    max_length: int = 256,
    shuffle: bool = True,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    group_by_top_n: bool = True
) -> DataLoader:
    """
    Create a DataLoader for chess MAV training
    
    Args:
        data_path: Path to JSONL file OR folder containing JSONL/JSON files
        tokenizer: ChessTokenizer instance
        batch_size: Samples per batch
        max_length: Max sequence length
        shuffle: Whether to shuffle data
        num_workers: DataLoader workers
        max_samples: Limit samples (for debugging)
        group_by_top_n: Use TopNBatchSampler for efficient batching
    
    Example:
        # Single file
        loader = create_dataloader("train.jsonl", tokenizer)
        
        # Folder with multiple files
        loader = create_dataloader("./train_data/", tokenizer)
    """
    dataset = ChessMAVDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_samples,
        shuffle_moves=True,  # Always shuffle move order
        preassign_top_n=group_by_top_n
    )
    
    collate_fn = lambda batch: chess_collate_fn(
        batch, 
        pad_token_id=tokenizer.tokenizer.pad_token_id or 0
    )
    
    if group_by_top_n:
        # Use custom sampler for efficient batching
        batch_sampler = TopNBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False
        )
        
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    else:
        # Standard DataLoader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )


# ============================================================================
# EXAMPLE USAGE & VERIFICATION
# ============================================================================

def verify_data_pipeline():
    """Test the data pipeline with sample data"""
    
    # Sample JSON data (as provided)
    sample_data = [
        {"fen": "4r1k1/6pp/2p2p2/8/8/BP4Q1/4r1PP/2KR4 w - - 0 30", "is_complete": True, "num_moves": 38, "moves": [["g3g4", 0.9027], ["g3d3", 0.9021], ["g3f4", 0.8977], ["g3h4", 0.8943], ["g3c3", 0.894]], "top5": [["g3g4", 0.9027], ["g3d3", 0.9021], ["g3f4", 0.8977], ["g3h4", 0.8943], ["g3c3", 0.894]], "best_action": "g3d6", "next_fen": "4r1k1/6pp/2pQ1p2/8/8/BP6/4r1PP/2KR4 b - - 1 30"},
        {"fen": "r3kbnr/pp2qppp/8/2pp4/3N4/NB1P4/PPPK1PPP/R1BbR3 b kq - 1 10", "is_complete": True, "num_moves": 31, "moves": [["d1g4", 0.766], ["d1h5", 0.7007], ["e8c8", 0.6762]], "top5": [["d1g4", 0.766], ["d1h5", 0.7007], ["e8c8", 0.6762]], "best_action": "d1h5", "next_fen": "r3kbnr/pp2qppp/8/2pp3b/3N4/NB1P4/PPPK1PPP/R1B1R3 w kq - 2 11"},
    ]
    
    print("=" * 60)
    print("VERIFICATION: Data Pipeline Test")
    print("=" * 60)
    
    # Note: This will fail without actual Qwen model, but shows the flow
    print("\n1. FEN Conversion Test:")
    test_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
    converted = fen_to_fixed_length(test_fen)
    print(f"   Original: {test_fen}")
    print(f"   Converted: {converted}")
    print(f"   Length: {len(converted)} chars")
    
    print("\n2. Win Probability to Bucket Test:")
    test_probs = [0.0, 0.25, 0.5, 0.75, 1.0, 0.9027]
    for prob in test_probs:
        bucket = win_prob_to_bucket(prob)
        print(f"   {prob:.4f} -> bucket {bucket}")
    
    print("\n3. UCI Move Generation Test:")
    print(f"   Generated {len(ALL_UCI_MOVES)} unique UCI moves")
    print(f"   Sample moves: {ALL_UCI_MOVES[:10]}")
    
    print("\n4. Format Example (mock):")
    # Create mock format without actual tokenizer
    item = sample_data[0]
    prev_fen = fen_to_fixed_length(item["fen"])
    print(f"   Input FEN: {prev_fen}")
    
    moves_str = []
    for move, prob in item["moves"][:5]:
        bucket = win_prob_to_bucket(prob)
        moves_str.append(f"{move}:<val_{bucket}>")
    print(f"   Top 5 moves: {' '.join(moves_str)}")
    
    best = item['best_action']
    if isinstance(best, list):
        best = best[0]
    print(f"   Best action: {best}")
    
    if item.get("next_fen"):
        next_fen = fen_to_fixed_length(item["next_fen"])
        print(f"   Next FEN: {next_fen}")
    
    print("\n" + "=" * 60)
    print("Verification complete! Pipeline ready for training.")
    print("=" * 60)


# ============================================================================
# DATASET INSPECTION (for debugging and verification)
# ============================================================================

def inspect_sample(
    dataset: ChessMAVDataset,
    idx: int = 0,
    show_tokens: bool = True,
    show_raw: bool = True
) -> Dict:
    """
    Inspect a single sample from the dataset
    
    Args:
        dataset: ChessMAVDataset instance
        idx: Sample index to inspect
        show_tokens: Print token-level details
        show_raw: Print raw text format
    
    Returns:
        Dict with all sample information
    """
    # Get raw data (works with both lazy and non-lazy loading)
    if dataset.lazy_loading:
        raw_item = dataset._get_item_lazy(idx)
    else:
        raw_item = dataset.data[idx]
    
    top_n = dataset.sample_top_n[idx] if dataset.sample_top_n else 5
    
    # Format to text (for inspection)
    input_text, output_text, actual_top_n = format_chess_example(
        raw_item,
        dataset.tokenizer,
        top_n=top_n,
        include_next_fen=dataset.include_next_fen,
        shuffle_moves=dataset.shuffle_moves
    )
    
    # Get tokenized sample
    sample = dataset[idx]
    
    print("=" * 70)
    print(f"SAMPLE INSPECTION - Index {idx}")
    print("=" * 70)
    
    if show_raw:
        print("\n[RAW JSON DATA]")
        print(f"  FEN: {raw_item['fen']}")
        print(f"  Moves: {raw_item['moves'][:5]}{'...' if len(raw_item['moves']) > 5 else ''}")
        print(f"  Best Action: {raw_item['best_action']}")
        print(f"  Next FEN: {raw_item.get('next_fen', 'N/A')[:50]}...")
        
        print("\n[FORMATTED TEXT]")
        print(f"  Assigned top_n: {top_n}")
        print("\n  INPUT:")
        print(f"    {input_text}")
        print("\n  OUTPUT:")
        print(f"    {output_text}")
        
        print("\n[FULL TRAINING SEQUENCE]")
        print("-" * 70)
        print(input_text + output_text)
        print("-" * 70)
    
    if show_tokens:
        print("\n[TOKENIZATION]")
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        value_mask = sample["value_token_mask"]
        
        print(f"  Sequence length: {len(input_ids)}")
        print(f"  Input tokens (masked): {(labels == -100).sum().item()}")
        print(f"  Output tokens (trained): {(labels != -100).sum().item()}")
        print(f"  Value tokens: {value_mask.sum().item()}")
        
        # Decode tokens if tokenizer available
        try:
            tokenizer = dataset.tokenizer
            
            print("\n[TOKEN BREAKDOWN]")
            print("  Format: [position] token_id -> 'decoded' (label | MASKED)")
            print()
            
            for i in range(min(50, len(input_ids))):  # First 50 tokens
                token_id = input_ids[i].item()
                label = labels[i].item()
                is_value = value_mask[i].item()
                
                # Decode token
                decoded = tokenizer.tokenizer.decode([token_id])
                
                # Format label info
                if label == -100:
                    label_str = "MASKED"
                else:
                    label_str = f"label={label}"
                
                # Mark value tokens
                value_str = " [VALUE]" if is_value else ""
                
                print(f"  [{i:3d}] {token_id:6d} -> '{decoded}' ({label_str}){value_str}")
            
            if len(input_ids) > 50:
                print(f"  ... ({len(input_ids) - 50} more tokens)")
        
        except Exception as e:
            print(f"  (Could not decode tokens: {e})")
    
    print("\n[TENSOR SHAPES]")
    for key, value in sample.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape} ({value.dtype})")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    
    return {
        "raw": raw_item,
        "input_text": input_text,
        "output_text": output_text,
        "top_n": top_n,
        "sample": sample
    }


def inspect_batch(
    dataloader,
    batch_idx: int = 0,
    tokenizer = None
) -> Dict:
    """
    Inspect a single batch from the dataloader
    
    Args:
        dataloader: DataLoader instance
        batch_idx: Which batch to inspect
        tokenizer: Optional tokenizer for decoding
    
    Returns:
        The batch dict
    """
    for i, batch in enumerate(dataloader):
        if i == batch_idx:
            print("=" * 70)
            print(f"BATCH INSPECTION - Batch {batch_idx}")
            print("=" * 70)
            
            print("\n[BATCH SHAPES]")
            for key, value in batch.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {value.shape} ({value.dtype})")
            
            print("\n[SEQUENCE LENGTHS IN BATCH]")
            # Calculate actual lengths (non-padded)
            attention_mask = batch["attention_mask"]
            lengths = attention_mask.sum(dim=1)
            print(f"  Min: {lengths.min().item()}")
            print(f"  Max: {lengths.max().item()}")
            print(f"  Mean: {lengths.float().mean().item():.1f}")
            
            print("\n[LABEL STATISTICS]")
            labels = batch["labels"]
            masked = (labels == -100).sum().item()
            total = labels.numel()
            print(f"  Masked (input): {masked} ({100*masked/total:.1f}%)")
            print(f"  Trained (output): {total - masked} ({100*(total-masked)/total:.1f}%)")
            
            print("\n[VALUE TOKEN STATISTICS]")
            value_mask = batch["value_token_mask"]
            print(f"  Value tokens in batch: {value_mask.sum().item()}")
            
            if tokenizer:
                print("\n[FIRST SEQUENCE DECODED]")
                first_seq = batch["input_ids"][0]
                first_labels = batch["labels"][0]
                
                # Find where actual content ends (before padding)
                seq_len = attention_mask[0].sum().item()
                
                print(f"  Length: {seq_len}")
                decoded = tokenizer.tokenizer.decode(first_seq[:seq_len].tolist())
                print(f"  Text: {decoded[:200]}...")
            
            print("\n" + "=" * 70)
            return batch
    
    print(f"Batch {batch_idx} not found!")
    return None


if __name__ == "__main__":
    verify_data_pipeline()
