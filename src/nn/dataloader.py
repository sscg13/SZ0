import struct
import numpy as np
import psutil
import os

HEADER_FORMAT = '<f B B b 64B'
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

# The global action space for Chess/Shatranj (64 from-squares * 64 to-squares)
ACTION_SPACE_SIZE = 4096 

def load_sparse_dataset(filepaths):
    move_counts, zs, qs, halfmoves = [], [], [], []
    
    raw_boards = bytearray()
    raw_indices = bytearray()
    raw_probs = bytearray()
    
    print(f"Loading {len(filepaths)} files into memory (Memory Optimized)...")
    
    files_loaded = 0
    
    for filepath in filepaths:
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, 'rb') as f:
            while True:
                header_bytes = f.read(HEADER_SIZE)
                if len(header_bytes) < HEADER_SIZE:
                    break
                
                # unpack_from lets us grab just the first 4 variables without parsing the 64 board bytes into Python ints
                q, num_moves, halfmove, z = struct.unpack_from('<f B B b', header_bytes)
                
                qs.append(q)
                halfmoves.append(halfmove)
                zs.append(z)
                move_counts.append(num_moves)
                
                # The board starts at byte 7 (4 for float + 1 + 1 + 1 for the 3 chars)
                raw_boards.extend(header_bytes[7:71])
                
                indices_bytes = f.read(num_moves * 2)
                probs_bytes = f.read(num_moves * 4)   
                
                raw_indices.extend(indices_bytes)
                raw_probs.extend(probs_bytes)
        
        files_loaded += 1
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info().rss / 1e9  # Convert bytes to GB
        print(f"Loaded file {files_loaded}/{len(filepaths)} | Current RAM: {mem_info:.2f} GB", flush=True)
                
    print(f"Finished loading {len(halfmoves)} positions sparsely!")
    
    # 3. Convert bytearrays directly to 1D NumPy arrays
    # Assuming standard little-endian architecture (which x86/GPUs use natively)
    flat_boards = np.frombuffer(raw_boards, dtype=np.uint8).reshape(-1, 64)
    flat_indices = np.frombuffer(raw_indices, dtype=np.uint16)
    flat_probs = np.frombuffer(raw_probs, dtype=np.float32)
    
    # 4. Create an offsets array for O(1) lookups
    counts_np = np.array(move_counts, dtype=np.int32)
    offsets = np.zeros(len(counts_np) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts_np)
    
    if zs:
        z_array = np.array(zs, dtype=np.float32)
        unique_zs, counts = np.unique(z_array, return_counts=True)
        total_positions = len(zs)
        
        print("\n--- WDL (Z) Distribution ---")
        for z_val, count in zip(unique_zs, counts):
            pct = (count / total_positions) * 100
            print(f"Z = {z_val:>5.2f} : {count:>8} positions ({pct:>5.1f}%)")
        print("----------------------------\n")
                
    return {
        "boards": flat_boards,
        "halfmoves": np.array(halfmoves, dtype=np.float32).reshape(-1, 1),
        "target_z": np.array(zs, dtype=np.float32).reshape(-1, 1),
        "target_q": np.array(qs, dtype=np.float32).reshape(-1, 1),
        
        "pi_indices": flat_indices,
        "pi_probs": flat_probs,
        "pi_offsets": offsets 
    }

class SparseInMemoryDataLoader:
    def __init__(self, dataset_dict, batch_size=284):
        self.data = dataset_dict
        self.batch_size = batch_size
        self.total_samples = self.data['boards'].shape[0]
        
    def get_batches(self):
        # 1. Global permutation for the epoch
        indices = np.random.permutation(self.total_samples)
        square_offsets = np.arange(64, dtype=np.int32) * 13
        
        for i in range(0, self.total_samples, self.batch_size):
            batch_idx = indices[i : i + self.batch_size]
            
            if len(batch_idx) != self.batch_size:
                continue
                
            # 2. On-the-fly Dense Expansion for just this batch
            dense_pi = np.zeros((self.batch_size, 4096), dtype=np.float32)
            
            for batch_row, raw_idx in enumerate(batch_idx):
                # O(1) lookup of where this position's moves start and end
                start_idx = self.data['pi_offsets'][raw_idx]
                end_idx = self.data['pi_offsets'][raw_idx + 1]
                
                # Slice the raw flat arrays
                m_indices = self.data['pi_indices'][start_idx:end_idx]
                m_probs = self.data['pi_probs'][start_idx:end_idx]
                
                # Vectorized scatter (Replaces the slow Python zip loop)
                valid_mask = m_indices < 4096
                dense_pi[batch_row, m_indices[valid_mask]] = m_probs[valid_mask]
                
            raw_boards = self.data['boards'][batch_idx].astype(np.int32)
            psq_boards = raw_boards + square_offsets
            
            yield {
                'boards': psq_boards,
                'halfmoves': self.data['halfmoves'][batch_idx],
                'target_z': self.data['target_z'][batch_idx],
                'target_q': self.data['target_q'][batch_idx],
                'target_pi': dense_pi
            }