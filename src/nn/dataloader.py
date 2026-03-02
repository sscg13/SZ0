import struct
import numpy as np
import os

# Your exact 71-byte C++ struct layout
HEADER_FORMAT = '<f B B b 64B'
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

# The global action space for Chess/Shatranj (64 from-squares * 64 to-squares)
ACTION_SPACE_SIZE = 4096 

def load_sparse_dataset(filepaths):
    boards, zs, qs, halfmoves = [], [], [], []
    sparse_pis = [] # Will hold tuples of (move_indices, move_probs)
    
    print(f"Loading {len(filepaths)} files into memory (Sparse Mode)...")
    
    for filepath in filepaths:
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, 'rb') as f:
            while True:
                header_bytes = f.read(HEADER_SIZE)
                if len(header_bytes) < HEADER_SIZE:
                    break
                
                unpacked = struct.unpack(HEADER_FORMAT, header_bytes)
                qs.append(unpacked[0])
                num_moves = unpacked[1]
                halfmoves.append(unpacked[2])
                zs.append(unpacked[3])
                boards.append(unpacked[4:68])
                
                indices_bytes = f.read(num_moves * 2)
                probs_bytes = f.read(num_moves * 4)   
                
                move_indices = struct.unpack(f'<{num_moves}H', indices_bytes)
                move_probs = struct.unpack(f'<{num_moves}f', probs_bytes)
                
                # Store the sparse data exactly as it comes from C++
                sparse_pis.append((move_indices, move_probs))
                
    print(f"Finished loading {len(boards)} positions sparsely!")
                
    return {
        "boards": np.array(boards, dtype=np.int32),
        "halfmoves": np.array(halfmoves, dtype=np.float32).reshape(-1, 1),
        "target_z": np.array(zs, dtype=np.float32).reshape(-1, 1),
        "target_q": np.array(qs, dtype=np.float32).reshape(-1, 1),
        "sparse_pis": sparse_pis # Note: This remains a regular Python list
    }

class SparseInMemoryDataLoader:
    def __init__(self, dataset_dict, batch_size=512):
        self.data = dataset_dict
        self.batch_size = batch_size
        self.total_samples = self.data['boards'].shape[0]
        
    def get_batches(self):
        # 1. Global permutation for the epoch
        indices = np.random.permutation(self.total_samples)
        
        for i in range(0, self.total_samples, self.batch_size):
            batch_idx = indices[i : i + self.batch_size]
            
            if len(batch_idx) != self.batch_size:
                continue
                
            # 2. On-the-fly Dense Expansion for just this batch
            dense_pi = np.zeros((self.batch_size, 4096), dtype=np.float32)
            
            for batch_row, raw_idx in enumerate(batch_idx):
                m_indices, m_probs = self.data['sparse_pis'][raw_idx]
                
                # Scatter probabilities into the dense array
                for move_id, prob in zip(m_indices, m_probs):
                    if move_id < 4096:
                        dense_pi[batch_row, move_id] = prob
                
            # 3. Yield the perfectly formatted batch
            yield {
                'boards': self.data['boards'][batch_idx],
                'halfmoves': self.data['halfmoves'][batch_idx],
                'target_z': self.data['target_z'][batch_idx],
                'target_q': self.data['target_q'][batch_idx],
                'target_pi': dense_pi
            }