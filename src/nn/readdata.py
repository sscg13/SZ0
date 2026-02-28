import struct

# <  : Little-endian 
# f  : float (root_q)
# B  : U8 (num_legal_moves)
# B  : U8 (halfmove_clock)
# b  : I8 (outcome)
# 64B: U8[64] (board_tokens)
HEADER_FORMAT = '<f B B b 64B'
HEADER_SIZE = struct.calcsize(HEADER_FORMAT) # 71

def print_canonical_board(tokens):
    piece_map = {
        0: '.', 
        1: 'P', 2: 'B', 3: 'Q', 4: 'N', 5: 'R', 6: 'K',
        7: 'p', 8: 'b', 9: 'q', 10: 'n', 11: 'r', 12: 'k'
    }
    
    print("  +-----------------+")
    for rank in range(7, -1, -1):
        row_str = f"{rank + 1} | "
        for file in range(8):
            sq = rank * 8 + file
            token = tokens[sq]
            row_str += piece_map.get(token, '?') + " "
        row_str += "|"
        print(row_str)
    print("  +-----------------+")
    print("    a b c d e f g h")

def index_to_algebraic(idx):
    from_sq = idx // 64
    to_sq = idx % 64
    
    def sq_to_str(sq):
        file = sq % 8
        rank = sq // 8
        return chr(ord('a') + file) + str(rank + 1)
        
    return f"{sq_to_str(from_sq)}{sq_to_str(to_sq)}"

def inspect_data(filename, num_positions=3):
    print(f"Opening {filename} for inspection...\n")
    
    try:
        with open(filename, 'rb') as f:
            for count in range(num_positions):
                header_bytes = f.read(HEADER_SIZE)
                if len(header_bytes) < HEADER_SIZE:
                    print("End of file reached.")
                    break
                    
                unpacked = struct.unpack(HEADER_FORMAT, header_bytes)
                root_q = unpacked[0]
                num_moves = unpacked[1]
                halfmove_clock = unpacked[2]
                outcome = unpacked[3]
                board_tokens = unpacked[4:68] # Indices 4 through 67 are the board
                
                indices_bytes = f.read(num_moves * 2) # uint16_t
                probs_bytes = f.read(num_moves * 4)   # float
                
                indices = struct.unpack(f'<{num_moves}H', indices_bytes)
                probs = struct.unpack(f'<{num_moves}f', probs_bytes)
                
                print(f"=== Record {count + 1} ===")
                print(f"Outcome (Z)    : {outcome} (1=Win, 0=Draw, -1=Loss)")
                print(f"Search Val (Q) : {root_q:+.4f}")
                print(f"Halfmove Clock : {halfmove_clock}")
                print(f"Legal Moves    : {num_moves}\n")
                
                print("Canonical Board View:")
                print_canonical_board(board_tokens)
                
                print("\nMCTS Policy Target (Top 5 moves):")
                policy = list(zip(indices, probs))
                policy.sort(key=lambda x: x[1], reverse=True)
                
                for idx, prob in policy[:5]:
                    move_str = index_to_algebraic(idx)
                    print(f"  {move_str}: {prob:.4f}")
                if num_moves > 5:
                    print("  ...")
                print("\n" + "="*40 + "\n")
                
    except FileNotFoundError:
        print(f"Error: Could not find {filename}")

if __name__ == "__main__":
    inspect_data("run2.data", 300)