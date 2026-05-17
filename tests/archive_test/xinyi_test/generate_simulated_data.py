import os
import numpy as np
import pandas as pd

OUTPUT_DIR = "/Users/xinyi/Projects/sequenzo/sequenzo/data_and_output/orignal data/different_len_states_data"

def generate_dataset_np(length, num_states, U, N, filename):
    # Calculate the number of unique sequences required
    num_unique = max(1, int(round(N * (U / 100.0))))
    
    # State names like S1, S2, ..., Sk
    states_arr = np.array([f"S{i}" for i in range(1, num_states + 1)])
    
    unique_seqs = set()
    unique_list = []
    
    # 1. Generate unique sequences
    batch_size = num_unique
    while len(unique_seqs) < num_unique:
        # Generate random indices corresponding to the states
        batch = np.random.randint(0, num_states, size=(batch_size, length), dtype=np.int16)
        for row in batch:
            # Use raw bytes for fast hashing
            t = row.tobytes()
            if t not in unique_seqs:
                unique_seqs.add(t)
                unique_list.append(row)
                if len(unique_seqs) == num_unique:
                    break
        
        # Increase batch size slightly for remaining if collisions occur
        remaining = num_unique - len(unique_seqs)
        batch_size = max(remaining, min(100, num_unique))

    unique_arr = np.array(unique_list, dtype=np.int16)
    
    # 2. Duplicate to fill up to total N
    remaining_seqs = N - num_unique
    if remaining_seqs > 0:
        # Randomly choose from existing unique sequences to fill the rest
        indices = np.random.randint(0, num_unique, size=remaining_seqs)
        duplicates = unique_arr[indices]
        total_arr = np.vstack([unique_arr, duplicates])
    else:
        total_arr = unique_arr
        
    # 3. Shuffle all sequences
    np.random.shuffle(total_arr)
    
    # 4. Save to CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    chunk_size = 5000
    columns = [f"T{i}" for i in range(1, length + 1)]
    
    # Write in chunks to prevent huge memory spikes for very large files
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)
            chunk_arr = total_arr[start_idx:end_idx]
            
            # Map indices to string states
            chunk_str = states_arr[chunk_arr]
            
            df_chunk = pd.DataFrame(chunk_str, columns=columns)
            # Insert ID column (1-based index)
            df_chunk.insert(0, "id", range(start_idx + 1, end_idx + 1))
            
            # Write headers only for the first chunk
            if start_idx == 0:
                df_chunk.to_csv(f, header=True, index=False)
            else:
                df_chunk.to_csv(f, header=False, index=False)

def run_experiment_1():
    print("="*40)
    print("Running Experiment 1: Fixed States, Varying Length")
    print("="*40)
    states = 10
    lengths = [10, 30, 50, 100, 300, 500, 1000, 3000]
    Us = [5, 25, 50, 85]
    Ns = [500, 1000, 5000, 10000, 30000, 40000, 45000, 50000]
    
    total = len(lengths) * len(Us) * len(Ns)
    count = 1
    for length in lengths:
        for U in Us:
            for N in Ns:
                filename = f"len_{length}_states_10_U_{U}_N_{N}.csv"
                print(f"[Exp1 {count}/{total}] Generating {filename}...")
                generate_dataset_np(length, states, U, N, filename)
                count += 1

def run_experiment_2():
    print("="*40)
    print("Running Experiment 2: Fixed Length, Varying States")
    print("="*40)
    length = 30
    states_list = [3, 5, 8, 12, 15, 20, 30]
    Us = [5, 25, 50, 85]
    Ns = [500, 1000, 5000, 10000, 30000, 40000, 45000, 50000]
    
    total = len(states_list) * len(Us) * len(Ns)
    count = 1
    for states in states_list:
        for U in Us:
            for N in Ns:
                filename = f"len_30_states_{states}_U_{U}_N_{N}.csv"
                print(f"[Exp2 {count}/{total}] Generating {filename}...")
                generate_dataset_np(length, states, U, N, filename)
                count += 1

if __name__ == "__main__":
    run_experiment_1()
    print("\n")
    run_experiment_2()
    print("\nAll data generation complete!")
