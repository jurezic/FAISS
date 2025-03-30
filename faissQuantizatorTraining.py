# faissQuantizatorTraining.py - UPDATED for ONE PARAMETER FILE for ALL COLLECTIONS
import psycopg2
import yaml
import joblib
import logging
import os
from typing import List, Optional

# --- PQ Libraries ---
import faiss
import numpy as np
import random
import ast # Import ast for safe string evaluation

# Add these new imports for memory monitoring
import psutil
import gc
import time

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load config function (reusing from previous scripts)
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def verify_index_stored(pq_params_filepath):
    try:
        logger.info(f"Loading Faiss Index from: {pq_params_filepath}")
        loaded_index = faiss.read_index(pq_params_filepath)
        logger.info("Faiss Index loaded successfully.")

        if hasattr(loaded_index, 'quantizer') and loaded_index.quantizer is not None:
            trained_pq_quantizer = loaded_index.quantizer
            logger.info("Product Quantizer found in the loaded index.")

            # --- Verification of LOADED Quantizer ---
            logger.info("--- Verification of LOADED Product Quantizer ---")
            logger.info(f"Is PQ Quantizer trained? : {trained_pq_quantizer.is_trained}")
            logger.info(f"Quantizer Dimension (d): {trained_pq_quantizer.d}")
            logger.info(f"Number of Subvectors (M): {trained_pq_quantizer.M}")
            logger.info(f"Bits per Subvector (nbits): {trained_pq_quantizer.nbits}")
            # --- End Verification ---
        else:
            logger.warning("No Product Quantizer found in the loaded index!")
    except Exception as e:
        logger.error(f"Error loading or verifying index: {e}")

def save_pq_quantizer_custom(pq_quantizer, filepath_prefix="pq_quantizer_custom"):
    """
    Custom save method for Faiss ProductQuantizer, saving essential components.
    [Simplified save - using faiss.vector_float_to_array - confirmed working]
    """
    params_filepath = f"{filepath_prefix}_params.joblib"
    codebooks_filepath = f"{filepath_prefix}_codebooks.joblib"
    sdc_table_filepath = f"{filepath_prefix}_sdc_table.joblib"

    logger.info(f"Saving ProductQuantizer components with prefix: {filepath_prefix}")

    # 1. Save Parameters
    params = {
        'd': pq_quantizer.d,
        'M': pq_quantizer.M,
        'nbits': pq_quantizer.nbits,
        'code_size': pq_quantizer.code_size
    }
    joblib.dump(params, params_filepath)
    logger.info(f"  - Saved parameters to: {params_filepath}")

    # 2. Save Codebooks (Centroids) - Convert to NumPy array
    codebooks_np = faiss.vector_float_to_array(pq_quantizer.centroids).reshape((pq_quantizer.M, 2**pq_quantizer.nbits, pq_quantizer.d // pq_quantizer.M)) # Reshape based on PQ structure
    joblib.dump(codebooks_np, codebooks_filepath)
    logger.info(f"  - Saved codebooks (centroids) to: {codebooks_filepath}")

    # 3. Save SDC Table - Convert to NumPy array (if it exists)
    if hasattr(pq_quantizer, 'sdc_table') and pq_quantizer.sdc_table is not None:
        sdc_table_np = faiss.vector_float_to_array(pq_quantizer.sdc_table)
        joblib.dump(sdc_table_np, sdc_table_filepath)
        logger.info(f"  - Saved SDC table to: {sdc_table_filepath}")
    else:
        logger.info("  - No SDC table found to save (may be normal for some PQ configurations).")

def load_pq_quantizer_custom(filepath_prefix="pq_quantizer_custom"):
    """
    Custom load method for Faiss ProductQuantizer, reconstructing from saved components.
    [UPDATED -  REPLACING numpy_to_vector_float with basic reconstruction]
    """
    params_filepath = f"{filepath_prefix}_params.joblib"
    codebooks_filepath = f"{filepath_prefix}_codebooks.joblib"
    sdc_table_filepath = f"{filepath_prefix}_sdc_table.joblib"

    logger.info(f"Loading ProductQuantizer components from prefix: {filepath_prefix}")

    # 1. Load Parameters
    params = joblib.load(params_filepath)
    logger.info(f"  - Loaded parameters from: {params_filepath}")
    d = params['d']
    M = params['M']
    nbits = params['nbits']
    code_size = params['code_size']

    # 2. Load Codebooks (Centroids) - Load NumPy array and convert to Faiss Vector (BASIC METHOD)
    codebooks_np = joblib.load(codebooks_filepath)
    logger.info(f"  - Loaded codebooks from: {codebooks_filepath}")
    codebooks_list = codebooks_np.flatten().tolist() # Flatten NumPy array to a list of floats
    codebooks_faiss = faiss.Float32Vector() # Create an empty Faiss Float32Vector
    for val in codebooks_list: # Iterate through the list and extend the Faiss vector
        codebooks_faiss.push_back(val) # Use push_back to add each float


    # 3. Load SDC Table - Load NumPy array and convert to Faiss Vector (BASIC METHOD - if saved)
    sdc_table_faiss = None
    try:
        sdc_table_np = joblib.load(sdc_table_filepath)
        logger.info(f"  - Loaded SDC table from: {sdc_table_filepath}")
        sdc_table_list = sdc_table_np.tolist() # Flatten NumPy array to list of floats
        sdc_table_faiss = faiss.Float32Vector() # Create an empty Faiss Float32Vector
        for val in sdc_table_list: # Iterate and extend
            sdc_table_faiss.push_back(val) # Use push_back
    except FileNotFoundError:
        logger.info("  - SDC table file not found, assuming it was not saved (may be normal).")

    # 4. Reconstruct ProductQuantizer
    pq_quantizer = faiss.ProductQuantizer(d, M, nbits)
    pq_quantizer.codebooks = codebooks_faiss # Set the loaded codebooks

    if sdc_table_faiss is not None: # Restore SDC table if loaded
        pq_quantizer.sdc_table = sdc_table_faiss

    logger.info(f"ProductQuantizer reconstructed from custom saved components.")
    return pq_quantizer


def apply_product_quantization(embeddings: List[List[float]], pq_num_subvectors: int = 8, pq_bits_per_subvector: int = 8):
    """
    Applies Product Quantization (PQ) to a list of float embeddings using Faiss.
    (Same PQ function as before - for reuse)
    """
    if not embeddings:
        return None, []

    # Convert list of lists to NumPy float32 array (Faiss expects float32)
    float_embeddings_np = np.array(embeddings, dtype=np.float32)
    dimension = float_embeddings_np.shape[1]
    n_vectors = float_embeddings_np.shape[0]

    # --- 1. Train the Product Quantizer ---
    pq_quantizer = faiss.ProductQuantizer(
        dimension,
        pq_num_subvectors,
        pq_bits_per_subvector
    )

    logger.info(f"Training Product Quantizer with {pq_num_subvectors} subvectors and {pq_bits_per_subvector} bits/subvector...")
    pq_quantizer.train(float_embeddings_np) # Train on your embeddings data

    logger.info("--- Verification AFTER PQ Training ---")
    logger.info(f"Quantizer Dimension (d): {pq_quantizer.d}")
    logger.info(f"Number of Subvectors (M): {pq_quantizer.M}")
    logger.info(f"Bits per Subvector (nbits): {pq_quantizer.nbits}")

    # --- 2. Encode the Embeddings --- (Not needed in training script, but kept for function consistency if reused elsewhere)
    logger.info("Encoding embeddings using trained Product Quantizer...")
    pq_codes_np = pq_quantizer.compute_codes(float_embeddings_np)

    # --- 3. Convert PQ codes to bytes for storage (BYTEA in PostgreSQL) --- (Not needed in training script, but kept for function consistency)
    # Faiss PQ codes are NumPy arrays of uint8. We can directly convert to bytes.
    pq_encoded_embeddings_bytes = [code.tobytes() for code in pq_codes_np]

    logger.info(f"Product Quantization training completed. Trained on {n_vectors} embeddings, dimension: {dimension}.")

    return pq_quantizer, pq_encoded_embeddings_bytes # Returning pq_encoded_embeddings_bytes even if not used directly in this script


def train_faiss_quantizator(subsample_size=400000, id_batch_size=100000, embedding_batch_size=1000):
    """
    Trains ONE Faiss Product Quantizer for ALL COLLECTIONS combined on a RANDOM SUBSAMPLE of existing embeddings
    in the database using BATCHED ID FETCHING and SUBSAMPLING (Solution 2). Creates ONE parameter file for all collections.
    """
    config = load_config()
    db_config = config['database']

    DB_HOST = db_config['host']
    DB_NAME = db_config['name']
    DB_USER = db_config['user']
    DB_PASSWORD = db_config['password']
    DB_PORT = db_config['port']
    pq_params_filepath = "pq_quantizer_params_all_collections" # File path for PQ parameters - ONE FILE FOR ALL COLLECTIONS

    conn = None
    try:
        logger.info("Starting Faiss Product Quantizer training script for ALL COLLECTIONS (ONE PARAMETER FILE) with BATCHED ID FETCHING and SUBSAMPLING (Solution 2)...") # Updated log message
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
        cur = conn.cursor()

        all_ids = []
        offset = 0
        logger.info("Fetching IDs in batches from ALL COLLECTIONS...")
        while True:
            sql_query_ids = """
                SELECT id
                FROM langchain_pg_embedding
                LIMIT %s OFFSET %s
            """
            cur.execute(sql_query_ids, (id_batch_size, offset))
            id_batch = cur.fetchall()
            if not id_batch:
                break
            all_ids.extend([id_tuple[0] for id_tuple in id_batch])
            offset += id_batch_size
            logger.info(f"  Fetched ID batch, current total IDs from ALL collections: {len(all_ids)}")

        num_total_ids = len(all_ids)
        logger.info(f"Fetched total {num_total_ids} IDs from PGVector (ALL COLLECTIONS).")

        if num_total_ids > subsample_size:
            logger.info(f"Subsampling IDs for training (ALL COLLECTIONS). Total IDs: {num_total_ids}, Subsample size: {subsample_size}")
            sampled_ids = random.sample(all_ids, subsample_size)
            logger.info(f"Random ID subsampling complete (ALL COLLECTIONS). Sampled {len(sampled_ids)} IDs.")
        else:
            logger.info(f"Total IDs ({num_total_ids}) from ALL COLLECTIONS is less than subsample size ({subsample_size}). Using all IDs.")
            sampled_ids = all_ids

        # Calculate and log estimated memory requirements
        embedding_dim = 1024  # Assuming 1536 dimensions for OpenAI embeddings
        avg_embedding_size_bytes = embedding_dim * 4  # 4 bytes per float32
        estimated_memory_gb = (len(sampled_ids) * avg_embedding_size_bytes) / (1024**3)
        logger.info(f"Estimated memory required for embeddings: {estimated_memory_gb:.2f} GB (assuming {embedding_dim} dimensions)")
        
        # Get current memory usage
        process = psutil.Process(os.getpid())
        current_memory_usage = process.memory_info().rss / (1024**3)
        logger.info(f"Current memory usage: {current_memory_usage:.2f} GB")
        
        # ---- SOLUTION 2: PRE-ALLOCATE NUMPY ARRAY ----
        logger.info(f"Pre-allocating NumPy array for {len(sampled_ids)} embeddings with dimension {embedding_dim}...")
        embeddings_for_training_np = np.zeros((len(sampled_ids), embedding_dim), dtype=np.float32)
        logger.info("NumPy array allocated successfully.")
        
        # Memory after allocation
        current_memory_usage = process.memory_info().rss / (1024**3)
        logger.info(f"Memory usage after allocation: {current_memory_usage:.2f} GB")
        
        # Add progress tracking
        batch_count = (len(sampled_ids) + embedding_batch_size - 1) // embedding_batch_size
        start_time = time.time()
        last_log_time = start_time
        current_idx = 0  # Index to keep track of where to write in the pre-allocated array
        
        logger.info("Fetching embeddings in batches and filling NumPy array (ALL COLLECTIONS)...")
        for i in range(0, len(sampled_ids), embedding_batch_size):
            batch_start_time = time.time()
            current_batch = i // embedding_batch_size + 1
            
            id_batch_for_embedding_fetch = sampled_ids[i:i + embedding_batch_size]
            sql_query_embeddings = """
                SELECT embedding
                FROM langchain_pg_embedding
                WHERE id IN ({})
            """.format(','.join(['%s'] * len(id_batch_for_embedding_fetch)))

            cur.execute(sql_query_embeddings, tuple(id_batch_for_embedding_fetch))
            embedding_batch_tuples = cur.fetchall()
            
            # Process each embedding directly into the NumPy array
            batch_parsed = 0
            for tuple_emb in embedding_batch_tuples:
                embedding_string = tuple_emb[0]
                if embedding_string is not None:
                    try:
                        embedding_list = ast.literal_eval(embedding_string)
                        if len(embedding_list) == embedding_dim:
                            # Directly write to the NumPy array instead of appending to a list
                            embeddings_for_training_np[current_idx] = embedding_list
                            current_idx += 1
                            batch_parsed += 1
                        else:
                            logger.warning(f"Skipping embedding with incorrect dimension: {len(embedding_list)}, expected {embedding_dim}")
                    except (ValueError, SyntaxError) as e:
                        logger.warning(f"Skipping embedding due to parsing error: {e}")
            
            # Log progress every 50 batches or if 10 seconds have passed since last log
            current_time = time.time()
            if current_batch % 50 == 0 or current_batch == batch_count or current_time - last_log_time > 10:
                # Calculate progress statistics
                progress_percent = (current_batch / batch_count) * 100
                elapsed_time = current_time - start_time
                estimated_total_time = elapsed_time / (current_batch / batch_count) if current_batch > 0 else 0
                estimated_remaining_time = estimated_total_time - elapsed_time
                
                # Get current memory usage
                current_memory_usage = process.memory_info().rss / (1024**3)
                
                # Calculate time for this batch
                batch_time = current_time - batch_start_time
                
                logger.info(f"Progress: {current_batch}/{batch_count} batches ({progress_percent:.1f}%) - "
                           f"Memory: {current_memory_usage:.2f} GB - "
                           f"Embeddings loaded: {current_idx} - "
                           f"Batch time: {batch_time:.2f}s - "
                           f"Est. remaining: {estimated_remaining_time/60:.1f} minutes")
                
                last_log_time = current_time
            
            # Periodically run garbage collection if memory usage is high
            if current_memory_usage > 30:  # If memory usage exceeds 30GB
                gc.collect()
                logger.info(f"Garbage collection run at {current_memory_usage:.2f} GB memory usage")
        
        # Trim the array if we didn't fill it completely (due to skipped embeddings)
        if current_idx < len(sampled_ids):
            logger.info(f"Only {current_idx} valid embeddings were loaded out of {len(sampled_ids)} sampled IDs. Trimming array.")
            embeddings_for_training_np = embeddings_for_training_np[:current_idx]
        
        num_embeddings_for_training = current_idx
        logger.info(f"Fetched total {num_embeddings_for_training} subsampled embeddings for training (ALL COLLECTIONS).")

        if num_embeddings_for_training == 0:
            logger.warning(f"No valid embeddings found after fetching and parsing from ALL COLLECTIONS. Training skipped.")
            return
            
        # Get current memory usage before training
        current_memory_usage = process.memory_info().rss / (1024**3)
        logger.info(f"Memory usage before PQ training: {current_memory_usage:.2f} GB")

        pq_num_subvectors = 8  # Tune these parameters as needed
        pq_bits_per_subvector = 8 # Tune these parameters as needed

        # Log training start
        logger.info(f"Starting PQ training for ALL COLLECTIONS. Using {num_embeddings_for_training} SUBSAMPLED embeddings.")
        
        # Call apply_product_quantization but modify it to work directly with the NumPy array
        # We can modify the function call without changing the function itself
        pq_quantizer, _ = apply_product_quantization_with_np(embeddings_for_training_np, 
                                                            pq_num_subvectors=pq_num_subvectors, 
                                                            pq_bits_per_subvector=pq_bits_per_subvector)

        # After training, remove the large array to free memory
        del embeddings_for_training_np
        gc.collect()
        
        # Get memory usage after training
        current_memory_usage = process.memory_info().rss / (1024**3)
        logger.info(f"Memory usage after PQ training: {current_memory_usage:.2f} GB")

        """ logger.info("--- Verification BEFORE Saving Faiss Index ---")
        logger.info(f"Quantizer Dimension (d): {pq_quantizer.d}")
        logger.info(f"Number of Subvectors (M): {pq_quantizer.M}")
        logger.info(f"Bits per Subvector (nbits): {pq_quantizer.nbits}") """

        #save_pq_quantizer_custom(pq_quantizer, pq_params_filepath)

        direct_pq_save_filepath = "pq_quantizer_direct.faiss" # Use a distinct extension/name

        logger.info("--- Verification BEFORE Saving using write_ProductQuantizer ---")
        logger.info(f"Quantizer Dimension (d): {pq_quantizer.d}")
        logger.info(f"Number of Subvectors (M): {pq_quantizer.M}")
        logger.info(f"Bits per Subvector (nbits): {pq_quantizer.nbits}")
        # Add a check for centroids existence/size before saving
        if hasattr(pq_quantizer, 'centroids') and pq_quantizer.centroids is not None:
             try:
                 logger.info(f"Centroids vector size before saving: {pq_quantizer.centroids.size()}")
             except AttributeError:
                 logger.warning("Could not get centroids vector size (might be older Faiss).")
        else:
             logger.warning("Quantizer seems to lack centroids before saving!")

        logger.info(f"Attempting to save quantizer directly using faiss.write_ProductQuantizer to: {direct_pq_save_filepath}")
        try:
            # *** Use the direct write function ***
            faiss.write_ProductQuantizer(pq_quantizer, direct_pq_save_filepath)
            logger.info(f"Trained Product Quantizer SAVED DIRECTLY to: {direct_pq_save_filepath}")

            # Optional: Verify immediately after saving
            logger.info("Verifying saved quantizer immediately...")
            loaded_verify_pq = faiss.read_ProductQuantizer(direct_pq_save_filepath)
            logger.info(f"  Verification successful: Loaded PQ Dim={loaded_verify_pq.d}, M={loaded_verify_pq.M}, nbits={loaded_verify_pq.nbits}")
            # Check codebooks norm again after loading
            try:
                 # --- Use faiss.vector_float_to_array for verification ---
                 if hasattr(loaded_verify_pq, 'centroids') and loaded_verify_pq.centroids is not None:
                     logger.debug("Attempting verification using faiss.vector_float_to_array")
                     verify_cb_np = faiss.vector_float_to_array(loaded_verify_pq.centroids)
                     verify_cb_norm = np.linalg.norm(verify_cb_np)
                     logger.info(f"  Verification codebook norm: {verify_cb_norm}")
                     if verify_cb_norm == 0:
                          logger.error("  Verification FAILED: Loaded codebooks have zero norm!")
                     # No need to delete verify_cb_np here, it's local scope
                 else:
                     logger.error("  Verification FAILED: Loaded quantizer has no centroids attribute!")
                 # --- End modification ---
            except AttributeError as e_vf2a:
                 logger.error(f"  Verification FAILED: faiss.vector_float_to_array not found? Error: {e_vf2a}")
            except Exception as e_verify_cb:
                 logger.error(f"  Verification codebook check error: {e_verify_cb}")

            del loaded_verify_pq # clean up loaded object for verification

        except Exception as e_save:
            logger.error(f"Failed to save ProductQuantizer directly: {e_save}", exc_info=True)
        logger.info(f"Trained Product Quantizer SAVED to: {pq_params_filepath} for ALL COLLECTIONS (using custom Faiss save)")
        logger.info("Faiss Product Quantizer training script completed successfully for ALL COLLECTIONS using BATCHED ID FETCHING and SUBSAMPLING (Solution 2).")

    except (Exception, psycopg2.Error) as error:
        logger.error("Error during Faiss Product Quantizer training process:", error, exc_info=True)
        if conn:
            conn.rollback()
    finally:
        if conn:
            cur.close()
            conn.close()
            logger.info("DB connection closed.")

# Add this new function to work directly with NumPy arrays
def apply_product_quantization_with_np(embeddings_np: np.ndarray, pq_num_subvectors: int = 8, pq_bits_per_subvector: int = 8):
    """
    Applies Product Quantization (PQ) to a numpy array of embeddings using Faiss.
    Takes a pre-allocated NumPy array to save memory.
    """
    if embeddings_np.size == 0 or embeddings_np.shape[0] == 0:
        return None, []

    # The input is already a NumPy array - no conversion needed
    dimension = embeddings_np.shape[1]
    n_vectors = embeddings_np.shape[0]

    # --- 1. Train the Product Quantizer ---
    pq_quantizer = faiss.ProductQuantizer(
        dimension,
        pq_num_subvectors,
        pq_bits_per_subvector
    )

    logger.info(f"Training Product Quantizer with {pq_num_subvectors} subvectors and {pq_bits_per_subvector} bits/subvector...")
    logger.info(f"Training on {n_vectors} embeddings, dimension: {dimension}")
    
    # Track memory before training
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024**3)
    logger.info(f"Memory before PQ training call: {mem_before:.2f} GB")
    
    # Time the training
    training_start = time.time()
    pq_quantizer.train(embeddings_np)
    training_end = time.time()
    
    # Track memory after training
    mem_after = process.memory_info().rss / (1024**3)
    logger.info(f"Memory after PQ training call: {mem_after:.2f} GB")
    logger.info(f"PQ training took {training_end - training_start:.2f} seconds")

    logger.info("--- Verification AFTER PQ Training ---")
    logger.info(f"Quantizer Dimension (d): {pq_quantizer.d}")
    logger.info(f"Number of Subvectors (M): {pq_quantizer.M}")
    logger.info(f"Bits per Subvector (nbits): {pq_quantizer.nbits}")

    # --- 2. Encode the Embeddings (not needed for training, but keeping for consistency) ---
    logger.info("Computing PQ codes (for verification only)...")
    pq_codes_np = pq_quantizer.compute_codes(embeddings_np)
    
    # --- 3. Convert to bytes (not needed for training) ---
    # Just create a small sample for verification
    sample_size = min(10, len(pq_codes_np))
    pq_encoded_embeddings_bytes = [pq_codes_np[i].tobytes() for i in range(sample_size)]

    logger.info(f"Product Quantization training completed. Trained on {n_vectors} embeddings, dimension: {dimension}.")

    return pq_quantizer, pq_encoded_embeddings_bytes


if __name__ == "__main__":
    subsample_size_arg = 10000000 # Default subsample size - you can change this default here - if as high as possible - take all rows in langchain_pg_embedding
    id_batch_size_arg = 100000 # Batch size for fetching IDs
    embedding_batch_size_arg = 1000 # Batch size for fetching embeddings
    train_faiss_quantizator(subsample_size=subsample_size_arg, id_batch_size=id_batch_size_arg, embedding_batch_size=embedding_batch_size_arg) # Run training with batched ID fetching and subsampling