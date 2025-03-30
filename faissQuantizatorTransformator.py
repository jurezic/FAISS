# faissQuantizatorApply.py - Apply trained quantizer (loaded via read_ProductQuantizer) to DB embeddings
import psycopg2
import psycopg2.extras # Needed for execute_batch
import yaml
import joblib # Keep for config loading if needed, but not for quantizer
import logging
import os
from typing import List, Optional

# --- PQ Libraries ---
import faiss
import numpy as np
import ast # Import ast for safe string evaluation

# Add these new imports for memory monitoring
import psutil
import gc
import time

# --- Logging Setup ---
# Set level to DEBUG to see more details during processing if needed
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Name for the new column to store quantized codes (BYTEA type)
PQ_CODES_COLUMN_NAME = "pq_codes"
# --- Path to the directly saved quantizer file ---
DIRECT_PQ_LOAD_FILEPATH = "pq_quantizer_direct.faiss"

# Load config function (reusing from previous scripts)
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# --- REMOVED load_pq_quantizer_custom function ---
# We no longer need the custom loading function

# --- Main Application Function ---
def apply_quantization_to_db(fetch_batch_size=1000, update_batch_size=1000):
    """
    Loads a trained Faiss Product Quantizer (using read_ProductQuantizer)
    and applies it to all embeddings in the langchain_pg_embedding table,
    storing the PQ codes in a new column.
    Uses server-side cursors and batch updates for optimization.
    """
    config = load_config()
    db_config = config['database']
    # quantizer_path_prefix is no longer needed

    DB_HOST = db_config['host']
    DB_NAME = db_config['name']
    DB_USER = db_config['user']
    DB_PASSWORD = db_config['password']
    DB_PORT = db_config['port']

    conn = None
    total_rows_processed = 0
    total_rows_updated = 0
    errors_parsing = 0
    errors_dimension = 0
    start_time = time.time()
    CURSOR_NAME = "embedding_fetch_cursor_hold" # Give it a unique name

    try:
        # --- 1. Load the Trained Quantizer using faiss.read_ProductQuantizer ---
        logger.info(f"Loading trained Faiss Product Quantizer via read_ProductQuantizer from: {DIRECT_PQ_LOAD_FILEPATH}")
        pq_quantizer = None # Initialize
        try:
            # *** Use the direct read function ***
            pq_quantizer = faiss.read_ProductQuantizer(DIRECT_PQ_LOAD_FILEPATH)
            logger.info("Quantizer loaded successfully using read_ProductQuantizer.")

            if pq_quantizer is None:
                raise ValueError("read_ProductQuantizer returned None!")

            quantizer_dim = pq_quantizer.d
            logger.info(f"Quantizer loaded successfully. Expected dimension: {quantizer_dim}")

            # --- Verification AFTER Loading (Direct) ---
            logger.info("--- Verification AFTER PQ Loading (Direct) ---")
            logger.info(f"Quantizer Dimension (d): {pq_quantizer.d}")
            logger.info(f"Number of Subvectors (M): {pq_quantizer.M}")
            logger.info(f"Bits per Subvector (nbits): {pq_quantizer.nbits}")
            # Check codebooks norm after loading using vector_float_to_array
            try:
                if hasattr(pq_quantizer, 'centroids') and pq_quantizer.centroids is not None:
                    logger.debug("Attempting verification using faiss.vector_float_to_array")
                    temp_codebooks_np = faiss.vector_float_to_array(pq_quantizer.centroids)
                    codebooks_norm = np.linalg.norm(temp_codebooks_np)
                    logger.info(f"Loaded codebooks norm: {codebooks_norm}")
                    if codebooks_norm == 0:
                        logger.error("CRITICAL: Loaded quantizer codebooks (Direct Load) have norm zero!")
                    else:
                        logger.debug(f"Loaded codebooks (Direct Load) seem non-zero (norm={codebooks_norm}).")
                    # No need to delete temp_codebooks_np here, local scope
                else:
                    logger.error("CRITICAL: Loaded quantizer has no centroids attribute!")
            except AttributeError as e_vf2a:
                logger.error(f"Loaded codebook verification FAILED: faiss.vector_float_to_array not found? Error: {e_vf2a}")
            except Exception as e_debug_cb:
                logger.error(f"Error during loaded codebook verification: {e_debug_cb}")
            # --- END Verification ---

        except FileNotFoundError:
             logger.error(f"Quantizer direct save file not found: {DIRECT_PQ_LOAD_FILEPATH}. Please run the training script first.")
             return
        except Exception as e_load:
            logger.error(f"Failed to load Product Quantizer using read_ProductQuantizer from {DIRECT_PQ_LOAD_FILEPATH}: {e_load}", exc_info=True)
            return
        # --- End Loading Section ---


        # --- 2. Connect to Database ---
        logger.info("Connecting to the database...")
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
        logger.info("Database connection successful.")


        # --- 3. Add PQ Codes Column (if it doesn't exist) ---
        # (Same logic as before)
        with conn.cursor() as cur_alter:
            try:
                logger.info(f"Checking/Adding column '{PQ_CODES_COLUMN_NAME}' (BYTEA) to langchain_pg_embedding table...")
                alter_sql = f"""
                ALTER TABLE langchain_pg_embedding
                ADD COLUMN IF NOT EXISTS {PQ_CODES_COLUMN_NAME} BYTEA;
                """
                cur_alter.execute(alter_sql)
                conn.commit() # Commit after altering table
                logger.info(f"Column '{PQ_CODES_COLUMN_NAME}' ensured.")
            except psycopg2.Error as e:
                logger.error(f"Error checking/adding column: {e}")
                conn.rollback() # Rollback if alter fails
                return # Exit if we can't add the column


        # --- 4. Process Embeddings using EXPLICITLY declared WITH HOLD Cursor ---
        # (The rest of this section remains the same as your corrected version
        #  using the explicitly declared WITH HOLD cursor, batch updates,
        #  progress logging, memory monitoring, etc.)

        logger.info("Starting embedding processing and quantization...")
        process = psutil.Process(os.getpid())

        # Use standard cursors now, we will manage the WITH HOLD cursor manually
        with conn.cursor() as cur_fetch, conn.cursor() as cur_update:

            # --- Explicitly Declare Cursor WITH HOLD ---
            logger.info(f"Declaring server-side cursor '{CURSOR_NAME}' WITH HOLD...")
            cur_fetch.execute(f"DECLARE {CURSOR_NAME} CURSOR WITH HOLD FOR SELECT id, embedding FROM langchain_pg_embedding")
            logger.info("Server-side cursor declared.")

            update_batch_data = []
            last_log_time = start_time
            # --- Add Flag for Debugging ---
            debug_first_few_rows = 5 # Number of rows to print detailed debug info for
            rows_debugged = 0 # Counter for debug logging

            while True:
                batch_start_time = time.time()
                logger.debug(f"Fetching {fetch_batch_size} rows from cursor {CURSOR_NAME}...")
                cur_fetch.execute(f"FETCH {fetch_batch_size} FROM {CURSOR_NAME}")
                rows_batch = cur_fetch.fetchall()
                logger.debug(f"Fetched {len(rows_batch)} rows.")

                if not rows_batch:
                    logger.info("No more rows to fetch from cursor.")
                    break # Exit the loop if no more rows

                total_rows_processed += len(rows_batch)

                # --- Process Batch ---
                for row_id, embedding_string in rows_batch:
                    # --- *** DEBUG: Detailed logging for first few rows *** ---
                    is_debug_row = rows_debugged < debug_first_few_rows
                    if is_debug_row:
                        logger.debug(f"--- Debugging Row ID: {row_id} ---")
                        logger.debug(f"Raw embedding string (start): {embedding_string[:100] if embedding_string else 'None'}...")

                    if embedding_string is None:
                        # logger.warning(f"Skipping row ID {row_id} due to NULL embedding.") # Reduced verbosity
                        continue

                    try:
                        embedding_list = ast.literal_eval(embedding_string)
                        if not isinstance(embedding_list, list):
                             raise ValueError("Parsed data is not a list")

                        if is_debug_row:
                             logger.debug(f"Parsed embedding list (first 10 elements): {embedding_list[:10]}")

                        embedding_dim_db = len(embedding_list)
                        if embedding_dim_db != quantizer_dim:
                            logger.warning(f"Skipping row ID {row_id} due to dimension mismatch. DB: {embedding_dim_db}, Quantizer: {quantizer_dim}")
                            errors_dimension += 1
                            continue

                        embedding_np = np.array([embedding_list], dtype=np.float32)
                        input_norm = np.linalg.norm(embedding_np) # Check norm before quantizing

                        if is_debug_row:
                            logger.debug(f"Input embedding_np shape: {embedding_np.shape}")
                            logger.debug(f"Input embedding_np norm: {input_norm}")
                        if input_norm == 0 and is_debug_row: # Only warn once per debug row if input is zero
                             logger.warning(f"Row ID {row_id}: Input embedding norm is zero before compute_codes!")

                        # Compute PQ codes using the LOADED quantizer
                        pq_codes_np = pq_quantizer.compute_codes(embedding_np)
                        pq_codes_bytes = pq_codes_np[0].tobytes()

                        # Debug output codes
                        if is_debug_row or np.all(pq_codes_np[0] == 0):
                            logger.debug(f"Row ID {row_id}: Computed pq_codes_np (raw uint8): {pq_codes_np[0]}")
                            logger.debug(f"Row ID {row_id}: Computed pq_codes_bytes (hex): {pq_codes_bytes.hex()}")
                            if np.all(pq_codes_np[0] == 0) and input_norm > 1e-6: # Warn only if input wasn't zero-ish
                                 logger.warning(f"Row ID {row_id}: Produced ALL ZERO PQ codes despite non-zero input norm ({input_norm:.4f})!")

                        # Add to batch update list
                        update_batch_data.append((pq_codes_bytes, row_id))

                        if is_debug_row:
                            rows_debugged += 1

                    except (ValueError, SyntaxError, TypeError) as e:
                        # logger.warning(f"Skipping row ID {row_id} due to parsing error: {e}") # Reduced verbosity
                        errors_parsing += 1
                        continue
                    except Exception as e_inner:
                        logger.error(f"Unexpected error processing row ID {row_id}: {e_inner}", exc_info=True)
                        continue

                # --- Execute Batch Update ---
                if update_batch_data:
                    update_sql = f"""
                    UPDATE langchain_pg_embedding
                    SET {PQ_CODES_COLUMN_NAME} = %s
                    WHERE id = %s
                    """
                    try:
                        logger.debug(f"Executing batch update for {len(update_batch_data)} rows...")
                        psycopg2.extras.execute_batch(cur_update, update_sql, update_batch_data, page_size=update_batch_size)
                        conn.commit() # Commit the updates - the WITH HOLD cursor will survive
                        total_rows_updated += len(update_batch_data)
                        logger.debug(f"Batch update committed. Total updated so far: {total_rows_updated}")
                        update_batch_data = []
                    except psycopg2.Error as e_update:
                        logger.error(f"Error during batch update: {e_update}")
                        conn.rollback()
                        logger.warning("Rolling back failed update batch...")
                        update_batch_data = []
                    except Exception as e_update_unexpected:
                        logger.error(f"Unexpected error during batch update: {e_update_unexpected}", exc_info=True)
                        conn.rollback()
                        update_batch_data = []


                # --- Progress Logging ---
                current_time = time.time()
                if current_time - last_log_time > 10 or not rows_batch: # Log every 10s or at the end
                    elapsed_time = current_time - start_time
                    rows_per_sec = total_rows_processed / elapsed_time if elapsed_time > 0 else 0
                    current_memory_usage_gb = process.memory_info().rss / (1024**3)

                    logger.info(f"Progress: Processed ~{total_rows_processed} rows - Updated: {total_rows_updated} - "
                               f"Rate: {rows_per_sec:.1f} rows/sec - Memory: {current_memory_usage_gb:.2f} GB - "
                               f"Errors (Parse/Dim): {errors_parsing}/{errors_dimension} - "
                               f"Elapsed: {elapsed_time/60:.1f} min")
                    last_log_time = current_time

                # Optional: Explicit garbage collection
                gc.collect()


            # --- Close the Explicit Cursor ---
            logger.info(f"Closing server-side cursor '{CURSOR_NAME}'...")
            cur_fetch.execute(f"CLOSE {CURSOR_NAME}")
            conn.commit() # Commit the close operation
            logger.info("Cursor closed.")


            # --- Final Log Summary ---
            end_time = time.time()
            total_time = end_time - start_time
            avg_rate = total_rows_processed / total_time if total_time > 0 else 0
            logger.info("--- Processing Complete ---")
            logger.info(f"Total rows processed: {total_rows_processed}")
            logger.info(f"Total rows successfully updated with PQ codes: {total_rows_updated}")
            logger.info(f"Total parsing errors: {errors_parsing}")
            logger.info(f"Total dimension mismatch errors: {errors_dimension}")
            logger.info(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
            logger.info(f"Average processing rate: {avg_rate:.1f} rows/sec")


    except (Exception, psycopg2.Error) as error:
        logger.error("Error during Faiss Quantizer application process:", exc_info=True) # Log with traceback
        if conn:
            conn.rollback() # Rollback any pending transaction on error
    finally:
        if conn:
            # --- Ensure explicit cursor is closed on any exit ---
            try:
                with conn.cursor() as cur_cleanup:
                    logger.info(f"Ensuring cursor {CURSOR_NAME} is closed in finally block...")
                    cur_cleanup.execute(f"CLOSE {CURSOR_NAME}")
                    conn.commit()
                    logger.info(f"Cursor {CURSOR_NAME} closed via finally block.")
            except psycopg2.Error as e_close:
                logger.warning(f"Could not close cursor {CURSOR_NAME} in finally (may already be closed): {e_close}")
                conn.rollback()

            conn.close()
            logger.info("DB connection closed.")

# --- __main__ block ---
if __name__ == "__main__":
    # Set Log Level to DEBUG for detailed testing initially
    logging.getLogger().setLevel(logging.DEBUG)

    fetch_batch_size_arg = 1000  # How many rows to fetch from DB at a time
    update_batch_size_arg = 500 # How many rows to update in a single batch query
    apply_quantization_to_db(fetch_batch_size=fetch_batch_size_arg, update_batch_size=update_batch_size_arg)