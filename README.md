# FAISS
FAISS Quantizer code snippets for optimised training and transforming embeddings

# Faiss Product Quantizer Training and Application Scripts

## Overview

This is a showcase of an optimised quantization process used for memory reduction and potential performance gains in building RAG applications on top of large datasets where instead of vector similarity, kNN search is used.

These Python scripts provide a workflow for leveraging Faiss's Product Quantization (PQ) capabilities on large datasets of high-dimensional vector embeddings. The primary goals are:

1.  **Memory Reduction:** Significantly reduce the storage footprint required for vector embeddings through lossy compression.
2.  **Potential Performance Gains:** Enable faster Approximate Nearest Neighbor (ANN) search compared to searching on raw, high-dimensional vectors (requires appropriate indexing and search implementation later).

The workflow consists of two main scripts:

1.  `faissQuantizatorTraining.py`: Trains a Faiss Product Quantizer on your existing embedding data.
2.  `faissQuantizatorApply.py` (or `faissQuantizatorTransformator.py`): Applies the trained quantizer to generate compressed PQ codes for your embeddings.

## Core Concept: Product Quantization (PQ)

Product Quantization is a vector quantization technique. It works by:

1.  Splitting high-dimensional vectors into several lower-dimensional sub-vectors.
2.  Running a clustering algorithm (like k-means) independently on the training data for each sub-vector space to create separate "codebooks".
3.  Representing each original vector by finding the nearest codebook entry (centroid) for each of its sub-vectors and storing the index of that centroid.

The result is a compressed representation (the PQ codes, typically stored as bytes) that requires significantly less storage than the original float vectors. This comes at the cost of some information loss (lossy compression), creating a trade-off between compression ratio, search speed, and search accuracy.

## Scripts

### 1. `faissQuantizatorTraining.py`

*   **Purpose:** To train a `faiss.ProductQuantizer` model based on the characteristics of your embedding data.
*   **Input:**
    *   A source of high-dimensional float vector embeddings (e.g., from a database, files).
    *   Configuration file (`config.yaml`) specifying database connections (if applicable), PQ parameters (`M` - number of subvectors, `nbits` - bits per subvector index), file paths, and batch sizes.
*   **Process:**
    *   Connects to the data source.
    *   Fetches embedding vectors, potentially using batching and subsampling for very large datasets to manage memory.
    *   Initializes and trains a `faiss.ProductQuantizer` instance using the fetched embeddings.
    *   Saves the trained quantizer object to a file using `faiss.write_ProductQuantizer` for later use.
    *   Includes logging and basic memory monitoring for robustness.
*   **Output:**
    *   A saved quantizer file (e.g., `pq_quantizer_direct.faiss`).
    *   Log output detailing the process.

### 2. `faissQuantizatorApply.py` / `faissQuantizatorTransformator.py`

*   **Purpose:** To load a previously trained quantizer and use it to generate the compressed PQ codes for all embeddings in your dataset.
*   **Input:**
    *   The saved quantizer file created by the training script.
    *   A source of high-dimensional float vector embeddings (typically the same source used for training).
    *   Configuration file (`config.yaml`).
*   **Process:**
    *   Loads the trained quantizer using `faiss.read_ProductQuantizer`.
    *   Connects to the data source.
    *   Fetches embeddings in batches (using techniques like server-side cursors if reading from a database).
    *   For each embedding, uses the loaded quantizer's `compute_codes` method to generate the `uint8` PQ code array.
    *   Converts the codes to bytes (`.tobytes()`).
    *   Stores the resulting PQ code bytes in a designated location (e.g., updating a database column, writing to files). Uses batching for efficient storage updates.
    *   Includes logging and basic memory monitoring.
*   **Output:**
    *   Stored PQ codes for the embeddings.
    *   Log output detailing the process.

## Prerequisites

*   Python 3.x
*   `faiss-cpu` or `faiss-gpu`
*   `numpy`
*   `PyYAML` (for configuration)
*   `psutil` (for memory monitoring)
*   Database connector (e.g., `psycopg2-binary`) if reading/writing from/to a database.

## Configuration

The scripts typically rely on a `config.yaml` file for settings:

*   **Database Connection Details:** Host, port, name, user, password (if applicable).
*   **File Paths:** Path to save/load the trained quantizer file.
*   **PQ Parameters:** `pq_num_subvectors` (M), `pq_bits_per_subvector` (nbits) used during training.
*   **Batch Sizes:** `id_batch_size`, `embedding_batch_size` (for training), `fetch_batch_size`, `update_batch_size` (for applying) to control memory usage and performance.
*   **Collection/Table Names:** Identifiers for data sources/destinations.

Ensure the `config.yaml` file is correctly configured before running the scripts.

## Usage

1.  **Configure:** Edit `config.yaml` with your specific data source details, desired PQ parameters, file paths, and batch sizes.
2.  **Train:** Run the training script:
    ```bash
    python faissQuantizatorTraining.py [--debug]
    ```
    This will generate the quantizer file (e.g., `pq_quantizer_direct.faiss`).
3.  **Apply:** Run the applying/transforming script:
    ```bash
    python faissQuantizatorApply.py [--debug]
    ```
    (Use the correct filename for your applying script). This will load the quantizer and generate/store the PQ codes.

## Important Considerations

*   **Accuracy Trade-off:** Product Quantization is **lossy**. Using PQ codes for Approximate Nearest Neighbor (ANN) search will generally be less accurate than searching on the original float vectors. The choice of `M` and `nbits` directly impacts this trade-off. Higher `nbits` usually leads to better accuracy but less compression.
*   **Using PQ Codes for Search:** These scripts only *generate* the PQ codes. To perform similarity search using these codes, you typically need to:
    *   Build a dedicated Faiss index (like `IndexIVFPQ` or `IndexPQ`) that uses the trained quantizer.
    *   Implement custom search logic (e.g., using Faiss search methods or database User-Defined Functions) that performs Asymmetric Distance Computation (ADC) or Symmetric Distance Computation (SDC). Standard vector database operators often cannot directly use raw PQ codes.
*   **Parameter Tuning:** The optimal values for `M` and `nbits` depend heavily on the dataset characteristics, desired compression level, target accuracy, and performance requirements. Experimentation is often needed.
*   **Memory:** While PQ codes save storage, training the quantizer and building/loading Faiss indexes (if used for search) requires sufficient RAM. The batching implemented helps manage memory during processing.
