"""Shared I/O utilities for the Clin-JEPA data pipeline.

Handles table loading (concepts CSV and raw .csv.gz), path resolution
from config dot-notation, and cohort chunking for SLURM array jobs.
"""

import gc
import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from clin_jepa.utils import expand_path as resolve_path

logger = logging.getLogger(__name__)


def resolve_table_path(paths_config: dict, dotpath: str) -> Path:
    """Resolve a dot-notation path to an absolute file path.

    Walks the paths_config dict using dot-separated keys. The final value
    should be a relative path string, which is resolved from the project root.

    Args:
        paths_config: The loaded mimic_paths.yaml config dict.
        dotpath: Dot-separated key path, e.g. "concepts.measurement.vitalsign".

    Returns:
        Absolute Path to the file.

    Raises:
        KeyError: If any segment of the dotpath is not found in the config.
    """
    node = paths_config
    for key in dotpath.split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(
                f"Key '{key}' not found in config at path '{dotpath}'. "
                f"Available keys: {list(node.keys()) if isinstance(node, dict) else 'N/A'}"
            )
        node = node[key]
    if not isinstance(node, str):
        raise ValueError(
            f"Dotpath '{dotpath}' did not resolve to a file path string. "
            f"Got {type(node).__name__}: {node}"
        )
    return resolve_path(node)


def load_concepts_table(
    paths_config: dict,
    dotpath: str,
    usecols: list[str] | None = None,
    dtype: dict | None = None,
) -> pd.DataFrame:
    """Load a concepts table (plain CSV) from MIMIC-IV.

    Args:
        paths_config: The loaded mimic_paths.yaml config dict.
        dotpath: Dot-separated config path, e.g. "concepts.measurement.vitalsign".
        usecols: Optional list of columns to read.
        dtype: Optional dtype dict for columns.

    Returns:
        DataFrame with the loaded data.

    Raises:
        FileNotFoundError: If the resolved file does not exist.
    """
    filepath = resolve_table_path(paths_config, dotpath)
    if not filepath.exists():
        raise FileNotFoundError(f"Concepts table not found: {filepath}")

    logger.info("Loading concepts table: %s", filepath.name)
    df = pd.read_csv(filepath, usecols=usecols, dtype=dtype)
    logger.info("  Loaded %d rows x %d cols from %s", len(df), len(df.columns), filepath.name)
    return df


def load_raw_table(
    paths_config: dict,
    section: str,
    table_name: str,
    usecols: list[str] | None = None,
    dtype: dict | None = None,
    chunksize: int | None = None,
    parse_dates: list[str] | None = None,
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    """Load a raw MIMIC-IV table (.csv.gz compressed).

    Args:
        paths_config: The loaded mimic_paths.yaml config dict.
        section: Config section ("icu" or "hosp").
        table_name: Table name within that section (e.g. "inputevents").
        usecols: Optional list of columns to read.
        dtype: Optional dtype dict.
        chunksize: If set, returns an iterator of DataFrames.
        parse_dates: Columns to parse as datetime.

    Returns:
        DataFrame or Iterator[DataFrame] if chunksize is set.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    dotpath = f"{section}.{table_name}"
    filepath = resolve_table_path(paths_config, dotpath)

    if not filepath.exists():
        raise FileNotFoundError(f"Raw table not found: {filepath}")

    compression = "gzip" if filepath.suffix == ".gz" else None
    logger.info(
        "Loading raw table: %s (compression=%s, chunksize=%s)",
        filepath.name, compression, chunksize,
    )

    result = pd.read_csv(
        filepath,
        compression=compression,
        usecols=usecols,
        dtype=dtype,
        chunksize=chunksize,
        parse_dates=parse_dates,
    )

    if chunksize is None:
        logger.info("  Loaded %d rows x %d cols from %s", len(result), len(result.columns), filepath.name)

    return result


LARGE_TABLE_THRESHOLD = 300_000_000  # 300 MB — use chunked reading above this
DEFAULT_CHUNKSIZE = 1_000_000


def load_and_filter_concepts_table(
    paths_config: dict,
    dotpath: str,
    join_col: str,
    cohort_ids: set,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    """Load a concepts table, filtering to cohort IDs during read.

    Uses chunked reading for large files (>300 MB) to limit memory.
    Each chunk is filtered to cohort_ids before accumulating.

    Args:
        paths_config: The loaded mimic_paths.yaml config dict.
        dotpath: Dot-separated config path.
        join_col: Column to filter on (e.g. "stay_id" or "subject_id").
        cohort_ids: Set of IDs to keep.
        usecols: Optional list of columns to read. If provided and join_col
                 is not included, it will be added automatically.

    Returns:
        Filtered DataFrame.
    """
    filepath = resolve_table_path(paths_config, dotpath)
    if not filepath.exists():
        raise FileNotFoundError(f"Concepts table not found: {filepath}")

    # Ensure join_col is in usecols
    if usecols is not None and join_col not in usecols:
        usecols = [join_col] + list(usecols)

    file_size = filepath.stat().st_size

    if file_size > LARGE_TABLE_THRESHOLD:
        logger.info(
            "Loading %s in chunks (%.0f MB, filtering on %s)",
            filepath.name, file_size / 1e6, join_col,
        )
        frames = []
        n_total = 0
        for chunk in pd.read_csv(
            filepath, usecols=usecols, chunksize=DEFAULT_CHUNKSIZE,
            encoding_errors="replace",
        ):
            n_total += len(chunk)
            filtered = chunk[chunk[join_col].isin(cohort_ids)]
            if len(filtered) > 0:
                frames.append(filtered)
            del chunk
        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        logger.info(
            "  Loaded %d / %d rows from %s",
            len(result), n_total, filepath.name,
        )
        return result
    else:
        logger.info("Loading %s (%.0f MB)", filepath.name, file_size / 1e6)
        df = pd.read_csv(filepath, usecols=usecols, low_memory=False, encoding_errors="replace")
        result = df[df[join_col].isin(cohort_ids)]
        logger.info(
            "  Loaded %d / %d rows from %s",
            len(result), len(df), filepath.name,
        )
        del df
        gc.collect()
        return result.reset_index(drop=True)


RAW_TABLE_CHUNKSIZE = 500_000


def load_and_filter_raw_table(
    paths_config: dict,
    source_dotpath: str,
    join_col: str,
    cohort_ids: set,
    usecols: list[str] | None = None,
    chunksize: int = RAW_TABLE_CHUNKSIZE,
) -> pd.DataFrame:
    """Load a raw MIMIC-IV table (.csv.gz), filtering to cohort during read.

    Reads in chunks to manage memory on large compressed files.
    Each chunk is filtered to cohort_ids before accumulating.

    Args:
        paths_config: The loaded mimic_paths.yaml config dict.
        source_dotpath: Dot-separated config path (e.g. "icu.inputevents").
        join_col: Column to filter on (e.g. "stay_id" or "hadm_id").
        cohort_ids: Set of IDs to keep.
        usecols: Optional list of columns to read. If provided and join_col
                 is not included, it will be added automatically.
        chunksize: Number of rows per chunk.

    Returns:
        Filtered DataFrame with all matching rows.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    filepath = resolve_table_path(paths_config, source_dotpath)
    if not filepath.exists():
        raise FileNotFoundError(f"Raw table not found: {filepath}")

    # Ensure join_col is in usecols
    if usecols is not None and join_col not in usecols:
        usecols = [join_col] + list(usecols)

    compression = "gzip" if filepath.suffix == ".gz" else None
    logger.info(
        "Loading raw table %s in chunks (compression=%s, join=%s)",
        filepath.name, compression, join_col,
    )

    frames = []
    n_total = 0
    for chunk in pd.read_csv(
        filepath, compression=compression, usecols=usecols,
        chunksize=chunksize, low_memory=False,
    ):
        n_total += len(chunk)
        filtered = chunk[chunk[join_col].isin(cohort_ids)]
        if len(filtered) > 0:
            frames.append(filtered)
        del chunk

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    logger.info(
        "  Loaded %d / %d rows from %s",
        len(result), n_total, filepath.name,
    )
    del frames
    gc.collect()
    return result


def chunk_stay_ids(
    stay_ids: np.ndarray,
    chunk_idx: int,
    total_chunks: int,
) -> np.ndarray:
    """Split stay_ids into chunks for SLURM array parallelization.

    Args:
        stay_ids: Array of all stay_ids (should be sorted).
        chunk_idx: This worker's chunk index (0-based).
        total_chunks: Total number of chunks/workers.

    Returns:
        Array of stay_ids for this chunk.

    Raises:
        ValueError: If chunk_idx >= total_chunks or total_chunks < 1.
    """
    if total_chunks < 1:
        raise ValueError(f"total_chunks must be >= 1, got {total_chunks}")
    if chunk_idx >= total_chunks:
        raise ValueError(f"chunk_idx ({chunk_idx}) must be < total_chunks ({total_chunks})")

    chunks = np.array_split(stay_ids, total_chunks)
    selected = chunks[chunk_idx]
    logger.info(
        "Chunk %d/%d: %d stay_ids (total: %d)",
        chunk_idx, total_chunks, len(selected), len(stay_ids),
    )
    return selected
