"""Data Loader — Standardized file import utility for CSV, Excel, and TXT formats.

This module provides a unified interface for loading data from various file formats
with flexible parameter customization. It handles CSV, Excel, and text files with
support for common preprocessing options like skipping rows, specifying delimiters,
and all pandas read_* function parameters.

Usage example
-------------
>>> from data_loader import load_data
>>> df = load_data("data.csv", skiprows=2)
>>> df_excel = load_data("data.xlsx", sheet_name="Sheet1")
>>> df_txt = load_data("data.txt", sep="|", encoding="utf-8")

Key features:
- Auto-detects file format from extension (.csv, .xlsx, .xls, .txt)
- Supports all standard pandas read_csv() and read_excel() parameters
- Allows custom delimiters and encodings
- Flexible row skipping and header specification
- Verbose output option for debugging
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_data(
    filepath: str | Path,
    file_format: str | None = None,
    skiprows: int | list[int] | None = None,
    sep: str | None = None,
    encoding: str | None = "utf-8",
    sheet_name: str | int = 0,
    verbose: bool = False,
    **kwargs: Any
) -> pd.DataFrame:
    """Load data from CSV, Excel, or TXT files with customizable parameters.

    This function provides a unified interface for loading tabular data from
    various file formats. It automatically detects the file format from the
    extension but allows explicit specification if needed.

    Parameters
    ----------
    filepath : str or Path
        Path to the file to load.

    file_format : str, optional
        File format: 'csv', 'excel', or 'txt'. If None, auto-detected from
        the file extension. Default is None.

    skiprows : int, list of int, or None, default=None
        Row number(s) to skip. Can be:
        - An integer: skip first N rows
        - A list of integers: skip specific row numbers
        - None: skip no rows (except potentially header)
        Example: skiprows=2 skips first 2 rows; skiprows=[0, 2] skips rows 0 and 2

    sep : str, optional
        Delimiter/separator for CSV and TXT files. Default is None (pandas auto-detects).
        Common options: ',' (CSV), '\t' (tab), '|' (pipe), ';' (semicolon)

    encoding : str, default='utf-8'
        Character encoding for text files. Common options: 'utf-8', 'latin-1', 'iso-8859-1'

    sheet_name : str or int, default=0
        For Excel files, sheet to read from. Can be:
        - 0 (first sheet by default)
        - Sheet name as string (e.g., "Results")
        - Integer index (e.g., 1 for second sheet)

    verbose : bool, default=False
        If True, print loading details and basic DataFrame info.

    **kwargs : dict
        Additional keyword arguments passed to pandas read functions:
        - For CSV: pd.read_csv() parameters (header, dtype, nrows, etc.)
        - For Excel: pd.read_excel() parameters (header, dtype, nrows, etc.)
        - For TXT: pd.read_csv() parameters (treated as text with custom separator)

        Common useful parameters:
        - dtype: Column data types (dict or callable)
        - nrows: Number of rows to read
        - header: Row number(s) to use as column names (default 0)
        - names: List of column names
        - usecols: Columns to read (list or callable)
        - na_values: Additional strings to recognize as NA
        - keep_default_na: Keep default NA values (default True)
        - thousands: Thousands separator (e.g., ',' for European format)
        - decimal: Decimal separator (e.g., ',' for European format)
        - engine: Engine to use ('python', 'c', 'pyarrow' for read_csv)
        - index_col: Column to use as index

    Returns
    -------
    pd.DataFrame
        Loaded data as a pandas DataFrame.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If file format is not recognized.
    pd.errors.ParserError
        If there's an issue parsing the file.

    Examples
    --------
    Load CSV with default parameters:
    >>> df = load_data("data.csv")

    Load CSV skipping first 2 rows:
    >>> df = load_data("data.csv", skiprows=2)

    Load specific Excel sheet:
    >>> df = load_data("data.xlsx", sheet_name="Sheet1")

    Load tab-separated file with specific encoding:
    >>> df = load_data("data.txt", sep="\\t", encoding="latin-1")

    Load CSV with custom data types and number of rows:
    >>> df = load_data(
    ...     "data.csv",
    ...     dtype={"col1": int, "col2": float},
    ...     nrows=1000
    ... )

    Load CSV with European decimal format:
    >>> df = load_data(
    ...     "data.csv",
    ...     decimal=",",
    ...     thousands=".",
    ...     sep=";"
    ... )

    Skip specific non-contiguous rows:
    >>> df = load_data("data.csv", skiprows=[0, 2, 5])
    """
    # Convert to Path object for easier manipulation
    filepath = Path(filepath)

    # Check if file exists
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Auto-detect format from extension if not specified
    if file_format is None:
        extension = filepath.suffix.lower()
        if extension == ".csv":
            file_format = "csv"
        elif extension in (".xlsx", ".xls"):
            file_format = "excel"
        elif extension == ".txt":
            file_format = "txt"
        else:
            raise ValueError(
                f"Unknown file format: {extension}. "
                f"Supported formats: .csv, .xlsx, .xls, .txt"
            )

    if verbose:
        print(f"Loading file: {filepath}")
        print(f"Format: {file_format}")
        if skiprows:
            print(f"Skipping rows: {skiprows}")

    # Load based on format
    try:
        if file_format == "csv":
            csv_kwargs = {
                "skiprows": skiprows,
                "encoding": encoding,
                **kwargs
            }
            if sep is not None:
                csv_kwargs["sep"] = sep
            df = pd.read_csv(filepath, **csv_kwargs)
        elif file_format == "excel":
            df = pd.read_excel(
                filepath,
                sheet_name=sheet_name,
                skiprows=skiprows,
                **kwargs
            )
        elif file_format == "txt":
            # Treat TXT as CSV with customizable separator (default whitespace)
            if sep is None:
                sep = "\s+"  # Default: whitespace
            df = pd.read_csv(
                filepath,
                skiprows=skiprows,
                sep=sep,
                encoding=encoding,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        if verbose:
            print(f"✓ Successfully loaded {len(df)} rows × {len(df.columns)} columns")
            print(f"Columns: {list(df.columns)}")

        return df

    except pd.errors.ParserError as e:
        print(f"Error parsing file {filepath}: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error loading {filepath}: {e}")
        raise


def load_multiple_files(
    filepaths: list[str | Path],
    axis: int = 0,
    ignore_index: bool = False,
    keys: list[str] | None = None,
    verbose: bool = False,
    **kwargs: Any
) -> pd.DataFrame:
    """Load and concatenate multiple data files.

    Parameters
    ----------
    filepaths : list of str or Path
        List of file paths to load.

    axis : {0, 1}, default=0
        Concatenation axis:
        - 0: Stack rows (default, same columns)
        - 1: Stack columns (same rows)

    ignore_index : bool, default=False
        If True, ignore file path indices when concatenating rows.

    keys : list of str, optional
        Hierarchical keys to create MultiIndex. If None, file paths are used.

    verbose : bool, default=False
        Print loading progress for each file.

    **kwargs : dict
        Additional arguments passed to load_data().

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame from all files.

    Examples
    --------
    Load and stack multiple CSV files:
    >>> files = ["data1.csv", "data2.csv", "data3.csv"]
    >>> df_combined = load_multiple_files(files, axis=0)

    Load with hierarchical index:
    >>> df_combined = load_multiple_files(
    ...     files,
    ...     keys=["Dataset1", "Dataset2", "Dataset3"]
    ... )
    """
    dataframes = []

    for i, filepath in enumerate(filepaths):
        if verbose:
            print(f"Loading file {i + 1}/{len(filepaths)}: {filepath}")

        df = load_data(filepath, verbose=False, **kwargs)
        dataframes.append(df)

    if verbose:
        print(f"Concatenating {len(dataframes)} DataFrames...")

    combined_df = pd.concat(
        dataframes,
        axis=axis,
        ignore_index=ignore_index,
        keys=keys
    )

    if verbose:
        print(f"✓ Combined shape: {combined_df.shape}")

    return combined_df
