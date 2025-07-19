# /// script
# requires-python = ">=3.13"
# dependencies = [
#    "kagglehub",
#    "pandas",
#    "pyarrow",
#    "rich",
# ]
# ///

import kagglehub
import sys
from pathlib import Path
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
import pyarrow.parquet as pq
import os
import shutil
import cProfile
import pstats


def main() -> None:
    CHUNK_SIZE = 1_000_000
    parquet_file = "all_stock_data.parquet"
    base_path = Path.cwd() / "data"
    dataset_dir = base_path / "datasets"
    output_dir = base_path / "ticker-data"
    base_path.mkdir(parents=True, exist_ok=True)
    output_dir.unlink(missing_ok=True)  # remove previously made data
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["KAGGLEHUB_CACHE"] = str(base_path)

    dataset_path = (
        dataset_dir
        / "jakewright"
        / "9000-tickers-of-stock-market-data-full-history"
        / "versions"
        / "2"
        / parquet_file
    )

    if not dataset_path.exists():
        dataset_location = kagglehub.dataset_download(
            handle="jakewright/9000-tickers-of-stock-market-data-full-history",
            path=parquet_file,
        )
        dataset_path = Path(dataset_location)
        print(f"saved dataset at: {dataset_path}")

    writers: dict[str, pq.ParquetWriter] = {}

    try:
        parquet_file_reader = pq.ParquetFile(dataset_path)
        total_rows = parquet_file_reader.metadata.num_rows

        with tqdm(total=total_rows, desc="Processing rows") as pbar:
            for batch in parquet_file_reader.iter_batches(
                batch_size=CHUNK_SIZE
            ):
                chunk = batch.to_pandas()

                for ticker, group in chunk.groupby("Ticker"):
                    out_file = output_dir / f"{ticker}.parquet"
                    table = pa.Table.from_pandas(group, preserve_index=False)

                    if ticker not in writers:
                        schema = table.schema
                        writers[ticker] = pq.ParquetWriter(out_file, schema)

                    writers[ticker].write_table(table)

                # Update progress bar
                pbar.update(len(chunk))

    finally:
        for writer in writers.values():
            writer.close()

    if len(sys.argv) > 1 and sys.argv[1] == "+cleanup":
        dataset_path.unlink()


if __name__ == "__main__":
    main()
