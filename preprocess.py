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


def setup_dataset(parquet_file: str, base_path: Path) -> Path:
    dataset_dir = base_path / "datasets"

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

    return dataset_path


def setup_output_directory(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def process_stock_data2(dataset_path: Path, output_dir: Path) -> int:
    print("importing dataset to pandas .", end="")
    df = pq.read_table(dataset_path).to_pandas()
    print("..", end=" ")
    tickers = df["Ticker"].unique()
    print("done")

    with Progress(
        TextColumn("[progress.description]{task.description} "),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("tickers"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("• {task.fields[rate]:.1f} tickers/sec"),
        refresh_per_second=30, # eye candy
    ) as progress:
        task = progress.add_task(
            "writing ticker files", total=len(tickers), rate=0.0
        )

        processed_count = 0

        for ticker, group in df.groupby("Ticker", sort=False):
            out_file = output_dir / f"{ticker}.parquet"

            group.to_parquet(
                out_file,
                index=False,
                engine="pyarrow",
                compression="snappy",
            )
            processed_count += 1

            elapsed = progress.tasks[task].elapsed or 1
            rate = processed_count / elapsed
            progress.update(task, advance=1, rate=rate)

    return len(list(output_dir.glob("*.parquet")))


def cleanup_dataset(dataset_path: Path) -> None:
    if dataset_path.exists():
        dataset_path.unlink()


def main() -> None:
    parquet_file = "all_stock_data.parquet"
    base_path = Path.cwd() / "data"
    output_dir = base_path / "ticker-data"

    base_path.mkdir(parents=True, exist_ok=True)

    dataset_path = setup_dataset(parquet_file, base_path)
    setup_output_directory(output_dir)

    output_files_count = process_stock_data2(dataset_path, output_dir)

    print(f"Created {output_files_count} ticker files")

    if len(sys.argv) > 1 and sys.argv[1] == "+cleanup":
        cleanup_dataset(dataset_path)


def profile(fn):
    profiler = cProfile.Profile()
    profiler.enable()

    fn()

    profiler.disable()

    stats = pstats.Stats(profiler)

    stats.sort_stats("cumulative")
    stats.dump_stats("stats.prof")

    print("\nprofile saved to 'stats.prof'")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "+profile" in args:
        profile(main)
    else:
        main()
