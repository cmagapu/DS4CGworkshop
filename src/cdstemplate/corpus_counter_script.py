"""An example of a script you can run. It tokenizes an folder of input documents and
writes the corpus counts to a user-specified CSV file
"""

# Import modules, functions and classes from external libraries
import argparse
import logging
from pathlib import Path

# Import the code from this project needed for this script
from cdstemplate import word_count, utils

from tqdm.contrib.concurrent import process_map
from functools import partial
from collections import Counter

logger = logging.getLogger(__name__)


def main_cli():
    """A wrapper function that defines command line arguments and help messages for
    when the user wants run this module's code as a script.
    """
    # The argument parser gives nice ways to include help message and specify which arguments
    # are required or optional, see https://docs.python.org/3/library/argparse.html#prog for usage instructions
    parser = argparse.ArgumentParser(description="A script to generate counts of tokens in a corpus")

    parser.add_argument("csv", help="Path to the output CSV storing token counts. Required.")

    parser.add_argument(
        "document_dir",
        type=Path,
        help="Path to folder containing raw .txt documents that make up the corpus. Required.",
    )
    parser.add_argument(
        "--case-insensitive",
        "-c",
        action="store_true",
        help="Default is to have case sensitive tokenization. Use this flag to make the token counting case insensitive. Optional.",
    )
    parser.add_argument(
        "--parallel",
        "-p",
        action="store_true",
        help="Default is to have non-parallel tokenization. Use this flag to make the token counting run in parallel (multiple processes to handle multiple documents). Optional.",
    )

    args = parser.parse_args()
    utils.configure_logging()
    logger.info("Command line arguments: %s", args)
    main(args.csv, args.document_dir, args.case_insensitive, args.parallel)


def _count_tokens_in_file(filepath: Path, case_insensitive: bool) -> dict:
    """
    Helper used in parallel mode. Each worker process runs this on one file,
    builds a tiny CorpusCounter, then returns a plain dict of {token: count}.
    """
    text = filepath.read_text()
    local_cc = word_count.CorpusCounter(case_insensitive=case_insensitive)
    local_cc.add_doc(text)
    # Return a simple dict so the parent can merge them without any pickling issues.
    df = local_cc.get_token_counts_as_dataframe()
    return {row["token"]: int(row["count"]) for _, row in df.iterrows()}  


def main(csv_out, document_dir, case_insensitive=False,use_parallel=False):
    """Determine cumulative word counts for a list of documents and write the results to a CSV file

    :param csv_out: output CSV file path
    :type csv_out: str or Path
    :param document_dir: Path to folder containing .txt files
    :type document_dir: Path
    :param case_insensitive: Set to True to lowercase all words in cumulative counts, defaults to False
    :type case_insensitive: bool, optional
    """
    document_dir = Path(document_dir)
    documents = sorted(document_dir.glob("*.txt"))
    cc = word_count.CorpusCounter(case_insensitive=case_insensitive)

    if not use_parallel:
        for i, doc in enumerate(documents):
            if i % 2 == 0:
                logger.info("Tokenizing document number %s: %s", i, doc)
                cc.add_doc(Path(doc).read_text())

        cc.save_token_counts(csv_out)
    
    else:
        logger.info(
            "Parallel mode: spawning workers for %d documents …", len(documents)
        )
        # Freeze the case_insensitive argument into the worker function:
        worker = partial(_count_tokens_in_file, case_insensitive=case_insensitive)

        # process_map will show a tqdm bar and return a list of dicts
        per_file_counts = process_map(
            worker,
            documents,
            desc="Tokenizing (parallel)",
            leave=True,
            # max_workers=4,      # optionally cap to a fixed number of processes
            # chunksize=1,        # tune this if you have many tiny files
        )

        # Merge each returned dict into our “cc” counter
        for file_dict in per_file_counts:
            cc.token_counter.update(file_dict)
        
        cc.save_token_counts(csv_out)



# The entry point of your script - if a user runs it from the command line, for example using `python -m <package>.<module>`
# or `python <script_path>.py`, this is what will be run.
if __name__ == "__main__":
    main_cli()
