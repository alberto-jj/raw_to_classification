import argparse
from eeg_raw_to_classification.pipelines.inspect import pipeline_inspect

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inspect MEEG datasets.')
    parser.add_argument('pipeline_file', type=str, help='Path to the pipeline.yml file')
    parser.add_argument('--max_files', type=int, default=None, help='Maximum number of files to process per dataset')
    args = parser.parse_args()
    pipeline_inspect(args.pipeline_file, max_files=args.max_files)