import argparse
from eeg_raw_to_classification.pipelines.features import pipeline_features
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MEEG feature extraction pipeline.')
    parser.add_argument('pipeline_file', type=str, help='Path to the pipeline YAML file.')
    parser.add_argument('--external_jobs', type=int, default=1, help='Number of external jobs for parallel processing.')
    parser.add_argument('--raise_on_error', action='store_true', help='Raise on error if set.')
    parser.add_argument('--retry_errors', action='store_true', help='Retry files that had errors.')
    parser.add_argument('--index', type=int, default=None, help='Index of the file to process. Total index taking into account the dataset outer loop.')
    parser.add_argument('--only_total', action='store_true', help='Just get the total number of files.')
    parser.add_argument('--inspect_only', action='store_true', help='Just get the status of the features for each file.')

    args = parser.parse_args()
    inspect_list = pipeline_features(args.pipeline_file, args.external_jobs, args.raise_on_error, args.external_jobs > 1, args.retry_errors, args.index, args.only_total, args.inspect_only)

