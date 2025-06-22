import argparse
from eeg_raw_to_classification.pipelines.preprocessing import pipeline_preprocess

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess MEEG data.')
    parser.add_argument('pipeline_yml', type=str, help='Path to the pipeline YAML file.')
    parser.add_argument('--max_files', type=int, default=None, help='Maximum number of files to process.')
    parser.add_argument('--external_jobs', type=int, default=1, help='Number of external jobs.')
    parser.add_argument('--internal_jobs', type=int, default=1, help='Number of internal jobs.')
    parser.add_argument('--retry_errors', action='store_true', help='Retry files that had errors.')
    parser.add_argument('--raise_on_error', action='store_true', help='Enable raise_on_error mode.')
    parser.add_argument('--index', type=int, default=None, help='Index of the file to process. Total index taking into account the dataset outer loop. Only works with external_jobs=1.')
    parser.add_argument('--only_total', action='store_true', help='Just get the total number of files. Only works with external_jobs=1.')
    args = parser.parse_args()

    pipeline_preprocess(args.pipeline_yml, max_files=args.max_files, external_njobs=args.external_jobs,
                        internal_njobs=args.internal_jobs, retry_errors=args.retry_errors,
                        raise_on_error=args.raise_on_error, index=args.index, only_total=args.only_total)
