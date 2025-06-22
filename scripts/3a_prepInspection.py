import argparse
from eeg_raw_to_classification.pipelines.inspect_prep import pipeline_inspect_prep
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the MEEG preprocessing inspection.')
    parser.add_argument('pipeline_file', type=str, help='Path to the pipeline.yml file')
    args = parser.parse_args()
    pipeline_inspect_prep(args.pipeline_file)
