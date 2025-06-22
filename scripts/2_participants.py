import argparse
from eeg_raw_to_classification.pipelines.participants import pipeline_participants
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process pipeline file.')
    parser.add_argument('pipeline_file', type=str, help='Path to the pipeline YAML file')
    args = parser.parse_args()
    pipeline_participants(args.pipeline_file)