from eeg_raw_to_classification.pipelines.dataset2bids import pipeline_dataset2bids
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the dataset to BIDS conversion pipeline.')
    parser.add_argument('pipeline_file', type=str, help='Path to the pipeline YAML file')
    args = parser.parse_args()
    pipeline_dataset2bids(args.pipeline_file)