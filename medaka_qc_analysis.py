import os
import sys
import argparse

sys.path.append("src")

import analysis as analyse

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", help = "Results from medaka_bpm", type = str, required = True)
    parser.add_argument("-o", "--out_dir",  help = "Directory to write analysis results", type = str, required = True)
    return parser.parse_args()

def main():
    args = process_args()
    raw_data = analyse.pd.read_csv(args.input_file)
    data, scale = analyse.process_data(raw_data, 20)
    classifier = analyse.decision_tree(data)
    limits = analyse.get_thresholds(raw_data, analyse.qc_features, classifier)
    analyse.write_results(raw_data, data, classifier, limits, args.out_dir)
    
if __name__ == "__main__":
    main()