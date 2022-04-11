import os


TEST_SET_SIZE = 0.3

PLOT_SAVE_DIR = os.path.relpath("figs")
DATA_SAVE_DIR = os.path.relpath("thresholds")


LABELS = "error"

qc_features = ["HROI Change Intensity", "Harmonic Intensity", "Heart size", "Movement detection max", "SNR", "Signal intensity", "Signal regional prominence", "Intensity/Harmonic Intensity (top 5 %)", "SNR Top 5%", "Signal Intensity Top 5%"]