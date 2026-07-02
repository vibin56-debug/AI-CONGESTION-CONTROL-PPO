import csv
from datetime import datetime

with open("../data/raw/network_metrics.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "timestamp",
        "rtt_ms",
        "cwnd",
        "throughput_mbps"
    ])

    print("Dataset initialized.")
