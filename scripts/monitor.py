import os
import subprocess
import csv
import re
import time
from datetime import datetime

# CHANGE THIS IF h2 PID CHANGES
PID = 1614

CSV_FILE = "../data/raw/network_metrics.csv"

# mnexec needs root to attach to another process's network namespace.
# Run this script itself with `sudo python3 monitor.py` -- if it isn't
# already root, sudo can't prompt for a password from inside the
# subprocess loop below and every call silently fails, which is why
# RTT/CWND/THR all showed 0 before.
if os.geteuid() != 0:
    raise SystemExit(
        "monitor.py must be run as root, e.g.: sudo python3 monitor.py"
    )

with open(CSV_FILE, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "timestamp",
        "rtt_ms",
        "cwnd",
        "throughput_mbps"
    ])

    print("Collecting Metrics...")

    while True:

        try:

            cmd = f"mnexec -a {PID} ss -ti"

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True
            )

            output = result.stdout
            print("\n========== RAW OUTPUT ==========")
            print(output)
            if result.returncode != 0 or result.stderr:
                print("---- STDERR ----")
                print(result.stderr)
            print("================================")

            # ss -ti prints one connection per block: a summary line
            # starting at column 0, followed by an indented details
            # line. Split on "newline followed by non-whitespace" so
            # each block stays intact, then only look at ESTAB blocks.
            blocks = re.split(r"\n(?=\S)", output)
            estab_blocks = [b for b in blocks if b.startswith("ESTAB")]

            # A single iperf3 run opens more than one socket (a mostly
            # idle control channel plus the actual data stream). Mixing
            # fields across sockets pairs the wrong RTT with the wrong
            # CWND/throughput, so pick ONE block -- the one actually
            # carrying data (largest bytes_sent) -- and read every
            # field from that same block.
            def bytes_sent_of(block):
                m = re.search(r"bytes_sent:(\d+)", block)
                return int(m.group(1)) if m else -1

            rtt = cwnd = throughput = 0
            if estab_blocks:
                data_block = max(estab_blocks, key=bytes_sent_of)

                rtt_match = re.search(r"rtt:(\d+\.\d+)", data_block)
                cwnd_match = re.search(r"cwnd:(\d+)", data_block)
                rate_match = re.search(
                    r"delivery_rate\s+([\d\.]+)([KMGkmg]?)bps", data_block
                )

                _unit_to_mbps = {"": 1e-6, "K": 1e-3, "M": 1, "G": 1e3}

                rtt = float(rtt_match.group(1)) if rtt_match else 0
                cwnd = int(cwnd_match.group(1)) if cwnd_match else 0
                throughput = (
                    float(rate_match.group(1))
                    * _unit_to_mbps[rate_match.group(2).upper()]
                    if rate_match else 0
                )

            timestamp = datetime.now()

            writer.writerow([
                timestamp,
                rtt,
                cwnd,
                throughput
            ])

            f.flush()

            print(
                f"RTT={rtt:.2f}ms | "
                f"CWND={cwnd} | "
                f"THR={throughput:.2f}Mbps"
            )

            time.sleep(1)

        except KeyboardInterrupt:

            print("\nStopped")

            break

        except Exception as e:

            print("Error:", e)

            time.sleep(1)
