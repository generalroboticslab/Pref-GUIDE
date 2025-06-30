# the sytess test script

import sys
import json
import time
import psutil
import argparse
import subprocess
from typing import List


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Spawn Unity Instances")
    parser.add_argument(
        "-nm", "--num_machine", help="Number of Machine Instances", type=int, default=10
    )
    parser.add_argument(
        "-nh", "--num_human", help="Number of Human Instances", type=int, default=1
    )

    args = parser.parse_args()

    with open("stress.json", "r") as inFile:
        configs = json.load(inFile)

    ps: List[psutil.Popen] = []

    serverExePath = configs["server"][sys.platform]
    print(f"Spawning server instance ({serverExePath})")

    commands = [serverExePath]
    ps.append(subprocess.Popen(commands, shell=False))
    time.sleep(5)

    clientExePath = configs["client"][sys.platform]
    print(f"Spawning {args.num_machine} machine Unity instances ({clientExePath})")

    commands = [clientExePath, "-headless", "-AgentPort", "55555", "-AgentID", "Test"]
    for i in range(args.num_machine):
        p = subprocess.Popen(commands + [str(i)], shell=False)
        ps.append(p)
        print(f"[{i}] {p.pid}")
        time.sleep(1)

    print(f"Spawning {args.num_human} human Unity instances ({clientExePath})")

    commands = [clientExePath]
    for i in range(args.num_human):
        p = subprocess.Popen(commands, shell=False)
        ps.append(p)
        print(f"[{i}] {p.pid}")
        time.sleep(1)

    ps.reverse()  # quit server last

    print("All processes started!")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping processes...")
        for p in ps:
            print(f"{p.pid}")
            try:
                process = psutil.Process(p.pid)
                for proc in process.children(recursive=True):
                    proc.kill()
                process.kill()
            except Exception:
                pass
        print("All processes stopped!")
