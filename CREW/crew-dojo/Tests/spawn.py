# spawn and align human interfaces on the screen

import os
import sys
import time
import signal
import psutil
import argparse
import subprocess
import tkinter as tk
from screeninfo import get_monitors
from typing import List


def get_valid_screen_size(offset):
    # test maximized window size
    root = tk.Tk()
    root.geometry(f"100x100+{offset[0]}+{offset[1]}")
    if sys.platform in ("win32", "darwin"):
        root.state("zoomed")
        bar_height = 32  # hard-coded
    else:
        root.attributes("-zoomed", True)
        bar_height = 36  # hard-coded
    root.update_idletasks()
    # get viable screen area
    size = [
        root.winfo_width(),
        root.winfo_height() + bar_height,
        bar_height,
    ]
    print(size, root.winfo_rooty())
    root.destroy()
    return size


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Human Interface Spawner")
    parser.add_argument("path", help="Path to Unity executable", type=str)
    parser.add_argument("-rows", help="Number of rows", type=int, default=2)
    parser.add_argument("-cols", help="Number of columns", type=int, default=2)
    parser.add_argument("-monitor", help="Monitor ID", type=int, default=0)
    parser.add_argument(
        "-menubottom",
        help="Is desktop bar on the bottom of screen?",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()
    assert args.rows > 0 and args.cols > 0
    assert os.path.isfile(args.path)

    monitors = get_monitors()
    assert len(monitors) > args.monitor

    monitorOffset = (monitors[args.monitor].x, monitors[args.monitor].y)
    resolution = get_valid_screen_size(monitorOffset)
    screenSize = (
        resolution[0] // args.cols,
        resolution[1] // args.rows - resolution[2],
    )

    print(screenSize)

    ps: List[psutil.Popen] = []
    commands = [
        args.path,
        "-DojoMonitorID",
        f"{args.monitor}",
        "-DojoScreenSize",
        f"{screenSize[0]}x{screenSize[1]}",
    ]

    for r in range(args.rows):
        for c in range(args.cols):
            screenPos = (
                c * screenSize[0],
                r * screenSize[1] + (r + (0 if args.menubottom else 1)) * resolution[2],
            )
            fullCommands = commands + [
                "-DojoScreenPos",
                f"{screenPos[0]},{screenPos[1]}",
            ]
            p = subprocess.Popen(fullCommands, shell=False)
            ps.append(p)
            print(f"Spawned at ({r},{c})")
            time.sleep(0.1)

    print("All processes started! CTRL+C to shutdown all!")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping processes...")
        for p in ps:
            try:
                process = psutil.Process(p.pid)
                for proc in process.children(recursive=True):
                    proc.send_signal(signal.SIGTERM)
                process.send_signal(signal.SIGTERM)
            except Exception:
                pass
        print("All processes stopped!")
