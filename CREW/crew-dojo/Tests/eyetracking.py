import time
import argparse
import tkinter as tk
from typing import List
import numpy as np
import tobii_research
from tobiiresearch.implementation.EyeTracker import EyeTracker


class TrackerDevice:
    def __init__(self, verbose: bool = False):
        # find all connected eye trackers
        eye_trackers: List[EyeTracker] = tobii_research.find_all_eyetrackers()

        print("Found %d eye trackers" % len(eye_trackers))
        for eye_tracker in eye_trackers:
            print("Address: " + eye_tracker.address)
            print("Model: " + eye_tracker.model)
            print("Serial number: " + eye_tracker.serial_number)
            print()

        self.left_point = (np.nan, np.nan)
        self.right_point = (np.nan, np.nan)
        self.left_size = np.nan
        self.right_size = np.nan
        self.left_openness = np.nan
        self.right_openness = np.nan

        self.verbose = verbose

        if len(eye_trackers) == 0:
            self.initialized = False

        else:
            self.initialized = True
            self.eye_tracker = eye_trackers[0]

            # subscribe to the gaze data from the eye tracker
            self.eye_tracker.subscribe_to(
                tobii_research.EYETRACKER_GAZE_DATA,
                self.gaze_data_callback,
                as_dictionary=True,
            )

            # subscribe to the eye openness data from the eye tracker
            self.eye_tracker.subscribe_to(
                tobii_research.EYETRACKER_EYE_OPENNESS_DATA,
                self.eye_openness_callback,
                as_dictionary=True,
            )

        self.step = 0

    def gaze_data_callback(self, gaze_data):
        self.left_point = gaze_data["left_gaze_point_on_display_area"]
        self.right_point = gaze_data["right_gaze_point_on_display_area"]

        self.left_size = gaze_data["left_pupil_diameter"]
        self.right_size = gaze_data["right_pupil_diameter"]

        timestamp = gaze_data["system_time_stamp"]

        if self.verbose:
            print(
                f"GAZE ({timestamp}) Left eye: {self.left_point} size = {self.left_size}, Right eye: {self.right_point} size = {self.right_size}"
            )

    def eye_openness_callback(self, eye_openness_data):
        self.left_openness = eye_openness_data["left_eye_openness_value"]
        self.right_openness = eye_openness_data["right_eye_openness_value"]

        timestamp = eye_openness_data["system_time_stamp"]

        if self.verbose:
            print(
                f"OPENNESS ({timestamp}) Left eye: {self.left_openness}, Right eye: {self.right_openness}"
            )

    @property
    def left_eye_open(self):
        return not np.isnan(self.left_size)

    @property
    def right_eye_open(self):
        return not np.isnan(self.right_size)


def test_eye_openness(tracker: TrackerDevice):
    window = tk.Tk()

    window.attributes("-fullscreen", True)
    window.configure(background="white")
    window.bind("<Escape>", lambda _: window.quit())

    label = tk.Label(window, text="Test 1: Eye Openness", font=("Helvetica", 32))
    label.pack(pady=20)

    def callback():
        if tracker.step == 0:
            label.configure(text="Close your left eye. Open your right eye.")
            label.update()
            time.sleep(4)
            if not tracker.left_eye_open and tracker.right_eye_open:
                label.configure(text="Success!")
                print("Left eye closed, right eye open (Passed)")

            else:
                label.configure(text="Failure!")
                print("Left eye closed, right eye open (Failed)")
            label.update()

        elif tracker.step == 1:
            label.configure(text="Open your left eye. Close your right eye.")
            label.update()
            time.sleep(4)
            if tracker.left_eye_open and not tracker.right_eye_open:
                label.configure(text="Success!")
                print("Left eye open, right eye closed (Passed)")

            else:
                label.configure(text="Failure!")
                print("Left eye open, right eye closed (Failed)")
            label.update()

        tracker.step += 1
        if tracker.step == 2:
            window.quit()

    button = tk.Button(window, text="Step", command=callback, font=("Helvetica", 32))
    button.pack(pady=20, side=tk.BOTTOM)

    window.mainloop()


def test_eye_point(tracker: TrackerDevice):
    window = tk.Tk()

    window.attributes("-fullscreen", True)
    window.configure(background="white")
    window.bind("<Escape>", lambda _: window.quit())

    print(f"Screen size: {window.winfo_screenwidth()}, {window.winfo_screenheight()}")

    canvas = tk.Canvas(window, bg="white", highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)

    label = tk.Label(window, text="Test 2: Eye Point Accuracy", font=("Helvetica", 32))
    label.place(relx=0.5, y=20, anchor=tk.N)

    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    def draw_circle(x, y, r, color, fill=True):
        x0 = x - r
        y0 = y - r
        x1 = x + r
        y1 = y + r
        if fill:
            return canvas.create_oval(x0, y0, x1, y1, fill=color)
        else:
            return canvas.create_oval(x0, y0, x1, y1, outline=color)

    def compute_diff(x, y, pointX, pointY):
        diffX = x - pointX
        diffY = y - pointY
        return (diffX * diffX + diffY * diffY) ** 0.5

    def callback():
        if tracker.step == 0:
            label.configure(text="Look at the top left corner of the screen.")
            x, y = 100, 100
            draw_circle(x, x, 20, "green")
            canvas.update()
            label.update()
            time.sleep(2)
            if tracker.left_eye_open and tracker.right_eye_open:
                leftX = tracker.left_point[0] * screen_width
                leftY = tracker.left_point[1] * screen_height
                rightX = tracker.right_point[0] * screen_width
                rightY = tracker.right_point[1] * screen_height

                left_diff = compute_diff(x, y, leftX, leftY)
                right_diff = compute_diff(x, y, rightX, rightY)
                print(f"Test 1 ({x}, {y}): ({left_diff}, {right_diff})")

                draw_circle(leftX, leftY, left_diff, "red", False)
                draw_circle(rightX, rightY, right_diff, "blue", False)

            else:
                print(f"Test 1 ({x}, {y}): (Failed)")

        elif tracker.step == 1:
            label.configure(text="Look at the top right corner of the screen.")
            x, y = screen_width - 100, 100
            draw_circle(x, y, 20, "green")
            canvas.update()
            label.update()
            time.sleep(2)
            if tracker.left_eye_open and tracker.right_eye_open:
                leftX = tracker.left_point[0] * screen_width
                leftY = tracker.left_point[1] * screen_height
                rightX = tracker.right_point[0] * screen_width
                rightY = tracker.right_point[1] * screen_height

                left_diff = compute_diff(x, y, leftX, leftY)
                right_diff = compute_diff(x, y, rightX, rightY)
                print(f"Test 2 ({x}, {y}): ({left_diff}, {right_diff})")

                draw_circle(leftX, leftY, left_diff, "red", False)
                draw_circle(rightX, rightY, right_diff, "blue", False)

            else:
                print(f"Test 2 ({x}, {y}): (Failed)")

        elif tracker.step == 2:
            label.configure(text="Look at the bottom left corner of the screen.")
            x, y = 100, screen_height - 100
            draw_circle(x, y, 20, "green")
            canvas.update()
            label.update()
            time.sleep(2)
            if tracker.left_eye_open and tracker.right_eye_open:
                leftX = tracker.left_point[0] * screen_width
                leftY = tracker.left_point[1] * screen_height
                rightX = tracker.right_point[0] * screen_width
                rightY = tracker.right_point[1] * screen_height

                left_diff = compute_diff(x, y, leftX, leftY)
                right_diff = compute_diff(x, y, rightX, rightY)
                print(f"Test 3 ({x}, {y}): ({left_diff}, {right_diff})")

                draw_circle(leftX, leftY, left_diff, "red", False)
                draw_circle(rightX, rightY, right_diff, "blue", False)

            else:
                print(f"Test 3 ({x}, {y}): (Failed)")

        elif tracker.step == 3:
            label.configure(text="Look at the bottom right corner of the screen.")
            x, y = screen_width - 100, screen_height - 100
            draw_circle(x, y, 20, "green")
            canvas.update()
            label.update()
            time.sleep(2)
            if tracker.left_eye_open and tracker.right_eye_open:
                leftX = tracker.left_point[0] * screen_width
                leftY = tracker.left_point[1] * screen_height
                rightX = tracker.right_point[0] * screen_width
                rightY = tracker.right_point[1] * screen_height

                left_diff = compute_diff(x, y, leftX, leftY)
                right_diff = compute_diff(x, y, rightX, rightY)
                print(f"Test 4 ({x}, {y}): ({left_diff}, {right_diff})")

                draw_circle(leftX, leftY, left_diff, "red", False)
                draw_circle(rightX, rightY, right_diff, "blue", False)

            else:
                print(f"Test 4 ({x}, {y}): (Failed)")

        label.configure(text="Recorded!")
        label.update()
        canvas.update()

        tracker.step += 1
        if tracker.step == 4:
            window.quit()

    button = tk.Button(window, text="Step", command=callback, font=("Helvetica", 32))
    button.place(relx=0.5, y=window.winfo_screenheight() - 100, anchor=tk.N)

    window.mainloop()


def test_eye_diameter(tracker: TrackerDevice):
    window = tk.Tk()

    window.attributes("-fullscreen", True)
    window.configure(background="white")
    window.bind("<Escape>", lambda _: window.quit())

    label = tk.Label(window, text="Test 3: Eye Diameter", font=("Helvetica", 32))
    label.pack(pady=20)

    def callback():
        if tracker.step == 0:
            label.configure(text="Close the light in the room.")
            label.update()
            window.configure(background="black")
            window.update()
            time.sleep(5)
            print("Light off")
            print("Left eye size: %f" % tracker.left_size)
            print("Right eye size: %f" % tracker.right_size)

        elif tracker.step == 1:
            label.configure(text="Open the light in the room.")
            label.update()
            window.configure(background="white")
            window.update()
            time.sleep(5)
            print("Light on")
            print("Left eye size: %f" % tracker.left_size)
            print("Right eye size: %f" % tracker.right_size)

        label.configure(text="Recorded!")
        label.update()

        tracker.step += 1
        if tracker.step == 2:
            window.quit()

    button = tk.Button(window, text="Step", command=callback, font=("Helvetica", 32))
    button.pack(pady=20, side=tk.BOTTOM)

    window.mainloop()


def test_sudoku(tracker: TrackerDevice):
    window = tk.Tk()
    window.attributes("-fullscreen", True)
    window.configure(background="white")
    label = tk.Label(window, text="Look at all number 5", font=("Helvetica", 32))
    label.pack(pady=20)
    label.update()
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    time.sleep(4)
    window.destroy()

    sub_width = screen_width // 5
    sub_height = screen_height // 2

    windows = []
    five_positions = []
    for i in range(2):
        for j in range(5):
            w = tk.Tk()
            w.geometry(f"{sub_width}x{sub_height}+{j * sub_width}+{i * sub_height}")
            w.configure(background="white")
            windows.append(w)

            arr = np.arange(1, 10)
            np.random.shuffle(arr)

            for m in range(3):
                for n in range(3):
                    if arr[m * 3 + n] == 5:
                        five_positions.append((i, j, m, n))

                    label = tk.Label(
                        w,
                        text=str(arr[m * 3 + n]),
                        font=("Helvetica", 32),
                        background="white",
                    )
                    label.place(
                        relx=(n + 0.5) / 3, rely=(m + 0.5) / 3, anchor=tk.CENTER
                    )

            w.update()

    time.sleep(2)

    accuracy_left = 0
    accuracy_right = 0
    unitX = screen_width // 5 // 3
    unitY = screen_height // 2 // 3

    for w, (i, j, m, n) in zip(windows, five_positions):
        print(f"Look at {i} {j} {m} {n}")
        w.configure(background="green")
        w.update()
        time.sleep(1)
        w.configure(background="white")
        w.update()
        time.sleep(4)

        left_eye = tracker.left_point
        right_eye = tracker.right_point

        if not tracker.left_eye_open or not tracker.right_eye_open:
            continue

        # check if is looking at 5
        leftX = left_eye[0] * screen_width - j * sub_width - n * sub_width / 3
        leftY = left_eye[1] * screen_height - i * sub_height - m * sub_height / 3
        rightX = right_eye[0] * screen_width - j * sub_width - n * sub_width / 3
        rightY = right_eye[1] * screen_height - i * sub_height - m * sub_height / 3

        if leftX > 0 and leftX < unitX and leftY > 0 and leftY < unitY:
            accuracy_left += 1

        if rightX > 0 and rightX < unitX and rightY > 0 and rightY < unitY:
            accuracy_right += 1

    print("Accuracy left: %f" % (accuracy_left / 10))
    print("Accuracy right: %f" % (accuracy_right / 10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--test",
        help="Run a test",
        choices=["openness", "point", "diameter", "sudoku"],
        required=True,
    )
    parser.add_argument("-v", "--verbose", help="Verbose output", action="store_true")

    args = parser.parse_args()
    tracker = TrackerDevice(verbose=args.verbose)

    if args.test == "openness":
        test_eye_openness(tracker)

    elif args.test == "point":
        test_eye_point(tracker)

    elif args.test == "diameter":
        test_eye_diameter(tracker)

    elif args.test == "sudoku":
        test_sudoku(tracker)
