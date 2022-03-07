# import the necessary packages
from collections import deque
import numpy as np
import cv2
import fire


def process(video_path, output=False):
    # Initializations
    thresh_flags = cv2.THRESH_BINARY
    thresh_flags += cv2.THRESH_OTSU

    width = 600
    height = None
    dim = None
    max_white_count_for_frame_diff_usable = 20_000
    linesLower = (80, 0, 0)
    linesUpper = (110, 255, 255)

    previous_frame = None
    current_frame = None

    # Grabbing inputed video
    vs = cv2.VideoCapture(video_path)

    # Starting video output to disk
    if output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_video = cv2.VideoWriter('output.mp4', fourcc, 20.0, (600, 337))

    while True:
        frame = vs.read()[1]

        if dim is None:
            scalar = width/frame.shape[1]
            height = int(frame.shape[0] * scalar)
            dim = (width, height)

        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        current_frame = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Finding ball and players
        frame_diff = cv2.absdiff(current_frame, previous_frame) if previous_frame is not None else frame
        frame_diff = cv2.Canny(frame_diff, 50, 200)
        _, frame_diff = cv2.threshold(frame_diff, 0, 255, thresh_flags)
        frame_diff = cv2.dilate(frame_diff, None, iterations=4)

        ball_contours, _ = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[-2:]
        ball_contours = sorted(ball_contours, key=lambda x: cv2.moments(x)["m00"], reverse=True)
        frame_diff = cv2.cvtColor(frame_diff, cv2.COLOR_GRAY2BGR)

        # Drawing players contours
        frame_diff_white_count = np.sum(frame_diff == 255)
        if frame_diff_white_count < max_white_count_for_frame_diff_usable:
            for contour in ball_contours[:2]:
                cv2.drawContours(frame, [contour], 0, (200, 0, 255), 3)
            ball_contours = ball_contours[2:]

        # Finding court lines
        blurred = cv2.GaussianBlur(hsv, (11, 11), 0)
        lines_mask = cv2.inRange(blurred, linesLower, linesUpper)
        contours, _ = cv2.findContours(lines_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[-2:]

        center_contour = None
        min_dist = np.inf
        min_area = 50_000
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            if rect[1][0] * rect[1][1] < min_area:
                continue
            dist = compare_to_frame_center(rect[0], width, height)
            if dist < min_dist:
                center_contour = contour
                min_dist = dist

        # Drawing court lines and ball
        if center_contour is not None:
            approx = cv2.approxPolyDP(center_contour, 6, True)
            if len(approx) < 16:
                cv2.drawContours(frame, [approx], 0, (0, 0, 255), 3)
                ball_candidates_coords = []
                if frame_diff_white_count < max_white_count_for_frame_diff_usable:
                    for contour in ball_contours:
                        M = cv2.moments(contour)
                        point = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                        if cv2.pointPolygonTest(center_contour, point, False) <= 0:
                            continue
                        ball_candidates_coords.append(point)
                    if len(ball_candidates_coords) > 0:
                        cv2.circle(frame, point, 3, (0, 255, 0), -1)
                    elif len(ball_contours) > 0:
                        M = cv2.moments(ball_contours[0])
                        cv2.circle(frame, point, 3, (0, 255, 0), -1)

        if output:
            out_video.write(frame)

        # Display frame
        cv2.imshow("Out", frame)
        key = cv2.waitKey(1) & 0xFF
        # If the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break

        # Updating previous frame
        previous_frame = current_frame

    vs.release()
    if output:
        out_video.release()
    # close all windows
    cv2.destroyAllWindows()


def compare_to_frame_center(point, frame_w, frame_h):
    return abs(point[0] - frame_w/2) + abs(point[1] - frame_h/2)


def closer_to_coord(points, coord):
    if points is None:
        raise Exception("points variable is None")
    elif coord is None:
        raise Exception("coord variable is None")
    if len(points) == 0:
        raise Exception("points array is empty")
    elif len(coord) == 0:
        raise Exception("coord array is empty")

    dists = [np.linalg.norm(np.array(point)-np.array(coord)) for point in points]
    return np.array(points[np.argmin(dists)])


if __name__ == "__main__":
    fire.Fire(process)