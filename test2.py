import cv2
import numpy as np

def nothing(x):
    pass

# Trackbars for color range adjustment
def create_trackbars(window_name):
    cv2.namedWindow(window_name)
    cv2.createTrackbar("L - H", window_name, 20, 179, nothing)
    cv2.createTrackbar("L - S", window_name, 100, 255, nothing)
    cv2.createTrackbar("L - V", window_name, 100, 255, nothing)
    cv2.createTrackbar("U - H", window_name, 40, 179, nothing)
    cv2.createTrackbar("U - S", window_name, 255, 255, nothing)
    cv2.createTrackbar("U - V", window_name, 255, 255, nothing)

def get_trackbar_values(window_name):
    l_h = cv2.getTrackbarPos("L - H", window_name)
    l_s = cv2.getTrackbarPos("L - S", window_name)
    l_v = cv2.getTrackbarPos("L - V", window_name)
    u_h = cv2.getTrackbarPos("U - H", window_name)
    u_s = cv2.getTrackbarPos("U - S", window_name)
    u_v = cv2.getTrackbarPos("U - V", window_name)
    return np.array([l_h, l_s, l_v]), np.array([u_h, u_s, u_v])

def detect_lanes(frame, mask):
    # Sliding window parameters
    window_height = 40
    window_width = 80
    midpoint = int(mask.shape[1] / 2)
    left_lane_pts = []
    right_lane_pts = []

    # Iterate over the mask from bottom to top
    for y in range(mask.shape[0] - window_height, 0, -window_height):
        # Left lane sliding window
        histogram = np.sum(mask[y:y + window_height, 0:midpoint], axis=0)
        left_base = np.argmax(histogram)

        # Right lane sliding window
        histogram = np.sum(mask[y:y + window_height, midpoint:], axis=0)
        right_base = np.argmax(histogram) + midpoint

        # Check if intensity sum is above threshold (indicating a lane segment)
        if np.sum(histogram) > window_height * window_width * 20:
            left_lane_pts.append((left_base + window_width // 2, y + window_height // 2))
        if np.sum(histogram[midpoint:]) > window_height * window_width * 20:
            right_lane_pts.append((right_base + window_width // 2, y + window_height // 2))

    # Draw detected lane lines (if any)
    if left_lane_pts:
        for pt in left_lane_pts:
            cv2.circle(frame, pt, 5, (0, 255, 0), thickness=-1)
    if right_lane_pts:
        for pt in right_lane_pts:
            cv2.circle(frame, pt, 5, (0, 255, 0), thickness=-1)

    return frame

def main():
    # Use camera (index 0) instead of reading from a video file
    vidcap = cv2.VideoCapture(0)

    create_trackbars("Trackbars")

    while True:
        success, image = vidcap.read()
        if not success:
            break

        frame = cv2.resize(image, (640, 480))

        ## Choosing points for perspective transformation
        tl = (150, 300)
        bl = (100, 460)
        tr = (550, 300)
        br = (600, 460)

        cv2.circle(frame, tl, 5, (0, 0, 255), -1)
        cv2.circle(frame, bl, 5, (0, 0, 255), -1)
        cv2.circle(frame, tr, 5, (0, 0, 255), -1)
        cv2.circle(frame, br, 5, (0, 0, 255), -1)

        ## Applying perspective transformation
        pts1 = np.float32([tl, bl, tr, br])
        pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])

        # Matrix to warp the image for birdseye window
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        transformed_frame = cv2.warpPerspective(frame, matrix, (640, 480))

        ### Object Detection
        # Convert to HSV, get trackbar values, define color range
        hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)
        lower_yellow, upper_yellow = get_trackbar_values("Trackbars")

        # Mask for yellow color
        yellow_mask = cv2.inRange(hsv_transformed_frame, lower_yellow, upper_yellow)

        # Detect lanes using sliding windows
        lanes_detected = detect_lanes(transformed_frame.copy(), yellow_mask)

        # Display frames
        cv2.imshow("Original", frame)
        cv2.imshow("Yellow Mask", yellow_mask)
        cv2.imshow("Lanes Detected", lanes_detected)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    vidcap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
