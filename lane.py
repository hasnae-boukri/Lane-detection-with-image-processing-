import cv2
import numpy as np

def region_of_intrests(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def draw_lines(image, lines, color=(0, 255, 0), thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)

def detect_lanes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    height, width = edges.shape
    roi_vertices = [(0, height), (width / 2, height / 2), (width, height) ]
    roi = region_of_intrests(edges, np.array([roi_vertices], np.int32))

    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)
    line_image = np.zeros_like(image)
    draw_lines(line_image, lines)

    result = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    return result
    return line_image
input_path = "LaneVideo.mp4"
input_type = input_path.split(".")[-1]

if input_type in ['jpg', 'png', 'jpeg']:
    image = cv2.imread(input_path)
    result = detect_lanes(image)
    cv2.namedWindow('Lane Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Lane Detection', 800, 600)
    cv2.imshow('Lane Detection', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif input_type in ['mp4', 'avi', 'mkv']:
    cap = cv2.VideoCapture(input_path)
    cv2.namedWindow('Lane Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Lane Detection', 800, 600)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result = detect_lanes(frame)
        cv2.imshow('Lane Detection', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

else:
    print("Unsupported input type. Please provide an image or video file.")