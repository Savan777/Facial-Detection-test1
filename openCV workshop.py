import numpy as np
import cv2

cap = cv2.VideoCapture(0) # replace 0 with file name of video to capture video
first_frame = None

while (cap.isOpened()):
    ret, img_original = cap.read()
    frame_draw = img_original.copy()
    img_grayscale = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(img_grayscale, (21, 21), 0)

    # Initialize the first frame
    if first_frame is None:
        first_frame = gray.copy()
        continue

    # Compute difference between frames
    frame_delta = cv2.absdiff(first_frame, gray)
    frame_thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilate
    kernel_dilate = np.ones((10, 10), np.uint8)
    motion_dilate = cv2.dilate(frame_thresh, kernel_dilate, iterations=2)

    # Get contours of binary image
    m2, contours, hierarchy = cv2.findContours(motion_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on frame_draw
    cv2.drawContours(frame_draw, contours, -1, (0, 255, 0), 3)

    # Use this frame for the next iteration
    first_frame = gray.copy()

    cv2.imshow('original', img_original)
    cv2.imshow('binary motion', frame_thresh)
    cv2.imshow('motion dilate', motion_dilate)
    cv2.imshow('frame_draw', frame_draw)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()