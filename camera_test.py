import cv2


cap = cv2.VideoCapture(2)  # Change index if needed
ret, frame = cap.read()
if ret:
    cv2.imwrite("now_image.jpg", frame)
    print("saved")
cap.release()
cv2.destroyAllWindows()
