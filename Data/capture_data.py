import numpy as np
import cv2

def rescale_frame(frame, percent=75):
    # width = int(frame.shape[1] * (percent / 100))
    # height = int(frame.shape[0] * (percent / 100))
    dim = (1000, 750)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


cap = cv2.VideoCapture(1)

capture_num = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small_frame = rescale_frame(frame)

    # Display the resulting frame
    cv2.imshow('frame',  frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('frame' + str(capture_num) + '.jpeg', frame)
        print('Saved ' + str(capture_num))
        capture_num += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()