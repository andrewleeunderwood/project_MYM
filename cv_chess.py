import re
import cv2
from keras.models import load_model
from cvchess_functions import (read_img,
                               canny_edge,
                               hough_line,
                               h_v_lines,
                               line_intersections,
                               cluster_points,
                               augment_points,
                               write_crop_images,
                               grab_cell_files,
                               classify_cells,
                               fen_to_image,
                               atoi)


# Resize the frame by scale by dimensions
def rescale_frame(frame, percent=75):
    # width = int(frame.shape[1] * (percent / 100))
    # height = int(frame.shape[0] * (percent / 100))
    dim = (1000, 750)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


# Find the number(s) in the text
def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


# Load in the CNN model
model = load_model('model_VGG16.h5')

# Select the live video stream source (0-webcam & 1-GoPro)
cap = cv2.VideoCapture(1)

# Show the starting board either as blank or with the initial setup
# start = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'
blank = '8/8/8/8/8/8/8/8'
board = fen_to_image(blank)
board_image = cv2.imread('current_board.png')
cv2.imshow('current board', board_image)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resizes each frame
    small_frame = rescale_frame(frame)

    # Display the resulting frame
    cv2.imshow('live', small_frame)

    if cv2.waitKey(1) & 0xFF == ord(' '):

        print('Working...')
        # Save the frame to be analyzed
        cv2.imwrite('frame.jpeg', frame)
        # Low-level CV techniques (grayscale & blur)
        img, gray_blur = read_img('frame.jpeg')
        # Canny algorithm
        edges = canny_edge(gray_blur)
        # Hough Transform
        lines = hough_line(edges)
        # Separate the lines into vertical and horizontal lines
        h_lines, v_lines = h_v_lines(lines)
        # Find and cluster the intersecting
        intersection_points = line_intersections(h_lines, v_lines)
        points = cluster_points(intersection_points)
        # Final coordinates of the board
        points = augment_points(points)
        # Crop the squares of the board a organize into a sorted list
        x_list = write_crop_images(img, points, 0)
        img_filename_list = grab_cell_files()
        img_filename_list.sort(key=natural_keys)
        # Classify each square and output the board in Forsyth-Edwards Notation (FEN)
        fen = classify_cells(model, img_filename_list)
        # Create and save the board image from the FEN
        board = fen_to_image(fen)
        # Display the board in ASCII
        print(board)
        # Display and save the board image
        board_image = cv2.imread('current_board.png')
        cv2.imshow('current board', board_image)
        print('Completed!')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # End the program
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
