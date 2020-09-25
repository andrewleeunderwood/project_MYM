import numpy as np
import chess
import chess.svg
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
                               fen_to_image)


def rescale_frame(frame, percent=75):
    # width = int(frame.shape[1] * (percent / 100))
    # height = int(frame.shape[0] * (percent / 100))
    dim = (1000, 750)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


model = load_model('model_VGG16.h5')

cap = cv2.VideoCapture(0)

# start = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'
blank = '8/8/8/8/8/8/8/8'
board = fen_to_image(blank)
board_image = cv2.imread('current_board.png')
cv2.imshow('current board', board_image)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    small_frame = rescale_frame(frame)

    # Display the resulting frame
    cv2.imshow('live', small_frame)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        print('Working...')

        cv2.imwrite('frame.jpeg', frame)
        img, gray_blur = read_img('frame.jpeg')
        edges = canny_edge(gray_blur)
        lines = hough_line(edges)
        h_lines, v_lines = h_v_lines(lines)
        intersection_points = line_intersections(h_lines, v_lines)
        points = cluster_points(intersection_points)
        points = augment_points(points)
        x_list = write_crop_images(img, points, 0)
        img_filename_list = grab_cell_files()
        img_filename_list.sort(key=natural_keys)
        fen = classify_cells(model, img_filename_list)
        board = fen_to_image(fen)
        print(board)

        board_image = cv2.imread('current_board.png')
        cv2.imshow('current board', board_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
