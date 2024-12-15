from boarddetection import RedBoardDetector
import cv2

size = (800, 800)

image = cv2.imread('images/board.jpg')
cv2.imshow('image', cv2.resize(image, size))

board = RedBoardDetector.detect(image)
cv2.imshow('board', cv2.resize(board, size))
cv2.waitKey(0)