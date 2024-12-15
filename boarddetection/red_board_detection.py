import cv2
import numpy as np
from .types import BoardDetector

class RedBoardDetector(BoardDetector):
    def detect(image):
        
        board = RedBoardDetector.isolateBoard(image)

        return board

    def isolateBoard(image):
        '''
        Get overall area of board
        No guarantee it will just contain the playing area
        General technique is:
            1. Preprocess with a blur
            2. Use an HSV mask to identify the red pixels
            3. Find the largest contour, which should be the board
            4. Roughly find the corenrs and warp the image to a 800x800 square
        '''
        blurred = cv2.GaussianBlur(image, (7, 7), 0)    # blur the image
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)   # convert to HSV
        
        # filter to where: s > 50, v > 0, h < 35 or h > 330
        lower = np.array([0, 127, 127], np.uint8)
        upper = np.array([35, 255, 255], np.uint8)
        mask1 = cv2.inRange(hsv, lower, upper)

        lower = np.array([130, 127, 127], np.uint8)
        upper = np.array([255, 255, 255], np.uint8)
        mask2 = cv2.inRange(hsv, lower, upper)

        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1) # increase mask size to fill in gaps

        # cv2.imshow('mask', cv2.resize(mask, (800, 800)))
        # cv2.waitKey(0)

        # todo: handle failure
        # find the largest contour (ideally the board)
        contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        board = max(contours, key=cv2.contourArea)

        # cv2.drawContours(image, [board], -1, (0, 255, 0), 3)
        # cv2.imshow('contours', cv2.resize(image, (800, 800)))
        # cv2.waitKey(0)

        corners = RedBoardDetector.getCorners([tuple(c[0]) for c in board])

        # colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255)]
        # for i, corner in enumerate(corners):
        #     cv2.circle(image, (int(corner[0]), int(corner[1])), 10, colors[i], -1)
        # cv2.imshow('corners', cv2.resize(image, (800, 800)))
        # cv2.waitKey(0)

        # use corners to isolate the board in 800x800 image
        width = 800
        height = 800
        dst = np.array([[0, height], [width, height], [width, 0], [0, 0]], np.float32)
        M = cv2.getPerspectiveTransform(corners, dst)
        image = cv2.warpPerspective(image, M, (width, height))

        return image

    def getCorners(points):
        '''
        Get the four corners of a set of points making up a quadrilateral
        Returns in order of bottom-left, bottom-right, top-right, top-left
        '''

        # x, y weights for the corneres
        # in the order of bottom-left, bottom-right, top-right, top-left
        weights = [(-2, 5), (2, 5), (2, -5), (-2, -5)]

        lowest = min(points, key=lambda x: x[1])
        leftmost = min(points, key=lambda x: x[0])
        corners = [-1, -1, -1, -1]
        values = [None, None, None, None]

        for p in points:
            for i, (xWeight, yWeight) in enumerate(weights):
                value = (p[0] - leftmost[0]) * xWeight + (p[1] - lowest[1]) * yWeight
                if values[i] is None or values[i] < value:
                    values[i] = value
                    corners[i] = p
        
        return np.array(corners, np.float32)