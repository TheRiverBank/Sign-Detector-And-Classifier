import cv2
import config
import numpy as np

def get_side_bars(img):
    """
    Cut out the left and right sides of a frame.
    :param img:
    :return:
    """
    N, M, C = img.shape

    left_bar = img[:, 10:70, :]
    right_bar = img[:, M-70:M, :]

    return (left_bar, right_bar)

def get_upper_bar():
    pass

def get_lower_bar():
    pass

def get_rects(imgs: tuple):
    N, M, C = imgs[0].shape
    n_imgs = len(imgs)
    print(n_imgs)
    rects = []
    print(N, M)

    for img in imgs:
        for y in range(0, N - config.RECT_SIZE_Y, config.RECT_SIZE_Y//2):
            for x in range(0, M - config.RECT_SIZE_X, config.RECT_SIZE_X//2):
                rect = img[y: y + config.RECT_SIZE_Y, x: x + config.RECT_SIZE_X, :]
                rects.append(rect)

    return rects

if __name__ == '__main__':
    test_img = cv2.imread("./test_images/40_test.png")
    side_bars = get_side_bars(test_img)
    rects = get_rects((side_bars[0], side_bars[1]))


    for rect in rects:
        cv2.imshow("img", rect)
        cv2.waitKey(0)

