import cv2
import numpy as np


def my_stamp():
    img_t = cv2.imread(r"D:\Ease\9.png", cv2.IMREAD_COLOR)
    window_name = "image"
    source_img_shape = img_t.shape
    img_w = 650
    scale = img_w / source_img_shape[1]
    resized_img = cv2.resize(img_t, (img_w, int(source_img_shape[0] * scale)), interpolation=cv2.INTER_LINEAR)
    # cv2.imshow(window_name, resized_img)
    # cv2.waitKey(0)

    red_low = np.array([156, 43, 46])
    red_high = np.array([179, 255, 255])
    hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, red_low, red_high)
    # 测试圆

    # img = cv2.medianBlur(th3, 5)
    mask_inv = cv2.bitwise_not(mask)
    # img = mask_inv
    img = cv2.medianBlur(mask_inv, 5)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # cv2.imshow(window_name, cimg)
    # cv2.waitKey(0)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 50,
                               param1=1, param2=8, minRadius=45, maxRadius=70)
    print(circles)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    cv2.imshow(window_name, cimg)
    cv2.waitKey(0)


    # mask_inv = cv2.bitwise_not(mask)
    # cv2.imshow(window_name, mask)
    # cv2.waitKey(0)
    # cv2.imshow(window_name, mask_inv)
    # cv2.waitKey(0)
    #
    # ret3, th3 = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    # cv2.imshow(window_name, th3)
    # cv2.waitKey(0)
    # 高斯滤波+大津法
    # blur = cv2.GaussianBlur(resized_img, (5, 5), 0)
    # ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow(window_name, blur)
    # cv2.waitKey(0)


def black_stamp():
    img_t = cv2.imread(r"D:\Ease\9.png", 1)
    window_name = "image"
    source_img_shape = img_t.shape
    img_w = 650
    scale = img_w / source_img_shape[1]
    resized_img = cv2.resize(img_t, (img_w, int(source_img_shape[0] * scale)), interpolation=cv2.INTER_LINEAR)
    cv2.imshow(window_name, resized_img)
    cv2.waitKey(0)

    blue_img, green_img, red_img = cv2.split(resized_img)
    # black_low = np.array([0, 0, 0])
    # black_high = np.array([169, 169, 169])
    # hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(hsv_img, black_low, black_high)
    ret3, th3 = cv2.threshold(red_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # mask_inv = cv2.bitwise_not(mask)
    # cv2.imshow(window_name, mask)
    # cv2.waitKey(0)
    cv2.imshow(window_name, th3)
    cv2.waitKey(0)

    # img = cv2.medianBlur(th3, 5)
    img = th3
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imshow(window_name, cimg)
    cv2.waitKey(0)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=10, minRadius=100, maxRadius=100)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    cv2.imshow(window_name, cimg)
    cv2.waitKey(0)
    # filter_condition = int(ret3 * 0.95)
    # ret1, th1 = cv2.threshold(th3, filter_condition, 255, cv2.THRESH_BINARY)
    # cv2.imshow(window_name, th1)
    # cv2.waitKey(0)
    # ret_img = np.expand_dims(th1, axis=2)
    # ret_img = np.concatenate((ret_img, ret_img, ret_img), axis=-1)
    # cv2.imshow(window_name, ret_img)
    # cv2.waitKey(0)
    # blur = cv2.GaussianBlur(resized_img, (5, 5), 0)

    # ret3, th3 = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    # cv2.imshow(window_name, th3)
    # cv2.waitKey(0)
    # ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow(window_name, blur)
    # cv2.waitKey(0)


if __name__ == '__main__':
    my_stamp()
