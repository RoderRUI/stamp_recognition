import cv2
import numpy as np
from matplotlib import pyplot as plt


def my_stamp():
    img_t = cv2.imread(r"D:\Ease\1.png", cv2.IMREAD_COLOR)
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
    rows = hsv_img.shape

    mask = hsv_img - mask
    cv2.imshow(window_name, mask)
    cv2.waitKey(0)
    return 0
    # 测试圆

    img = cv2.medianBlur(resized_img, 5)
    # mask_inv = cv2.bitwise_not(resized_img)
    # img = resized_img
    # edges = cv2.Canny(img, 100, 200)
    # img = cv2.medianBlur(mask_inv, 5)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imshow(window_name, cimg)
    cv2.waitKey(0)
    # cv2.imshow(window_name, cimg)
    # cv2.waitKey(0)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 150,
                               param1=100, param2=75, minRadius=45, maxRadius=70)
    print(circles)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (36, 51, 235), 2)
        # draw the center of the circle
        # cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
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
    img_t = cv2.imread(r"D:\Ease\9.png", cv2.IMREAD_COLOR)
    window_name = "image"
    rows, cols, _ = img_t.shape
    img_w = 650
    scale = img_w / cols
    resized_img = cv2.resize(img_t, (img_w, int(rows * scale)), interpolation=cv2.INTER_LINEAR)
    print(1)
    cv2.imshow(window_name, resized_img)
    cv2.waitKey(0)
    # hsv = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
    # cv2.imshow(window_name, hsv)
    # cv2.waitKey(0)

    blue_img, green_img, red_img = cv2.split(resized_img)
    # rows, cols, _ = resized_img.shape
    # pixelSequence = red_img.reshape([rows * cols, ])
    # # 计算直方图
    # plt.figure()
    # # manager = plt.get_current_fig_manager()
    # # manager.windows.state('zoomed')
    #
    # histogram, bins, patch = plt.hist(pixelSequence, 256, facecolor='black', histtype='bar')  # facecolor设置为黑色
    #
    # # 设置坐标范围
    # y_maxValue = np.max(histogram)
    # plt.axis([0, 255, 0, y_maxValue])
    # # 设置坐标轴
    # plt.xlabel("gray Level", fontsize=20)
    # plt.ylabel('number of pixels', fontsize=20)
    # plt.title("Histgram of red channel", fontsize=25)
    # plt.xticks(range(0, 255, 10))
    # # 显示直方图
    # plt.pause(0.05)
    # plt.savefig("histgram.png", dpi=260, bbox_inches="tight")
    # plt.show()

    # cv2.imshow(window_name, blue_img)
    # cv2.waitKey(0)
    # cv2.imshow(window_name, green_img)
    # cv2.waitKey(0)
    print(2)
    cv2.imshow(window_name, red_img)
    cv2.waitKey(0)
    # black_low = np.array([0, 0, 0])
    # black_high = np.array([169, 169, 169])
    # hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(hsv_img, black_low, black_high)
    # blur = cv2.GaussianBlur(red_img, (5, 5), 0)
    # cv2.imshow(window_name, blur)
    # cv2.waitKey(0)
    ret3, th3 = cv2.threshold(red_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    filter_condition = int(ret3 * 0.95)
    ret1, th1 = cv2.threshold(red_img, filter_condition, 255, cv2.THRESH_BINARY)
    print(3)
    cv2.imshow(window_name, th1)
    cv2.waitKey(0)
    ret_img = np.expand_dims(th1, axis=2)
    ret_img = np.concatenate((ret_img, ret_img, ret_img), axis=-1)
    print(4)
    cv2.imshow(window_name, ret_img)
    cv2.waitKey(0)
    # ret3, th3 = cv2.threshold(red_img, 190, 255, cv2.THRESH_BINARY)
    # mask_inv = cv2.bitwise_not(mask)
    # cv2.imshow(window_name, mask)
    # cv2.waitKey(0)
    # cv2.imshow(window_name, th3)
    # cv2.waitKey(0)
    # return 0

    img = cv2.GaussianBlur(th1, (5, 5), 0)
    print(5)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    # img = th3
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # cv2.imshow(window_name, cimg)
    # cv2.waitKey(0)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 150,
                               param1=100, param2=45, minRadius=40, maxRadius=100)
    # print(circles)
    # if circles:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    print(6)
    cv2.imshow(window_name, cimg)
    cv2.waitKey(0)
    # except Exception as e:
    #     print(e)
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
    # my_stamp()
    black_stamp()
