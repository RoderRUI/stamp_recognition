import cv2
import numpy as np
import paddlehub as hub
import re
import decimal
import statistics
from matplotlib import pyplot as plt


def __get_signature_text(res, res_results, form):
    ###
    # raw_signature = ''.join(
    #     re.compile('[^\u4e00-\u9fa5]').split(res.get("text").split(re.compile('[：:]'))[-1].strip()))
    if re.compile('[：:]').search(res.get("text").strip()):
        raw_signature = re.compile('[：:]').split(res.get("text").strip())[-1]
        # 如果手写签名被识别到同一个text中，直接返回
        # if raw_signature and re.compile('^[\u4E00-\u9FA5]{2,4}$').match(raw_signature):
        if raw_signature:
            return raw_signature
        else:
            pass
    # 否则，对附近的区域进行检测单独的签名
    # else:
    if form == 1:
        low_x_box_position = res.get('text_box_position')[1][0] - 50
        # high_x_box_position = res.get('text_box_position')[1][0]
        low_y_box_position = res.get('text_box_position')[1][1] - 150
        high_y_box_position = res.get('text_box_position')[2][1] + 150
        for reco in res_results:
            min_y_box_position = min(reco.get('text_box_position')[0][1], reco.get('text_box_position')[1][1])
            max_y_box_position = max(reco.get('text_box_position')[2][1], reco.get('text_box_position')[3][1])
            min_x_box_position = min(reco.get('text_box_position')[0][0], reco.get('text_box_position')[3][0])
            max_height = max(reco.get('text_box_position')[3][1] - reco.get('text_box_position')[0][1],
                             reco.get('text_box_position')[2][1] - reco.get('text_box_position')[1][1])
            if len(reco.get(
                    'text')) < 8 and min_x_box_position > low_x_box_position and min_y_box_position > low_y_box_position and max_y_box_position < high_y_box_position and max_height > 70:
                raw_signature = reco.get('text')
                # if re.compile('^[\u4E00-\u9FA5]{2,4}$').match(raw_signature):
                return raw_signature
    else:
        raw_signature = 0
        high_y_box_position = res.get('text_box_position')[2][1]
        # high_x_box_position = res.get('text_box_position')[1][0]
        low_x_box_position = res.get('text_box_position')[3][0]
        high_x_box_position = res.get('text_box_position')[2][0]
        for reco in res_results:
            min_y_box_position = min(reco.get('text_box_position')[0][1], reco.get('text_box_position')[1][1])
            # max_y_box_position = max(reco.get('text_box_position')[2][1], reco.get('text_box_position')[3][1])
            min_x_box_position = min(reco.get('text_box_position')[0][0], reco.get('text_box_position')[3][0])
            max_height = max(reco.get('text_box_position')[3][1] - reco.get('text_box_position')[0][1],
                             reco.get('text_box_position')[2][1] - reco.get('text_box_position')[1][1])
            if min_x_box_position < low_x_box_position and min_y_box_position > high_y_box_position and max_height > 70:
                # raw_signature = reco.get('text')
                # if re.compile('^[\u4E00-\u9FA5]{2,4}$').match(raw_signature):
                # return raw_signature
                raw_signature += 1
        return raw_signature
    return ''


def __transfer_signature_date_format_to_standard(res, res_results):
    """
    将手写的日期格式转化为标准格式，用于对比时间，因为提取出的为纯数字的格式，如：2020911
    """
    raw_date = res.get('text').strip()
    if re.match("^.*年.*月.*日.*$", raw_date):
        form = 1
    else:
        form = 2
    valid_time = ''
    raw_date = re.compile('[：:]').split(raw_date)[-1]
    if form == 1:
        year_index = raw_date.index('年')
        moth_index = raw_date.index('月')
        day_index = raw_date.index('日')
        year = raw_date[:year_index]
        month = raw_date[year_index + 1:moth_index]
        day = raw_date[moth_index + 1:day_index]
        # if str.isdigit(year) and str.isdigit(month) and str.isdigit(day):
        valid_time = f'{year}-{month}-{day}'
    else:
        if raw_date:
            date_list = re.compile(r'\D+').split(raw_date)
            if len(date_list) == 3:
                valid_time = '-'.join(date_list).strip()
        else:
            low_x_box_position = res.get('text_box_position')[1][0] - 50
            # high_x_box_position = res.get('text_box_position')[1][0]
            low_y_box_position = res.get('text_box_position')[1][1] - 150
            high_y_box_position = res.get('text_box_position')[2][1] + 150
            for reco in res_results:
                min_y_box_position = min(reco.get('text_box_position')[0][1], reco.get('text_box_position')[1][1])
                max_y_box_position = max(reco.get('text_box_position')[2][1], reco.get('text_box_position')[3][1])
                min_x_box_position = min(reco.get('text_box_position')[0][0], reco.get('text_box_position')[3][0])
                max_height = max(reco.get('text_box_position')[3][1] - reco.get('text_box_position')[0][1],
                                 reco.get('text_box_position')[2][1] - reco.get('text_box_position')[1][1])
                if len(reco.get(
                        'text')) <= 10 and min_x_box_position > low_x_box_position and min_y_box_position > low_y_box_position and max_y_box_position < high_y_box_position and max_height > 60:
                    raw_signature = reco.get('text')
                    # if re.compile('^[\u4E00-\u9FA5]{2,4}$').match(raw_signature):
                    return raw_signature
    # if valid_time != '' and re.match(r"^\d{1,4}-(?:1[0-2]|0?[1-9])-(?:0?[1-9]|[1-2]\d|30|31)$", valid_time):
    return valid_time


def remove_red_ocr(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    ##
    window_name = "image"
    cv2.namedWindow(window_name, 0)
    cv2.resizeWindow(window_name, 742, 1056)

    blue_img, green_img, red_img = cv2.split(img)
    # print(2)
    # cv2.imshow(window_name, red_img)
    # cv2.waitKey(0)
    ret3, th3 = cv2.threshold(red_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_inv = cv2.bitwise_not(th3)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.dilate(mask_inv, kernel, iterations=1)
    # dilation = cv2.erode(erosion, kernel, iterations=1)
    # print(6)
    # cv2.imshow(window_name, erosion)
    # cv2.waitKey(0)
    # print(3)
    # cv2.imshow(window_name, th3)
    # cv2.waitKey(0)
    # ret_img = np.expand_dims(th3, axis=2)
    # ret_img = np.concatenate((ret_img, ret_img, ret_img), axis=-1)
    # result_img = np.stack((th3, th3, th3), axis=2)
    result_img = cv2.cvtColor(erosion, cv2.COLOR_BGR2RGB)
    print(4)

    # cv2.imshow(window_name, result_img)
    # cv2.waitKey(0)

    # dilation = cv2.erode(erosion, kernel, iterations=1)
    # print(7)
    # cv2.imshow(window_name, dilation)
    # cv2.waitKey(0)
    # closing = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow(window_name, closing)
    # cv2.waitKey(0)
    # return 0
    ##

    # 2. 获取识别结果列表
    # chinese_ocr_db_crnn_server识别率更好，但是花费更长时间，基于服务器性能考虑，采用mobile轻量级库
    # text_detector = hub.Module(name="chinese_text_detection_db_server", enable_mkldnn=True)
    ocr = hub.Module(name="chinese_ocr_db_crnn_server")  # mkldnn加速仅在CPU下有效
    result = ocr.recognize_text(images=[result_img], visualization=False,
                                use_gpu=False, box_thresh=0.5,
                                text_thresh=0.5,
                                angle_classification_thresh=0.9)
    reco_results = result[0]['data']
    # return 0
    for item in reco_results:
        cv2.rectangle(img, item['text_box_position'][1], item['text_box_position'][3], (255, 0, 0), 4, 8)
        print(f"text:{item['text']}, confidence:{item['confidence']}, box:{item['text_box_position']}")
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    return 0
    calc_res = []
    date_signature_num = 0
    name_signature_num = 0
    raw_signature = 0
    date_signature = ''
    name_signature = ''
    valid_amount = ''
    for res in reco_results:
        # if re.match("^.*日期.*[:：].*$", res.get('text')) or re.match("^.*年.*月.*日.*$", res.get('text')):
        #     valid_time = __transfer_signature_date_format_to_standard(res, reco_results)
        #     if valid_time:
        #         date_signature_num += 1
        #         date_signature = date_signature + valid_time + '、'
        #         # cv2.rectangle(img, res['text_box_position'][0], res['text_box_position'][2], (255, 0, 0), 4, 8)
        # if re.match("^.*(?:签名|签字|专家组组长|负责人|经理|专家|用户确认).*$", res.get('text').strip()):
        #     raw_signature = __get_signature_text(res, reco_results)
        #     if raw_signature:
        #         name_signature_num += 1
        #         name_signature = name_signature + raw_signature + '、'
        # if re.match(".*中标价.*", res.get('text').strip()):
        #     # raw_amount = __get_bid_amount(res, reco_results)
        #     amount_str = re.compile('[:：]').split(res.get('text'))[-1]
        #     raw_amount = amount_str[:amount_str.index('万元')].strip()
        #     if re.compile(r'^\d+(\.\d+)?$').match(raw_amount):
        #         valid_amount = decimal.Decimal(raw_amount) * 10000
        if re.match(".*姓名.*", res.get('text').strip()):
            # raw_amount = __get_bid_amount(res, reco_results)
            raw_signature = __get_signature_text(res, reco_results, 2)
            # if raw_signature:
            #     name_signature_num += 1
            #     name_signature = name_signature + raw_signature + '、'
            # cv2.rectangle(img, res_tmp['text_box_position'][0], res_tmp['text_box_position'][2], (255, 0, 0), 4, 8)
    # if date_signature_num >= 2 and name_signature_num >= 2:
    #     print(1)
    # cv2.imwrite(img_path, img)
    print(date_signature)
    print(raw_signature)
    print(valid_amount)
    # calc_res.append({'check_status': 3,
    #                  'result_message': f'识别到{date_signature_num}个签订日期：{date_signature[:-1]}，{name_signature_num}位签订人：{name_signature[:-1]}'})
    # calc_res.append({'id': item['document_content_rule_detail_id'], 'check_status': 3,
    #                  'result_message': f'签订日期：{date_signature[:-1]}，签订人：{name_signature[:-1]}'})
    # print(5)
    # cv2.imshow(window_name, img)
    # cv2.waitKey(0)
    # for res in reco_results:
    #     if re.match(".*合同价格.*", res.get('text').strip()):
    #         # raw_amount = "".join(list(filter(str.isdigit, res.get('text'))))
    #         amount_match = re.search('￥.*元', res.get('text'))
    #         if amount_match:
    #             print(1)
    # if re.match(".*中标金.*", res.get('text').strip()):
    #     reco_result = res
    #     low_x = reco_result.get('text_box_position')[0][0] - 100
    #     high_x = reco_result.get('text_box_position')[1][0] + 100
    #     low_y = reco_result.get('text_box_position')[3][1]
    #     high_y = reco_result.get('text_box_position')[3][1] + 370
    #     for reco in reco_results:
    #         if low_x < reco.get('text_box_position')[0][0] and reco.get('text_box_position')[1][0] < high_x and reco.get('text_box_position')[0][1] > low_y and reco.get('text_box_position')[1][1] < high_y:
    #             if re.compile(r'^\d+(\.\d+)?$').match(reco.get('text').strip()):
    #                 d = decimal.Decimal(reco.get('text').strip()) * 10000
    #                 print(d)
    # num_arr = [tmp['text_box_position'][0][0] for tmp in reco_results]
    # low = statistics.median_low(num_arr)
    # high = statistics.median_high(num_arr)
    print("finish")
    # hsv = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
    # cv2.imshow(window_name, hsv)
    # cv2.waitKey(0)


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


def black_stamp(img):
    # img_t = cv2.imread(r"D:\Ease\9.png", cv2.IMREAD_COLOR)
    window_name = "image"
    rows, cols, _ = img.shape
    img_w = 650
    scale = img_w / cols
    resized_img = cv2.resize(img, (img_w, int(rows * scale)), interpolation=cv2.INTER_LINEAR)
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
    print(3)
    cv2.imshow(window_name, th3)
    cv2.waitKey(0)
    filter_condition = int(ret3 * 0.95)
    ret1, th1 = cv2.threshold(red_img, filter_condition, 255, cv2.THRESH_BINARY)
    print(3)
    cv2.imshow(window_name, th1)
    cv2.waitKey(0)
    # ret_img = np.expand_dims(th1, axis=2)
    # ret_img = np.concatenate((ret_img, ret_img, ret_img), axis=-1)
    # print(4)
    # cv2.imshow(window_name, ret_img)
    # cv2.waitKey(0)
    # ret3, th3 = cv2.threshold(red_img, 190, 255, cv2.THRESH_BINARY)
    # mask_inv = cv2.bitwise_not(mask)
    # cv2.imshow(window_name, mask)
    # cv2.waitKey(0)
    # cv2.imshow(window_name, th3)
    # cv2.waitKey(0)
    # return 0

    img = cv2.bitwise_not(th1)
    print(4)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)

    # edges = cv2.Canny(img, 50, 150)
    # print(5)
    # cv2.imshow(window_name, edges)
    # cv2.waitKey(0)
    # img = cv2.medianBlur(mask_inv, 5)

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    print(6)
    cv2.imshow(window_name, blur)
    cv2.waitKey(0)

    # kernel = np.ones((7, 7), np.uint8)
    # closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # erosion = cv2.dilate(img, kernel, iterations=1)
    # print(7)
    # cv2.imshow(window_name, closing)
    # cv2.waitKey(0)

    # dilation = cv2.erode(img, kernel, iterations=1)
    # print(7)
    # cv2.imshow(window_name, dilation)
    # cv2.waitKey(0)
    # return 0
    # img = th3
    cimg = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
    # cv2.imshow(window_name, cimg)
    # cv2.waitKey(0)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 150,
                               param1=100, param2=60, minRadius=40, maxRadius=100)
    print(circles)
    if circles is not None:
        row, row1, row2 = circles.shape
        print(row)
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
