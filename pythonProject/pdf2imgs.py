import fitz
import os
import shutil


def transfer_png(img_path, pg, pdf_path, trans):
    pdf = fitz.open(pdf_path)
    # 设置缩放和旋转系数
    pm = pdf[pg].getPixmap(matrix=trans, alpha=False)
    # 开始写图像
    pm.writePNG(img_path + str(pg + 1) + ".png")
    # pm.writePNG(imgPath)
    pdf.close()


def pdf_image(pdf_path, img_path, zoom_x=5, zoom_y=5, rotation_angle=0):
    set_dir(img_path)
    # 打开PDF文件
    # for pdf_path in pdf_paths:
    pdf = fitz.open(pdf_path)
    count = pdf.pageCount
    pdf.close()
    # pool = Pool(cpu_count())
    trans = fitz.Matrix(zoom_x, zoom_y).preRotate(rotation_angle)
    # 逐页读取PDF
    for pg in range(0, count):
        # pool.apply_async(transfer_png, (img_path, pg, pdf_path, trans),
        #                  error_callback=error)
        transfer_png(img_path, pg, pdf_path, trans)
    # pool.close()
    # pool.join()
    return count


def set_dir(filepath):
    """
    如果文件夹不存在就创建，如果文件存在就清空
    param filepath:需要创建的文件夹路径
    """
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)
