from Recognizer import *
import os


# 用于获取验证码模板
# 使用方法:
#        运行程序后，点击分割出来的验证码字符按任意键（例如ESC）
#        然后控制台会出现 "该字符是:" 后 输入该字符的正确值（例如图片是Y 你就输入Y）后回车
#        再点击新分割出来的验证码字符图片重复操作即可。

while True:
    OriginalImage = GetNewCaptcha()
    Img = RemoveNoise(OriginalImage)  # 去除验证码噪点

    OriginalImage = cv2.imread(OriginalImage)

    OpeningKernel = np.ones((3, 3), np.uint8)  # 获取自定义开运算核
    Openning = cv2.morphologyEx(Img, cv2.MORPH_OPEN, OpeningKernel)  # 进行开运算

    Thresh = cv2.threshold(Openning, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]  # 进行图像阈值处理
    Thresh = cv2.threshold(Thresh, 0, 255, cv2.THRESH_BINARY_INV)[1]  # 反转一下颜色

    ThreshCnts, Hierarchy = cv2.findContours(Thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓

    Cnts = ThreshCnts

    TempImg = Img.copy()
    cv2.drawContours(TempImg, Cnts, -1, (0, 0, 255), 3)  # 画出轮廓

    ResultCaptcha = {}

    for (i, c) in enumerate(Cnts):  # 过滤寻找到的轮廓

        x, y, w, h = cv2.boundingRect(c)

        if w > 10 and h > 10:  # 过滤掉面积很小的轮廓 肯定不是验证码字符

            cv2.rectangle(OriginalImage, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 画出寻找到的四个字符 后续显示用

            SingleCharacterImg = Thresh[y:y + h, x:x + w]  # 分割出单个字符

            Contours, Hierarchy = cv2.findContours(SingleCharacterImg.copy(), cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
            SingleCharacterCnts = Contours

            SingleCharacterImgCopy = SingleCharacterImg.copy()
            SingleCharacterImgCopy = cv2.morphologyEx(SingleCharacterImgCopy, cv2.MORPH_OPEN, OpeningKernel)

            cv2.imshow('',SingleCharacterImgCopy)
            cv2.waitKey(0)
            character = str(input("该字符是:"))
            cv2.destroyAllWindows()
            path = './Template/' + character + '/'
            if not os.path.exists(path):
                os.makedirs(path)
            salt = ''.join(random.sample(string.ascii_letters + string.digits, 8))
            cv2.imwrite(path + salt + '.png' , SingleCharacterImgCopy.copy())
            print("验证码模板字符 \"" + character + "\" 已保存至: " + path + salt + '.png')