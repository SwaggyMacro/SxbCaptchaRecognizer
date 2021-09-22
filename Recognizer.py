import os
import random
import string
import numpy as np
import cv2
import requests
from skimage import morphology
import matplotlib.pyplot as plt
import time

class Recognizer:

    def show(self, title, img):
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __init__(self):
        self.TemplateData, self.labels = self.getTemplateByContour()  # 加载验证码模板
        print("Loaded TemplateData")
    def GetNewCaptcha(self):
        import os
        if not os.path.exists('./TempCaptcha/'):
            os.makedirs('./TempCaptcha/')
        salt = ''.join(random.sample(string.ascii_letters + string.digits, 8))
        salt = './TempCaptcha/' + salt + '.jpg'
        with open(salt, 'wb') as w:
            r = requests.get('https://passport.shangxueba.com/VerifyCode.aspx')
            w.write(r.content)
        return salt


    def noise_remove_cv2(self, image_name, k):
        """
        8邻域降噪
        Args:
            image_name: 图片文件命名
            k: 判断阈值

        Returns:

        """

        def calculate_noise_count(img_obj, w, h):
            """
            计算邻域非白色的个数
            Args:
                img_obj: img obj
                w: width
                h: height
            Returns:
                count (int)
            """
            count = 0
            width, height = img_obj.shape
            for _w_ in [w - 1, w, w + 1]:
                for _h_ in [h - 1, h, h + 1]:
                    if _w_ > width - 1:
                        continue
                    if _h_ > height - 1:
                        continue
                    if _w_ == w and _h_ == h:
                        continue
                    if img_obj[_w_, _h_] < 230:  # 二值化的图片设置为255
                        count += 1
            return count

        img = cv2.imread(image_name, 1)
        # 灰度
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        w, h = gray_img.shape
        for _w in range(w):
            for _h in range(h):
                if _w == 0 or _h == 0:
                    gray_img[_w, _h] = 255
                    continue
                # 计算邻域pixel值小于255的个数
                pixel = gray_img[_w, _h]
                if pixel == 255:
                    continue

                if calculate_noise_count(gray_img, _w, _h) < k:
                    gray_img[_w, _h] = 255

        return gray_img

    def resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
            dim = None
            (h, w) = image.shape[:2]
            if width is None and height is None:
                return image
            if width is None:
                r = height / float(h)
                dim = (int(w * r), height)
            else:
                r = width / float(w)
                dim = (width, int(h * r))
            resized = cv2.resize(image, dim, interpolation=inter)
            return resized

    def getTemplateByContour(self): # 获取验证码模板
        TemplateData = []
        labels = []
        for root,dirs,files in os.walk('./Template/'):
            for dir in dirs:
                listImg = './Template/' + str(dir) + '/'
                data = os.listdir(listImg)  # 列出文件夹下所有的目录与文件
                character = []
                for j in range(0, len(data)):
                    path = os.path.join(listImg, data[j])
                    # if os.path.isfile(path):
                    tempImg = cv2.imread(path)
                    ref = cv2.cvtColor(tempImg, cv2.COLOR_BGR2GRAY)
                    ref = cv2.threshold(ref, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                    character.append(ref)
                TemplateData.append(character)
                labels.append(dir)
        return TemplateData, labels

    def RemoveNoise(self, img, SHOWIMAGE = False):
        img = self.noise_remove_cv2(img, 4)
        # 转换为灰度图
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 转换为二值图
        ret, thresh1 = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)
        # ret, thresh1 = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # 转换为布尔值
        thresh1 = thresh1 > 1

        # 去除外部噪点
        stage1 = morphology.remove_small_objects(thresh1, min_size=120, connectivity=2)

        # 去除内部孔洞，注意到第二个参数为area_threshold,而不是min_size
        stage2 = morphology.remove_small_holes(stage1, area_threshold=64, connectivity=1)

        stage2 = stage2 * 255
        stage2 = np.array(stage2, dtype=np.uint8)
        if SHOWIMAGE:
            plt.figure()
            plt.subplot(1, 4, 1)
            plt.imshow(img, cmap='gray')
            plt.title('Original')
            plt.subplot(1, 4, 2)
            plt.imshow(thresh1, cmap='gray')
            plt.title('Thresh')
            plt.subplot(1, 4, 3)
            plt.imshow(stage1, cmap='gray')
            plt.title('Remove OBJ')
            plt.subplot(1, 4, 4)
            plt.imshow(stage2, cmap='gray')
            plt.title('Remove Hole')
            plt.savefig('test1.png')
            plt.show()
            # show('',stage2)
            # show('',cv2.imread('code1.png'))

        return stage2


    def RecognizeCaptcha(self, Img, DebugPrint = False):
        StartTime = time.time() # 获取识别函数开始时间
        OriginalImage = Img
        Img = self.RemoveNoise(OriginalImage) # 去除验证码噪点

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
                # cv2.drawContours(cur_img, cnts1, -1, (0, 0, 0), -1)  # 画出轮廓

                for (index, template) in enumerate(self.TemplateData):  # 遍历所有模板图片进行模板匹配

                    for TemplateImg in template:  # TemplaData是个二维列表，第二维度才是图片数据

                        TemplateImg = cv2.resize(TemplateImg, (
                        SingleCharacterImgCopy.shape[1], SingleCharacterImgCopy.shape[0]))  # Resize 模板大小《= 原图大小

                        Result = cv2.matchTemplate(SingleCharacterImgCopy, TemplateImg, cv2.TM_CCOEFF_NORMED)  # 模板匹配
                        (_, Score, _, _) = cv2.minMaxLoc(Result)
                        # 模板匹配后的数值如果大于0.7的话就算识别成功，加入到存储识别验证码的列表
                        if Score > 0.7:
                            if ResultCaptcha.get(x) is not None:  # 不存在识别到的X坐标则新建，字典索引根据X坐标决定
                                if Score > ResultCaptcha[x]['score']:  # 如果Score准确率大于上一个字符的匹配结果则更新
                                    if DebugPrint:
                                        print(
                                        "ResultUpdate Str:{0} NewScore:{1} LastScore:{2} X:{3}".format(self.labels[index], Score,
                                                                                                       ResultCaptcha[x][
                                                                                                           'score'],
                                                                                                       x))  # 输出识别率超过0.7并且大于上一次识别的的结果
                                    ResultCaptcha[x] = {'str': self.labels[index], 'score': Score}
                            else:
                                ResultCaptcha[x] = {'str': self.labels[index], 'score': Score}
                                if DebugPrint:
                                    print("Result {0}'s Score:{1}".format(self.labels[index], Score))

        ResultCaptchaIdx = sorted(ResultCaptcha) # 排序一下字典，因为字典索引是X坐标，所以排序后就是正常的验证码顺序了

        # 遍历所有数组元素 把识别出来的字符提取出来
        ResultData = []
        for i in ResultCaptchaIdx:
            ResultData.append(ResultCaptcha[i])

        ResultStr = ""
        for i in ResultData: # 插入单个识别结果字符 组成完整验证码
            ResultStr = ResultStr + i['str']
        TookTime = time.time() - StartTime
        return ResultData, ResultStr, OriginalImage, TookTime
