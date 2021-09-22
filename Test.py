import Recognizer

Recognizer = Recognizer.Recognizer()
while True:
    # 调用RecognizeCaptcha 识别验证码 第一个参数为需要识别的验证码图片路径 第二个参数为是否输出函数调试信息 返回值：验证码模板匹配数据 识别结果 分割识别后的图片
    # GetNewCaptcha 是通过Request库获取新的上学吧验证码 返回验证码图片路径
    ResultData, ResultStr, ResultImg, TookTime = Recognizer.RecognizeCaptcha(Recognizer.GetNewCaptcha(), False)
    print('识别数据：' + str(ResultData), "\n识别结果: " + ResultStr, "\n识别耗时:" + str(TookTime) + "s")
    Recognizer.show(ResultStr, ResultImg)
