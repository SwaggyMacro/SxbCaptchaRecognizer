# 项目说明
项目是前两天帮朋友做的，识别准确率大概80%左右。

项目用处不大，但是做都做了就发出来了。
  
用的模板匹配的方式（项目要求），主要就是降噪、开运算、二值化，然后findContours分割出字符进行模板匹配。

总共有1193张模板图片，可以执行GetTemplate.py获取更多的模板图片以提高准确率。（准确率越高需要的模板图片也会越高 提升很难）

由于项目限制，做到80%的准确率不错了，如果深度学习的话准确率肯定能上95%。

# 使用说明
所有依赖库都已经打包在requirements.txt里了，直接pip install -r requirements.txt即可。

验证码识别直接调用Recognizer.Recognizer()即可。
![avatar](https://raw.githubusercontent.com/SwaggyMacro/SxbCaptchaRecognizer/main/TempCaptcha/%40MZ18W05T5K1L1AIC5Q.png?token=AJILZ4WHEW5J2VU7I4YHCVLBJKJD2)
 
