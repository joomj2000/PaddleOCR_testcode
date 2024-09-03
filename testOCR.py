from paddleocr import PaddleOCR, draw_ocr
import cv2
from matplotlib import pyplot as plt

# PaddleOCR 모델 초기화 - 영어
#ocr = PaddleOCR(use_angle_cls=True, lang='en')

# PaddleOCR 모델 초기화 - 한국어
ocr = PaddleOCR(use_angle_cls=True, lang='korean')

# 이미지 파일 경로
img_path = 'test_korean2.png'

# 이미지에서 텍스트 인식
result = ocr.ocr(img_path, cls=True)

# 결과 출력
for line in result:
    print(line)

# 텍스트 박스 좌표, 텍스트, 신뢰도 추출
boxes = [elements[0] for line in result for elements in [line]]
texts = [elements[1][0] for line in result for elements in [line]]
scores = [elements[1][1] for line in result for elements in [line]]

# 결과 시각화
image = cv2.imread(img_path)
image = draw_ocr(image, boxes, texts, scores, font_path='C:/Windows/Fonts/arial.ttf')
plt.imshow(image)
plt.show()
