import cv2
import os
import easyocr

import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv() 

client = OpenAI(api_key=os.getenv('OPEN_AI_API'))

IMAGE_PATH = './image/'

def crop_image(image_path: str, coordinates: tuple):
    """
    이미지에서 특정 영역을 잘라내는 함수

    Args:
    - image_path (str): 이미지 경로
    - coordinates (tuple): (x, y, width, height)

    Returns:
    - cropped_image (numpy.ndarray): 잘라낸 이미지
    """
    # 이미지 읽기
    image = cv2.imread(image_path)
  

    # 좌표 추출
    x, y, w, h = coordinates

    # 영역 잘라내기
    cropped_image = image[y:y + h, x:x + w]
    
    
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()    
    
    cv2.imwrite('./result/weight_value.png', cropped_image)
    
    return cropped_image
  

def preprocess_image(image):
    """
    전처리: 그레이스케일 및 이진화

    Args:
    - image (numpy.ndarray): 이미지 배열

    Returns:
    - processed_image (numpy.ndarray): 전처리된 이미지
    """
    # Grayscale 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive Thresholding
    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    cv2.imwrite('./result/weight_label_processed.png', processed)

    return processed


if __name__ == "__main__":
    cropped = crop_image(IMAGE_PATH + 'inbody2.jpeg', (20, 506, 606, 44))

    processed = preprocess_image(cropped)
    
    reader = easyocr.Reader(['ko','en']) 
    result = reader.readtext('./result/weight_value.png')
    
    for bbox, text, confidence in result:
      print(f"Text: {text}, Confidence: {confidence}")
    
    print('result[0]', result[0])
    print('result[0][1]', result[0][1])
    

