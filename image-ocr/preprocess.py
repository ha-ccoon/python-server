import cv2
import os
from PIL import Image
import base64
import io

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
    plt.title(f"Cropped Image - {cropped_image}")
    plt.axis('off')
    plt.show()
    
    pil_img = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))


    return pil_img


def encode_image(image):
    """PIL 이미지 객체를 base64로 인코딩"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')



if __name__ == "__main__":
    print("This is being run directly!")
    cropped = crop_image(IMAGE_PATH + 'inbody.original.jpeg', (316, 326, 203, 49))
    
    encode_image = encode_image(cropped)
    
    response = client.responses.create(
      model="gpt-4.1-mini",
      input=[{
          "role": "user",
          "content": [
              {"type": "input_text", "text": "체중 값을 가져와줘"},
              {
                  "type": "input_image",
                  "image_url": f"data:image/jpeg;base64,{encode_image}",
              },
          ],
      }],
    )
    
    print(response.output_text)
