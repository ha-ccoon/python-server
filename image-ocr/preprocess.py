import cv2
import easyocr
import os

import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv() 

IMAGE_PATH = './image/'
SAVE_DIR = "./result"

def crop_image(image_path: str):
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

  
  # alpha = 1.5  # 대비 (1.0 - 3.0)
  # beta = 20    # 밝기 (0 - 100)

  # image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


  coordinates_list = [
    (316, 326, 203, 49), # 체중
    (20, 506, 606, 44),  # 골격근량
    (19, 546, 609, 42),  # 체지방량
    (24, 724, 611, 46),  # 체지방률
  ]
  
  cropped_paths = []


  # 영역 잘라내기
  for idx, coords in enumerate(coordinates_list):
    x, y, w, h = coords
    cropped_img = image[y:y + h, x:x + w]

    # 크롭된 이미지 저장 경로
    save_path = os.path.join(SAVE_DIR, f"crop_{idx + 1}.png")
    cv2.imwrite(save_path, cropped_img)
    cropped_paths.append(save_path)

    # 크롭된 이미지 시각화 (옵션)
    plt.subplot(1, len(coordinates_list), idx + 1)
    plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Crop {idx + 1}")
    plt.axis('off')

  plt.tight_layout()
  plt.show()  

  
  return cropped_paths
  

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
    
    cv2.imwrite('./result/processed.png', processed)

    return processed
  
def extract_text(image_paths: list):
  reader = easyocr.Reader(['ko','en'], gpu=False) 
  ocr_results = {}
      
  
  for path in image_paths:
      results = reader.readtext(path)

      # OCR 결과 텍스트만 추출
      extracted_texts = [text for _, text, _ in results]

      corrected_text = extracted_texts[-1].replace(",", ".")

      ocr_results[extracted_texts[0]] = corrected_text

  print('ocr_results', ocr_results)
  return ocr_results


def delete_cropped_images(directory: str = 'result', prefix: str = "crop_") -> None:
  """
  특정 폴더 내에서 지정된 접두사(prefix)로 시작하는 파일들을 삭제

  Args:
  - directory (str): 파일들이 저장된 폴더 경로
  - prefix (str): 삭제할 파일의 접두사 (기본값: "crop_")
  """
  if not os.path.exists(directory):
      print(f"경로가 존재하지 않습니다: {directory}")
      return

  deleted_files = []

  for filename in os.listdir(directory):
      # 접두사(prefix)로 시작하는 파일만 삭제
      if filename.startswith(prefix):
          file_path = os.path.join(directory, filename)
          try:
              os.remove(file_path)
              deleted_files.append(filename)
          except Exception as e:
              print(f"파일 삭제 오류: {file_path} - {e}")

  print(f"삭제된 파일들: {deleted_files}")

if __name__ == "__main__":
    cropped_paths = crop_image(IMAGE_PATH + 'inbody2.jpeg')
    
    extract_text(cropped_paths)
    
    delete_cropped_images()

