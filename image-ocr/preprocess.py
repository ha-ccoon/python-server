import cv2
import matplotlib.pyplot as plt

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

    return cropped_image
  

if __name__ == "__main__":
    print("This is being run directly!")
    crop_image(IMAGE_PATH + 'inbody.original.jpeg', (316, 326, 203, 49))