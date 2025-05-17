from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
from ocr import perform_ocr
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

app = FastAPI()

# 환경변수에서 CORS 설정 불러오기
allowed_origins = os.getenv("ALLOWED_ORIGINS")
if allowed_origins:
    allowed_origins = allowed_origins.split(",")
else:
    allowed_origins = ["http://localhost:8000"] 

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "image"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    """
    이미지 파일을 받아 OCR 처리를 수행하고 결과를 반환하는 엔드포인트
    """
    try:
        # 파일 저장 경로
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        # 파일 저장
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # OCR 수행
        ocr_result = perform_ocr(file_path)

        # 결과 반환
        return JSONResponse(content=ocr_result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 파일 삭제 (메모리 관리)
        if os.path.exists(file_path):
            os.remove(file_path)
