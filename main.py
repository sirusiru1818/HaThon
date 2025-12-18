# main.py - uvicorn main:app 호환을 위한 루트 레벨 파일
# app 패키지에서 app을 import하여 uvicorn main:app 명령어가 동작하도록 함
from app import app
import uvicorn

__all__ = ["app"]

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8020,
        reload=True
    )
