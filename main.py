# main.py - uvicorn main:app 호환을 위한 루트 레벨 파일
# app 패키지에서 app을 import하여 uvicorn main:app 명령어가 동작하도록 함
from app import app

__all__ = ["app"]
