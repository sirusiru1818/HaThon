# main.py - uvicorn main:app 호환을 위한 파일
# category.py의 app을 import하여 uvicorn main:app 명령어가 동작하도록 함
from .category import app

__all__ = ["app"]

