import uvicorn
from fastapi import FastAPI, HTTPException
import asyncio
import json
import os
from stt_processor import process_audio_and_get_query_async # 사용자님의 메인 함수

# ----------------------------------------------------
# 1. FastAPI 앱 인스턴스 생성
# ----------------------------------------------------
app = FastAPI(
    title="Minwon STT & Classification API",
    description="실시간 음성 인식을 통해 민원 요청 텍스트를 LLM 분류를 위해 반환합니다."
)

# ----------------------------------------------------
# 2. API 엔드포인트 정의
# ----------------------------------------------------
@app.post("/transcribe")
async def handle_transcribe_request(
    # API 호출 시 녹음 시간을 지정하도록 파라미터를 받습니다. (초 단위)
    duration_seconds: int = 5
):
    """
    마이크 입력을 받아 Amazon Transcribe 스트리밍을 수행하고
    정제된 텍스트와 메타데이터를 JSON으로 반환합니다.
    """
    print(f"\n--- [API 호출됨] 녹음 시간: {duration_seconds}초 ---")
    
    try:
        # stt_processor.py의 비동기 메인 함수를 호출합니다.
        # 이 함수는 마이크 입력을 받고 Transcribe를 거쳐 최종 JSON을 반환합니다.
        final_json_output = await process_audio_and_get_query_async(duration_seconds)
        
        # 오류가 있다면 HTTP 500 에러 반환
        if "error" in final_json_output:
            raise HTTPException(
                status_code=500, 
                detail=final_json_output.get("error")
            )
            
        return final_json_output

    except Exception as e:
        print(f"❌ API 처리 중 예상치 못한 오류 발생: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"서버 오류: {str(e)}"
        )


# ----------------------------------------------------
# 3. 서버 실행 (터미널에서 직접 실행)
# ----------------------------------------------------
if __name__ == "__main__":
    # 서버를 실행하면 http://127.0.0.1:8020/docs 에서 API 문서 확인 가능
    print("🌍 FastAPI 서버 시작 중...")
    uvicorn.run(
        "api_server:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False  # 해커톤 환경에서는 False로 설정하여 안정성을 높입니다.
    )