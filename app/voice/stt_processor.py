import asyncio
import json
import time
import base64
from typing import Union, List, Dict
import pyaudio # 마이크 입력 라이브러리
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
import os

# aws_config.py에서 Bedrock 클라이언트를 가져옵니다. (사용은 안 하지만 import는 유지)
from aws_config import get_bedrock_client
# embedding_generator.py는 임베딩 로직 제거로 인해 더 이상 사용하지 않습니다.

# ----------------------------------------------------
# 1. Transcribe 결과 처리 핸들러 (Async)
# ----------------------------------------------------

class MinwonTranscriptHandler(TranscriptResultStreamHandler):
    """ Transcribe 스트리밍 결과를 실시간으로 받아 처리하는 비동기 핸들러 """
    def __init__(self, transcript_queue: asyncio.Queue, stream: object):
        super().__init__(stream)
        self.transcript_queue = transcript_queue
        self.final_transcript = ""

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results
        for result in results:
            # 최종 결과만 처리
            if not result.is_partial:
                text = result.alternatives[0].transcript
                
                # 최종 텍스트 누적
                self.final_transcript += " " + text
                await self.transcript_queue.put(text)
                
                print(f"✅ Transcribed Chunk (Final): {text}")

# ----------------------------------------------------
# 2. 마이크 스트림 클래스 (pyaudio 기반)
# ----------------------------------------------------

# 마이크 설정: Transcribe 요구사항 (16000Hz, 1채널, 16비트 PCM)
RATE = 16000
CHUNK = 1024 * 4 # 4KB 청크

class MicrophoneStream:
    """ 마이크에서 오디오 스트림을 생성하는 클래스 (pyaudio 사용) """
    def __init__(self, rate=RATE, chunk=CHUNK):
        self.rate = rate
        self.chunk = chunk
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = None
        
    async def __aenter__(self):
        # 마이크 스트림 열기
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
            stream_callback=None
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # 종료 시 스트림과 인터페이스 정리
        if self._audio_stream:
            self._audio_stream.stop_stream()
            self._audio_stream.close()
        self._audio_interface.terminate()
    
    async def generator(self):
        """ 오디오 청크를 비동기로 생성 """
        while self._audio_stream.is_active():
            # I/O 블록킹을 방지하기 위해 run_in_executor 사용
            data = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._audio_stream.read(self.chunk, exception_on_overflow=False)
            )
            yield data

async def write_chunks(stream, input_stream, duration_seconds: int):
    """ 마이크 스트림에서 오디오 청크를 읽어서 Transcribe로 전송 """
    start_time = asyncio.get_event_loop().time()
    async for chunk in stream.generator():
        await input_stream.send_audio_event(audio_chunk=chunk)
        
        # 지정된 시간이 지나면 종료
        if asyncio.get_event_loop().time() - start_time > duration_seconds:
            print(f"\n⏱️  {duration_seconds}초 녹음 시간 종료.")
            break
    
    await input_stream.end_stream()

# ----------------------------------------------------
# 3. Transcribe API 호출 및 최종 통합
# ----------------------------------------------------

def post_process_transcript(raw_text: str) -> str:
    """ ASR 결과에 대한 최종 텍스트 정제 로직 (Step 3: 도메인 최적화) """
    # 예시: 주민등록 등본처럼 띄어쓰기가 자주 잘못되는 용어를 표준화
    cleaned_text = raw_text.replace("주민등록 등본", "주민등록등본")
    return cleaned_text.strip()


async def stream_transcribe_mic(duration_seconds: int) -> str:
    """ 마이크 입력을 실시간으로 텍스트로 변환하고 최종 텍스트를 반환합니다. """
    client = TranscribeStreamingClient(
        region=os.environ.get("AWS_REGION", "us-east-1") # 🚨 환경 변수에서 리전 자동 로드
    )

    transcript_queue = asyncio.Queue()
    
    # 🚨 마이크 스트림 시작
    async with MicrophoneStream() as stream:
        
        stream_response = await client.start_stream_transcription(
            language_code="ko-KR", 
            media_sample_rate_hz=RATE, 
            media_encoding="pcm",
            # 커스텀 용어를 사용할 경우 CustomVocabularyName 파라미터 추가
        )
        
        handler = MinwonTranscriptHandler(transcript_queue, stream_response.output_stream)

        print(f"\n{'='*80}\n🎙️  실시간 음성 인식 시작. {duration_seconds}초 동안 말씀하세요...\n{'='*80}")
        
        # 전송 작업과 응답 수신 작업을 동시에 실행
        await asyncio.gather(
            write_chunks(stream, stream_response.input_stream, duration_seconds), 
            handler.handle_events()
        )
        
        return post_process_transcript(handler.final_transcript)


async def process_audio_and_get_query_async(duration_seconds: int = 5) -> dict:
    """
    스트리밍 STT를 수행하고 정제된 텍스트를 최종 JSON으로 반환하는 메인 함수입니다.
    """
    
    # 1. 텍스트 변환 (Async Call)
    minwon_text = await stream_transcribe_mic(duration_seconds)
    
    if not minwon_text:
        return {"error": "Transcribe 스트리밍 실패 또는 인식된 음성 없음"}

    print(f"\n\n📝 최종 인식 텍스트: \"{minwon_text}\"")

    # 2. 🚨 임베딩 로직을 제거하고 최종 텍스트 JSON만 반환합니다.
    final_query_data = {
        "user_query_text": minwon_text,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "READY_FOR_LLM_CLASSIFICATION"
    }

    return final_query_data

# ----------------------------------------------------
# 4. 테스트 실행 (Async Main)
# ----------------------------------------------------

if __name__ == '__main__':
    # 🚨 녹음 시간을 짧게 설정하여 테스트합니다. (예: 5초)
    RECORD_DURATION = 5 
    
    try:
        # pyaudio가 설치되어 있는지 재확인
        if 'pyaudio' not in globals() and 'pyaudio' not in locals():
            print("🚨 오류: pyaudio 모듈이 로드되지 않았습니다. pip install pyaudio를 다시 확인하세요.")
            # return

        final_json_output = asyncio.run(process_audio_and_get_query_async(duration_seconds=RECORD_DURATION))
        
        print("\n--- 최종 결과 (다음 팀에게 전달할 쿼리 JSON) ---")
        print(json.dumps(final_json_output, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"\n❌ 최종 실행 오류: {e}")
        print("💡 마이크 장치, AWS 인증 정보, 또는 리전 설정을 확인하세요.")