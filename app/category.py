# category.py
import os
import sys
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from enum import Enum
from dotenv import load_dotenv
from typing import Optional, List
from datetime import datetime
import json

from langchain_aws import ChatBedrockConverse  # 새로 추가
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import uuid

# .env 파일에서 환경 변수 로드
load_dotenv()

# voice 모듈 경로 추가
voice_path = os.path.join(os.path.dirname(__file__), "voice")
if voice_path not in sys.path:
    sys.path.insert(0, voice_path)

# PDF 생성 모듈 import
from .print_pdf import PdfManager, CATEGORY_FOLDER_MAP as PDF_CATEGORY_MAP

# 필수 환경 변수 확인
required_env_vars = ["AWS_MODEL_ID", "AWS_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]

if missing_vars:
    raise ValueError(
        f"필수 환경 변수가 .env 파일에 없습니다: {', '.join(missing_vars)}\n"
        f".env 파일에 다음 변수들을 추가해주세요:\n"
        f"AWS_MODEL_ID=your_model_id\n"
        f"AWS_REGION=your_region\n"
        f"AWS_ACCESS_KEY_ID=your_access_key_id\n"
        f"AWS_SECRET_ACCESS_KEY=your_secret_access_key"
    )

app = FastAPI()

# ============================================
# 실시간 로그 시스템
# ============================================
class LogEntry(BaseModel):
    timestamp: str
    event_type: str  # 'category_request', 'category_result', 'form_start', 'form_chat', 'mode_change', 'error'
    session_id: Optional[str] = None
    form_session_id: Optional[str] = None
    current_mode: str = "category"
    input_text: Optional[str] = None
    category: Optional[str] = None
    response: Optional[str] = None
    details: Optional[dict] = None

# 로그 저장소 (최근 100개만 유지)
log_entries: List[LogEntry] = []
MAX_LOG_ENTRIES = 100

def add_log(event_type: str, **kwargs):
    """로그 추가"""
    entry = LogEntry(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        event_type=event_type,
        **kwargs
    )
    log_entries.append(entry)
    # 최대 개수 초과 시 오래된 것 삭제
    if len(log_entries) > MAX_LOG_ENTRIES:
        log_entries.pop(0)
    print(f"[LOG] {entry.timestamp} | {event_type} | mode={entry.current_mode} | cat={entry.category}")

# Static 파일 서빙 설정
static_path = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# talk_to_fill 모듈 import
from .talk_to_fill import (
    process_form_conversation,
    get_filled_form,
    close_form_session,
    init_form_session,
    get_form_session,
    FormConversationRequest,
    CATEGORY_FOLDER_MAP
)

# 1) LLM 전역 초기화 (앱 시작할 때 한 번만)
llm = ChatBedrockConverse(
    model_id=os.getenv("AWS_MODEL_ID"),
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

# 카테고리 Enum 정의
class Category(str, Enum):
    """행정복지센터 키오스크 질문 카테고리"""
    국민연금 = "국민연금"
    전입신고 = "전입신고"
    토지건축물 = "토지-건축물"
    청년월세 = "청년월세"
    주거급여 = "주거급여"
    ETC = "etc"
    
    @property
    def description(self) -> str:
        """각 카테고리에 대한 설명 반환"""
        descriptions = {
            Category.국민연금: "국민이 노후·장애·사망 등에 대비해 보험료를 내고, 일정 요건을 충족하면 연금이나 급여를 받는 사회보험 제도. 가입·자격변동, 보험료 납부·예외, 노령·장애·유족연금 수급과 각종 증명서 발급 업무를 포함한다. 대표 업무로는 국민연금 신규 가입, 자격 취득·상실 신고 / 보험료 납부, 미납분 정리, 납부예외 신청 / 노령·장애·유족연금 신청 및 변경·정지·재개 / 국민연금 가입·납부이력, 각종 증명서 발급등이 있다.",
            Category.전입신고: "거주지가 바뀌거나 세대 구성이 달라졌을 때, 주민등록 주소와 세대 정보를 실제 거주 상황에 맞게 변경·등록하기 위한 신고 절차. 이사, 세대분리·합가, 해외 체류 후 귀국 등 주소 이동을 행정상으로 반영하는 업무를 다룬다. 대표 업무로는 시·군·구 간 이사, 동일 시 내 동·호 이동 전입신고 / 세대 분리(청년 단독세대)·세대 합가 신고 / 해외 체류 후 귀국 시 전입신고 / 전입신고 기한 및 과태료 관련 안내 등이 있다.",
            Category.토지건축물: "토지와 건축물의 소재지, 면적, 구조, 용도 등 기본 정보를 공적으로 기록·관리하고, 변동 사항을 반영하는 업무 영역. 토지·건축물 대장과 각종 확인·증명 서류를 통해 부동산의 법적·행정적 상태를 확인·정리한다. 대표 업무로는 토지대장·건축물대장 발급 및 열람 / 지목, 면적, 용도, 구조 등의 정보 확인·정정 / 용도변경, 증축·개축 등 건축물 변동 반영 / 부동산 매매·등기용 토지·건축물 관련 서류 발급 등이 있다.",
            Category.청년월세: "일정 요건을 갖춘 청년 가구의 임차료 부담을 줄이기 위해, 월세의 전부 또는 일부를 국가나 지자체가 지원하는 제도. 나이·소득·주거 형태 기준에 따라 대상자를 선정하고, 신청·심사·지급·변경 관리를 수행한다. 대표 업무로는 청년월세 지원 대상·자격 요건 안내 / 청년월세 신청 절차, 기간, 구비서류 안내 / 지급액·지급기간, 중복지원 가능 여부 문의 / 이사·임대차 계약 변경 시 변경 신고 처리 등이 있다.",
            Category.주거급여: "기초생활보장제도 안에서 저소득 가구의 주거비 부담을 덜어주기 위해 임차료나 집 수선비를 지원하는 급여. 소득인정액과 가구원 수, 주거 형태 등을 기준으로 수급 자격을 판단하고, 신청·조사·지급·변경·중지 업무를 처리한다. 대표 업무로는 주거급여 수급 자격·기준 상담 / 주거급여 신규 신청 및 구비서류 안내 / 지급액 산정, 지급 개시·중지·변경 처리 / 이사·임대차 계약 변경에 따른 급여 조정·신고 등이 있다.",
            Category.ETC: "위 5개 카테고리와 전혀 관련이 없는 인사, 날씨, 일반적인 대화 등 (위 5개의 카테고리와 완전히 다른 카테고리로, 어느쪽에도 분류할 수 없을 때만 선택)"
        }
        return descriptions[self]

# 카테고리 분류 모델 정의
class CategoryClassifier(BaseModel):
    """행정복지센터 키오스크 질문 카테고리 분류 모델"""
    category: Category = Field(
        description=(
            "질문의 카테고리를 분류합니다. "
            "각 카테고리: 국민연금(국민연금 가입/납부/수급), 전입신고(전입신고/주소변경/주민등록), "
            "토지-건축물(토지/건축물/부동산 등기), 청년월세(청년월세 지원금/청년주거급여), "
            "주거급여(주거급여 신청/자격/절차), etc(위 카테고리와 전혀 관련 없을 때만). "
            "주의: etc는 다른 카테고리로 분류할 수 없을 때만 선택하세요."
            #굳이 필요없음
        )
    )
    answer: str = Field(
        description="질문에 대한 답변입니다. 간결하되 문장이 자연스럽게 이어지도록 작성하세요. 답변 끝에는 '도와드릴까요?', '필요하신가요?', '궁금하신가요?' 같은 의문 뉘앙스로 끝내서 사용자와의 상호작용을 자연스럽게 유도하세요. 불필요한 장황한 설명은 피하되, 대화가 자연스럽게 흐르도록 적당한 길이로 답변하세요. 300자 이내로 작성하세요."
    )
    reason: str = Field(description="카테고리를 이와같이 판단한 근거를 설명해줘.", max_length=100)

# 프롬프트 정의 - 행정복지센터 키오스크용
system_prompt = (
    "답변은 반드시 한국어로 작성하세요."
    "당신은 행정복지센터 키오스크 상담원입니다. 시민의 질문을 반드시 다음 6개 카테고리 중 하나로 분류하고 간결하게 답변하세요.\n\n"
    "카테고리 분류 규칙:\n"
    "1. 국민연금: 국민연금 가입, 납부, 수급 관련 질문\n"
    "2. 전입신고: 전입신고, 주소변경, 주민등록 관련 질문\n"
    "3. 토지-건축물: 토지, 건축물, 부동산 등기 관련 질문\n"
    "4. 청년월세: 청년월세 지원금, 청년주거급여 관련 질문\n"
    "5. 주거급여: 주거급여 신청, 자격, 절차 관련 질문\n"
    "6. etc: 위 5개 카테고리와 전혀 관련 없는 모든 질문, 인사, 날씨, 일반적인 대화 등\n\n"
    "중요: 모든 질문은 반드시 위 6개 카테고리 중 하나로 분류해야 합니다. 관련 없는 질문이면 반드시 'etc'를 선택하세요.\n\n"
    "답변 원칙:\n"
    "- 간결하되 문장이 자연스럽게 이어지도록 답변하세요. 핵심을 전달하되 대화가 매끄럽게 흐르도록 적당한 길이로 작성하세요.\n"
    "- 자연스러운 대화체로 응답하되, 불필요하게 길지는 않게 하세요 (300자 이내)\n"
    "- 답변 끝에는 '도와드릴까요?', '필요하신가요?', '궁금하신가요?' 같은 의문 뉘앙스로 자연스럽게 끝내세요\n"
    "- 필요한 서류나 절차는 간단히 설명하고, 추가로 도와드릴 점이 있는지 물어보세요\n"
    "- etc 카테고리일 경우:\n"
    "  * 인사나 시답잖은 이야기: 자연스럽게 응답하고 서비스를 안내하며 의문형으로 끝내세요.\n"
    "    예시: '안녕하세요! 어떤 도움이 필요하신가요? 국민연금, 전입신고, 청년월세 등 다양한 서비스를 안내해드릴 수 있어요.'\n"
    "  * 관련 없는 질문: 자연스럽게 설명하고 서비스를 안내하며 의문형으로 끝내세요.\n"
    "    예시: '죄송하지만 그 내용은 저희가 안내하기 어려운 부분이에요. 대신 국민연금, 전입신고, 청년월세 등의 서비스는 도와드릴 수 있는데, 어떤 서비스가 필요하신가요?'"
)

kiosk_prompt = ChatPromptTemplate([
    ("system", system_prompt),
    ("human", [
        {"type": "text", "text": "{question}"},
    ]),
])

# 구조화된 출력을 위한 LLM 체인
kiosk_llm = llm.with_structured_output(CategoryClassifier)
kiosk_chain = kiosk_prompt | kiosk_llm

# 세션별 히스토리 저장소 (etc 카테고리일 때 대화 유도를 위한 세션 관리)
session_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """세션 ID에 해당하는 히스토리를 반환하거나 새로 생성"""
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

def close_session(session_id: str) -> None:
    """세션 ID에 해당하는 세션을 종료하고 히스토리를 삭제"""
    if session_id in session_store:
        del session_store[session_id]

# etc 카테고리일 때 다른 카테고리로 유도하기 위한 프롬프트
etc_guidance_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 행정복지센터 키오스크 상담원입니다. 
사용자가 etc 카테고리(일반 대화, 인사 등)로 분류된 질문을 하고 있습니다.
이전 대화 히스토리를 참고하여, 사용자를 자연스럽게 다음 5개 서비스 중 하나로 유도하세요:

1. 국민연금: 국민연금 가입, 납부, 수급 관련
2. 전입신고: 전입신고, 주소변경, 주민등록 관련
3. 토지-건축물: 토지, 건축물, 부동산 등기 관련
4. 청년월세: 청년월세 지원금, 청년주거급여 관련
5. 주거급여: 주거급여 신청, 자격, 절차 관련

대화 유도 원칙:
- 이전 대화 맥락을 고려하여 자연스럽게 관련 서비스를 제안하세요
- 강압적이지 않게, 친절하고 자연스럽게 유도하세요
- 사용자의 상황(나이, 주거 상황 등)을 파악하여 적절한 서비스를 추천하세요
- 간결하되 문장이 자연스럽게 이어지도록 답변하세요 (300자 이내)
- 답변 끝에는 '도와드릴까요?', '필요하신가요?', '궁금하신가요?' 같은 의문 뉘앙스로 자연스럽게 끝내세요"""),
    MessagesPlaceholder(variable_name="history"),  # 대화 히스토리 주입
    ("human", "{question}")
])

# etc 카테고리 유도용 체인
etc_guidance_chain = etc_guidance_prompt | llm

# RunnableWithMessageHistory로 래핑하여 세션별 히스토리 관리
etc_guidance_chain_with_history = RunnableWithMessageHistory(
    etc_guidance_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

# STT 모듈이 나중에 보내줄 데이터 형식
class UserInquiry(BaseModel):
    text: str  # STT 모듈에서 변환된 사용자 질문 텍스트
    session_id: Optional[str] = Field(default=None, description="세션 ID (etc 카테고리일 때 대화 유도를 위해 사용)")

@app.post("/process")
async def process_inquiry(input_data: UserInquiry):
    """
    행정복지센터 키오스크에서 사용자 질문을 받아 카테고리 분류 및 답변을 제공하는 엔드포인트.
    음성 모듈(STT)에서 변환된 텍스트를 받아 처리합니다.
    category가 etc일 경우 세션을 유지하며 다른 카테고리로 유도합니다.
    """
    user_question = input_data.text
    
    try:
        # 프롬프트에 포맷될 데이터 생성
        inquiry_input = {"question": user_question}

        # 체인 실행 (구조화된 출력 반환)
        response = kiosk_chain.invoke(inquiry_input)

        # 응답이 None이거나 속성이 없는 경우 처리
        if response is None:
            raise ValueError("LLM 응답이 None입니다.")
        
        if not hasattr(response, 'category') or not hasattr(response, 'answer') or not hasattr(response, 'reason'):
            raise ValueError("LLM 응답에 필수 속성이 없습니다.")

        # category가 etc인 경우, 세션을 유지하며 다른 카테고리로 유도
        if response.category.value == "etc":
            # etc일 때는 세션을 유지해야 함
            # 1. 요청에 session_id가 있으면 기존 세션 사용
            # 2. 없으면 새 세션 생성
            incoming_session_id = input_data.session_id
            if incoming_session_id:
                # 기존 세션 사용
                session_id = incoming_session_id
            else:
                # 새 세션 생성
                session_id = str(uuid.uuid4())
            
            # 세션 히스토리를 활용하여 대화 유도
            config = {"configurable": {"session_id": session_id}}
            
            # 세션 히스토리와 함께 대화 유도 체인 실행
            # RunnableWithMessageHistory가 get_session_history(session_id)를 호출하여
            # 같은 session_id에 대한 히스토리를 가져와서 사용
            guidance_response = etc_guidance_chain_with_history.invoke(
                {"question": user_question},
                config=config
            )
            
            # 유도 응답 추출
            if hasattr(guidance_response, 'content'):
                guidance_answer = guidance_response.content[:300]
            else:
                guidance_answer = str(guidance_response)[:300]
            
            # 유도된 대화를 기반으로 다시 카테고리 분류 시도
            # (사용자의 다음 응답이 다른 카테고리로 분류될 수 있도록)
            result = {
                "question": user_question,
                "category": "etc",
                "answer": guidance_answer,
                "message": "질문이 정상적으로 처리되었습니다. 다른 서비스로 유도 중입니다.",
                "reason": response.reason,
                "session_id": session_id,  # 세션 ID 반환하여 클라이언트가 유지할 수 있도록
                "is_guidance": True  # 유도 중임을 표시
            }
            return result

        # etc가 아닌 경우 정상 처리 및 세션 종료
        answer_text = response.answer[:300] if len(response.answer) > 300 else response.answer
        
        # etc가 아닌 경우: 세션이 존재하면 종료 (히스토리 삭제)
        session_id_to_close = input_data.session_id
        if session_id_to_close and session_id_to_close in session_store:
            close_session(session_id_to_close)

        result = {
            "question": user_question,
            "category": response.category.value,  # Enum의 value 반환
            "answer": answer_text,
            "message": "질문이 정상적으로 처리되었습니다.",
            "reason": response.reason,
            "session_id": None,  # 세션 종료되었으므로 None 반환
            "session_closed": True  # 세션이 종료되었음을 명시
        }
        return result
    
    except Exception as e:
        # 오류 발생 시 LLM을 다시 호출하여 자연스러운 응답 생성 (etc 카테고리)
        try:
            # 단순한 LLM 호출로 자연스러운 응답 생성
            fallback_response = llm.invoke(
                f"행정복지센터 키오스크 상담원으로서 다음 질문에 자연스럽게 답변해주세요. "
                f"간결하되 문장이 자연스럽게 이어지도록 적당한 길이로 답변하세요. "
                f"답변 끝에는 '도와드릴까요?', '필요하신가요?', '궁금하신가요?' 같은 의문 뉘앙스로 자연스럽게 끝내세요. "
                f"인사면 자연스럽게 받아주고, 관련 없는 질문이면 행정복지센터 서비스(국민연금, 전입신고, 토지-건축물, 청년월세, 주거급여)를 자연스럽게 안내해주세요.\n\n질문: {user_question}"
            )
            
            if hasattr(fallback_response, 'content'):
                answer = fallback_response.content[:300]
            else:
                answer = str(fallback_response)[:300]
        except:
            # LLM 호출도 실패한 경우 자연스러운 기본 응답
            answer = "안녕하세요! 어떤 도움이 필요하신가요? 국민연금, 전입신고, 청년월세, 주거급여 등의 서비스를 안내해드릴 수 있습니다."
        
        return {
            "question": user_question,
            "category": "etc",
            "answer": answer,
            "message": "질문이 정상적으로 처리되었습니다.",
            "reason": "오류 발생으로 etc 카테고리로 분류되었습니다.",
            "session_id": input_data.session_id,
            "error": str(e)
        }


# ============================================
# 폼 작성 대화 관련 엔드포인트
# ============================================

@app.post("/form/start")
async def start_form_session(category: str, session_id: str = None):
    """
    폼 작성 세션을 시작합니다.
    카테고리에 해당하는 문서들을 로드하고 세션을 초기화합니다.
    """
    if category not in CATEGORY_FOLDER_MAP:
        return {
            "error": f"지원하지 않는 카테고리입니다: {category}",
            "available_categories": list(CATEGORY_FOLDER_MAP.keys())
        }
    
    # 세션 ID가 없으면 새로 생성
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # 세션 초기화
    form_state = init_form_session(session_id, category)
    
    if not form_state["documents"]:
        return {
            "error": f"카테고리 '{category}'에 해당하는 문서가 없습니다.",
            "session_id": session_id
        }
    
    return {
        "message": f"{category} 신청에 필요한 정보를 안내해드리겠습니다.",
        "session_id": session_id,
        "category": category,
        "total_fields": sum(doc["total_count"] for doc in form_state["documents"].values())
    }


@app.post("/form/chat")
async def form_chat(request: FormConversationRequest):
    """
    폼 작성 대화를 처리합니다.
    사용자 입력에서 정보를 추출하고 다음 질문을 생성합니다.
    """
    result = await process_form_conversation(
        session_id=request.session_id,
        user_input=request.user_input,
        category=request.category
    )
    
    return result


@app.get("/form/status/{session_id}")
async def get_form_status(session_id: str):
    """
    현재 폼 작성 상태를 조회합니다.
    """
    session = get_form_session(session_id)
    
    if not session:
        return {"error": "세션을 찾을 수 없습니다."}
    
    return {
        "session_id": session_id,
        "category": session["category"],
        "current_document": session["current_document"],
        "completed": session["completed"],
        "documents": {
            doc_name: {
                "filled_count": doc["filled_count"],
                "total_count": doc["total_count"],
                "progress": f"{doc['filled_count']}/{doc['total_count']}"
            }
            for doc_name, doc in session["documents"].items()
        }
    }


@app.get("/form/result/{session_id}")
async def get_form_result(session_id: str):
    """
    완성된 폼 데이터를 조회합니다.
    """
    result = get_filled_form(session_id)
    
    if not result:
        return {"error": "세션을 찾을 수 없습니다."}
    
    return result


@app.delete("/form/session/{session_id}")
async def delete_form_session(session_id: str):
    """
    폼 세션을 종료하고 삭제합니다.
    """
    result = close_form_session(session_id)
    
    if not result:
        return {"error": "세션을 찾을 수 없습니다."}
    
    return {
        "message": "세션이 종료되었습니다.",
        "final_data": get_filled_form(session_id)
    }


@app.get("/categories")
async def get_categories():
    """
    사용 가능한 카테고리 목록을 반환합니다.
    """
    return {
        "categories": list(CATEGORY_FOLDER_MAP.keys())
    }


# ============================================
# 음성 관련 엔드포인트 (Voice Integration)
# ============================================

# STT 모듈 import 시도 (pyaudio가 없을 수 있음)
try:
    from stt_processor import process_audio_and_get_query_async
    VOICE_AVAILABLE = True
    print("✅ Voice 모듈 로드 성공")
except ImportError as e:
    VOICE_AVAILABLE = False
    print(f"⚠️ Voice 모듈 로드 실패 (pyaudio 필요): {e}")


class VoiceTranscribeRequest(BaseModel):
    """음성 인식 요청 모델"""
    duration_seconds: int = Field(default=5, description="녹음 시간 (초)")
    session_id: Optional[str] = Field(default=None, description="세션 ID")


@app.post("/voice/transcribe")
async def voice_transcribe(duration_seconds: int = 5, session_id: str = None):
    """
    마이크 입력을 받아 Amazon Transcribe 스트리밍을 수행하고
    텍스트로 변환한 후 카테고리 분류까지 수행합니다.
    
    ⚠️ 이 엔드포인트는 서버에 마이크가 연결되어 있어야 합니다.
    브라우저 기반 음성 입력은 /voice/process 엔드포인트를 사용하세요.
    """
    if not VOICE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Voice 모듈을 사용할 수 없습니다. pyaudio와 amazon-transcribe-streaming-sdk를 설치하세요."
        )
    
    try:
        # STT 수행
        stt_result = await process_audio_and_get_query_async(duration_seconds)
        
        if "error" in stt_result:
            raise HTTPException(status_code=500, detail=stt_result["error"])
        
        transcribed_text = stt_result.get("user_query_text", "")
        
        if not transcribed_text:
            return {
                "transcribed_text": "",
                "message": "인식된 음성이 없습니다.",
                "category": None,
                "answer": None
            }
        
        # 카테고리 분류 수행
        inquiry_data = UserInquiry(text=transcribed_text, session_id=session_id)
        classification_result = await process_inquiry(inquiry_data)
        
        return {
            "transcribed_text": transcribed_text,
            "timestamp": stt_result.get("timestamp"),
            **classification_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"음성 처리 오류: {str(e)}")


class VoiceTextRequest(BaseModel):
    """브라우저에서 전송된 음성 텍스트 처리 요청"""
    text: str = Field(description="Web Speech API로 인식된 텍스트")
    session_id: Optional[str] = Field(default=None, description="세션 ID")
    mode: str = Field(default="category", description="처리 모드: 'category' 또는 'form'")
    form_session_id: Optional[str] = Field(default=None, description="폼 세션 ID (mode='form'일 때)")
    category: Optional[str] = Field(default=None, description="카테고리 (mode='form'일 때)")


@app.post("/voice/process")
async def voice_process(request: VoiceTextRequest):
    """
    브라우저의 Web Speech API로 인식된 텍스트를 처리합니다.
    
    흐름:
    1. mode='category': 카테고리 분류 수행
       - etc가 아닌 카테고리로 확정되면 → 자동으로 폼 세션 시작 → mode='form'으로 전환
       - etc면 → 계속 카테고리 모드 유지
    2. mode='form': 폼 작성 대화 수행
    """
    if not request.text or not request.text.strip():
        add_log("error", input_text="(empty)", details={"error": "텍스트가 비어있습니다."})
        return {
            "error": "텍스트가 비어있습니다.",
            "transcribed_text": ""
        }
    
    transcribed_text = request.text.strip()
    
    # 요청 로그
    add_log(
        "request_received",
        current_mode=request.mode,
        session_id=request.session_id,
        form_session_id=request.form_session_id,
        input_text=transcribed_text[:50] + "..." if len(transcribed_text) > 50 else transcribed_text,
        category=request.category,
        details={"full_text": transcribed_text}
    )
    
    if request.mode == "form":
        # 폼 작성 모드
        add_log(
            "form_mode_processing",
            current_mode="form",
            form_session_id=request.form_session_id,
            category=request.category,
            input_text=transcribed_text[:50]
        )
        
        if not request.form_session_id:
            add_log("error", current_mode="form", details={"error": "폼 세션 ID가 없음"})
            return {
                "error": "폼 세션 ID가 필요합니다.",
                "transcribed_text": transcribed_text
            }
        
        print(f"[FORM DEBUG] 폼 세션 요청 시작 - session_id: {request.form_session_id}")
        print(f"[FORM DEBUG] 사용자 입력: {transcribed_text}")
        print(f"[FORM DEBUG] 카테고리: {request.category}")
        
        form_result = await process_form_conversation(
            session_id=request.form_session_id,
            user_input=transcribed_text,
            category=request.category
        )
        
        print(f"[FORM DEBUG] 폼 응답 받음:")
        print(f"[FORM DEBUG]   - response: {form_result.get('response', 'None')[:150]}")
        print(f"[FORM DEBUG]   - unfilled_count: {form_result.get('unfilled_count')}")
        print(f"[FORM DEBUG]   - completed: {form_result.get('completed')}")
        print(f"[FORM DEBUG]   - extracted_fields: {form_result.get('extracted_fields', {})}")
        
        add_log(
            "form_response",
            current_mode="form",
            form_session_id=request.form_session_id,
            category=request.category,
            response=form_result.get("response", "")[:100] if form_result.get("response") else None,
            details={
                "unfilled_count": form_result.get("unfilled_count"),
                "completed": form_result.get("completed"),
                "extracted_fields": form_result.get("extracted_fields", {})
            }
        )
        
        # 폼 작성이 완료되면 자동으로 PDF 생성
        pdf_files = None
        if form_result.get("completed"):
            print(f"[PDF AUTO] 폼 작성 완료 감지 - PDF 자동 생성 시작")
            add_log(
                "pdf_generation_started",
                current_mode="form",
                form_session_id=request.form_session_id,
                category=request.category
            )
            
            try:
                from .talk_to_fill import get_filled_form
                
                # 완성된 폼 데이터 가져오기
                filled_data = get_filled_form(request.form_session_id)
                
                if filled_data:
                    category_folder = PDF_CATEGORY_MAP.get(request.category)
                    
                    if category_folder:
                        # PDF Manager 초기화
                        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        pdf_manager = PdfManager(project_root)
                        
                        # 출력 디렉토리 설정
                        output_dir = os.path.join(project_root, "output", request.form_session_id)
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # 모든 문서에 대해 PDF 생성
                        pdf_files = []
                        for doc_name, doc_data in filled_data["documents"].items():
                            try:
                                output_path = os.path.join(output_dir, f"{doc_name}.pdf")
                                result_path = pdf_manager.process_request(
                                    category_folder=category_folder,
                                    document_name=doc_name,
                                    user_data=doc_data,
                                    output_filename=output_path,
                                    debug=False
                                )
                                pdf_files.append({
                                    "document_name": doc_name,
                                    "file_path": result_path
                                })
                                print(f"[PDF AUTO] ✅ '{doc_name}' PDF 생성 완료: {result_path}")
                            except Exception as e:
                                print(f"[PDF AUTO] ❌ '{doc_name}' PDF 생성 실패: {e}")
                        
                        add_log(
                            "pdf_generation_completed",
                            current_mode="form",
                            form_session_id=request.form_session_id,
                            category=request.category,
                            details={"pdf_files": pdf_files}
                        )
                    else:
                        print(f"[PDF AUTO] ⚠️ 카테고리 '{request.category}'에 대한 폴더 매핑이 없습니다.")
                else:
                    print(f"[PDF AUTO] ⚠️ 폼 데이터를 가져올 수 없습니다.")
                    
            except Exception as e:
                print(f"[PDF AUTO] ❌ PDF 자동 생성 오류: {e}")
                import traceback
                traceback.print_exc()
                add_log(
                    "pdf_generation_error",
                    current_mode="form",
                    form_session_id=request.form_session_id,
                    category=request.category,
                    details={"error": str(e)}
                )
        
        response_data = {
            "transcribed_text": transcribed_text,
            "mode": "form",
            "form_session_id": request.form_session_id,
            "category": request.category,
            **form_result
        }
        
        # PDF 파일 정보 추가
        if pdf_files:
            response_data["pdf_files"] = pdf_files
        
        return response_data
    else:
        # 카테고리 분류 모드 (기본)
        add_log(
            "category_mode_processing",
            current_mode="category",
            session_id=request.session_id,
            input_text=transcribed_text[:50]
        )
        
        inquiry_data = UserInquiry(text=transcribed_text, session_id=request.session_id)
        classification_result = await process_inquiry(inquiry_data)
        
        category = classification_result.get("category")
        
        add_log(
            "category_classified",
            current_mode="category",
            session_id=request.session_id,
            category=category,
            response=classification_result.get("answer", "")[:100] if classification_result.get("answer") else None,
            details={"reason": classification_result.get("reason")}
        )
        
        # etc가 아닌 카테고리로 확정되면 → 자동으로 폼 세션 시작
        if category and category != "etc" and category in CATEGORY_FOLDER_MAP:
            # 새 폼 세션 생성
            form_session_id = str(uuid.uuid4())
            form_state = init_form_session(form_session_id, category)
            
            add_log(
                "mode_change_to_form",
                current_mode="category→form",
                form_session_id=form_session_id,
                category=category,
                details={
                    "documents": list(form_state["documents"].keys()) if form_state.get("documents") else [],
                    "total_fields": sum(doc["total_count"] for doc in form_state["documents"].values()) if form_state.get("documents") else 0
                }
            )
            
            # 폼 세션이 정상적으로 생성되었는지 확인
            if form_state and form_state.get("documents"):
                # 첫 번째 폼 대화 시작 - 초기 질문 생성
                initial_form_result = await process_form_conversation(
                    session_id=form_session_id,
                    user_input="안녕하세요. 신청에 필요한 정보를 알려주세요.",
                    category=category
                )
                
                add_log(
                    "form_session_started",
                    current_mode="form",
                    form_session_id=form_session_id,
                    category=category,
                    response=initial_form_result.get("response", "")[:100] if initial_form_result.get("response") else None
                )
                
                return {
                    "transcribed_text": transcribed_text,
                    "mode": "form",  # 폼 모드로 전환!
                    "mode_changed": True,  # 모드가 변경되었음을 알림
                    "category": category,
                    "form_session_id": form_session_id,
                    "category_answer": classification_result.get("answer"),  # 카테고리 분류 답변
                    "documents": list(form_state["documents"].keys()),
                    "total_fields": sum(doc["total_count"] for doc in form_state["documents"].values()),
                    **initial_form_result  # 폼 작성 첫 질문 포함
                }
            else:
                # 폼 세션 생성 실패 (문서가 없음) - 카테고리 모드 유지
                add_log(
                    "form_session_failed",
                    current_mode="category",
                    category=category,
                    details={"error": "문서가 없음"}
                )
                return {
                    "transcribed_text": transcribed_text,
                    "mode": "category",
                    "category": category,
                    "answer": classification_result.get("answer"),
                    "message": f"{category} 카테고리로 분류되었지만, 해당 서류가 없습니다.",
                    "session_id": request.session_id
                }
        
        # etc 카테고리거나 폼이 없는 경우 - 카테고리 모드 유지
        add_log(
            "staying_in_category_mode",
            current_mode="category",
            session_id=request.session_id,
            category=category,
            details={"reason": "etc 카테고리 또는 폼 없음"}
        )
        return {
            "transcribed_text": transcribed_text,
            "mode": "category",
            **classification_result
        }


@app.get("/voice/status")
async def voice_status():
    """
    음성 모듈 상태를 확인합니다.
    """
    return {
        "voice_available": VOICE_AVAILABLE,
        "web_speech_api": True,  # 브라우저 Web Speech API는 항상 사용 가능
        "server_stt": VOICE_AVAILABLE,
        "message": "Web Speech API를 통한 브라우저 음성 인식을 사용할 수 있습니다." if not VOICE_AVAILABLE else "서버 STT와 브라우저 음성 인식 모두 사용 가능합니다."
    }


# ============================================
# 실시간 모니터링 엔드포인트
# ============================================

@app.get("/api/logs")
async def get_logs(limit: int = 50):
    """최근 로그 조회"""
    return {
        "logs": [entry.dict() for entry in log_entries[-limit:]],
        "total": len(log_entries)
    }

@app.get("/api/logs/clear")
async def clear_logs():
    """로그 초기화"""
    log_entries.clear()
    return {"message": "로그가 초기화되었습니다."}

@app.get("/online", response_class=HTMLResponse)
async def get_monitor_page():
    """실시간 LLM 흐름 모니터링 페이지"""
    html_path = os.path.join(os.path.dirname(__file__), "static", "online.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return """<html>
        <head><title>모니터링 페이지</title></head>
        <body>
            <h1>Static 파일이 없습니다.</h1>
            <p>app/static/online.html 파일을 생성해주세요.</p>
        </body>
    </html>"""


# ============================================
# 프론트엔드 페이지
# ============================================

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """메인 테스트 페이지"""
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return """
    <html>
        <head><title>HaThon Kiosk</title></head>
        <body>
            <h1>Static 파일이 없습니다.</h1>
            <p>app/static/index.html 파일을 생성해주세요.</p>
        </body>
    </html>
    """


# ============================================
# PDF 생성 엔드포인트
# ============================================

class PdfGenerateRequest(BaseModel):
    """PDF 생성 요청 모델"""
    session_id: str = Field(description="폼 세션 ID")
    document_name: Optional[str] = Field(default=None, description="특정 문서만 생성 (None이면 전체)")
    debug: bool = Field(default=False, description="디버그 모드 (빨간 박스 표시)")


@app.post("/pdf/generate")
async def generate_pdf(request: PdfGenerateRequest):
    """
    완성된 폼 데이터로 PDF를 생성합니다.
    
    Args:
        session_id: 폼 세션 ID
        document_name: 특정 문서만 생성 (None이면 전체 문서 생성)
        debug: 디버그 모드
        
    Returns:
        생성된 PDF 파일 경로 목록
    """
    from .talk_to_fill import get_form_session, get_filled_form
    
    # 세션 확인
    session = get_form_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    
    # 완성된 폼 데이터 가져오기
    filled_data = get_filled_form(request.session_id)
    if not filled_data:
        raise HTTPException(status_code=404, detail="폼 데이터를 찾을 수 없습니다.")
    
    category = filled_data["category"]
    category_folder = PDF_CATEGORY_MAP.get(category)
    
    if not category_folder:
        raise HTTPException(status_code=400, detail=f"'{category}' 카테고리에 대한 폴더 매핑이 없습니다.")
    
    # PDF Manager 초기화
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_manager = PdfManager(project_root)
    
    # 출력 디렉토리 설정
    output_dir = os.path.join(project_root, "output", request.session_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # PDF 생성
    generated_files = []
    errors = []
    
    documents_to_process = {}
    if request.document_name:
        # 특정 문서만
        if request.document_name in filled_data["documents"]:
            documents_to_process[request.document_name] = filled_data["documents"][request.document_name]
    else:
        # 전체 문서
        documents_to_process = filled_data["documents"]
    
    for doc_name, doc_data in documents_to_process.items():
        try:
            output_path = os.path.join(output_dir, f"{doc_name}.pdf")
            
            result_path = pdf_manager.process_request(
                category_folder=category_folder,
                document_name=doc_name,
                user_data=doc_data,
                output_filename=output_path,
                debug=request.debug
            )
            
            generated_files.append({
                "document_name": doc_name,
                "file_path": result_path,
                "success": True
            })
            
        except Exception as e:
            errors.append({
                "document_name": doc_name,
                "error": str(e)
            })
    
    return {
        "session_id": request.session_id,
        "category": category,
        "generated_files": generated_files,
        "errors": errors if errors else None,
        "output_directory": output_dir
    }


@app.get("/pdf/download/{session_id}/{document_name}")
async def download_pdf(session_id: str, document_name: str):
    """
    생성된 PDF 파일을 다운로드합니다.
    
    Args:
        session_id: 폼 세션 ID
        document_name: 문서명 (예: "위임장", "대리수령")
        
    Returns:
        PDF 파일
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_path = os.path.join(project_root, "output", session_id, f"{document_name}.pdf")
    
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"PDF 파일을 찾을 수 없습니다: {document_name}.pdf")
    
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=f"{document_name}.pdf"
    )


@app.get("/pdf/list/{session_id}")
async def list_pdfs(session_id: str):
    """
    세션의 생성된 PDF 파일 목록을 반환합니다.
    
    Args:
        session_id: 폼 세션 ID
        
    Returns:
        PDF 파일 목록
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "output", session_id)
    
    if not os.path.exists(output_dir):
        return {
            "session_id": session_id,
            "pdf_files": [],
            "message": "생성된 PDF가 없습니다."
        }
    
    pdf_files = []
    for filename in os.listdir(output_dir):
        if filename.endswith('.pdf'):
            doc_name = os.path.splitext(filename)[0]
            pdf_files.append({
                "document_name": doc_name,
                "filename": filename,
                "download_url": f"/pdf/download/{session_id}/{doc_name}"
            })
    
    return {
        "session_id": session_id,
        "pdf_files": pdf_files,
        "count": len(pdf_files)
    }