# category.py
import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
from enum import Enum
from dotenv import load_dotenv

from langchain_aws import ChatBedrockConverse  # 새로 추가
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import uuid

# .env 파일에서 환경 변수 로드
load_dotenv()

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
    session_id: str | None = Field(default=None, description="세션 ID (etc 카테고리일 때 대화 유도를 위해 사용)")

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
            "session_id": session_id,
            "error": str(e)
        }