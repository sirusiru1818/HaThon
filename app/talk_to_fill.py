# talk_to_fill.py
# 카테고리가 결정된 후, 해당 카테고리 폴더의 문서들을 LLM이 학습하고
# 사용자와 대화하며 필요한 정보(폼 필드)를 채워나가는 모듈

import os
import json
import re
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum

from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# LLM 초기화
llm = ChatBedrockConverse(
    model_id=os.getenv("AWS_MODEL_ID"),
    region_name=os.getenv("AWS_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

# 카테고리와 폴더 매핑
CATEGORY_FOLDER_MAP = {
    "국민연금": "1_Welfare",
    "전입신고": "2_Report", 
    "토지-건축물": "3_Land",
    "청년월세": "4_Monthly",
    "주거급여": "5_Salary"
}

# docs 폴더 기본 경로
DOCS_BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")


def parse_json_with_comments(content: str) -> Dict[str, Any]:
    """
    주석이 포함된 JSON 형식의 텍스트를 파싱합니다.
    // 스타일의 주석을 제거하고 JSON으로 변환합니다.
    """
    # // 스타일 주석 제거 (문자열 내부의 //는 유지)
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # 문자열 밖의 // 주석만 제거
        in_string = False
        result = []
        i = 0
        while i < len(line):
            char = line[i]
            if char == '"' and (i == 0 or line[i-1] != '\\'):
                in_string = not in_string
                result.append(char)
            elif not in_string and i + 1 < len(line) and line[i:i+2] == '//':
                # 주석 시작, 나머지 줄 무시
                break
            else:
                result.append(char)
            i += 1
        
        # 주석 제거 후 trailing 공백/탭 제거
        cleaned_line = ''.join(result).rstrip()
        
        # 주석 전용 줄이거나 빈 줄이 아니면 추가
        if cleaned_line and not cleaned_line.strip().startswith('//'):
            cleaned_lines.append(cleaned_line)
    
    cleaned_content = '\n'.join(cleaned_lines)
    
    # 마지막 콤마 제거 (JSON 표준에 맞게)
    cleaned_content = re.sub(r',(\s*[}\]])', r'\1', cleaned_content)
    
    try:
        parsed = json.loads(cleaned_content)
        print(f"[DEBUG] ✅ JSON 파싱 성공: {len(parsed)} 개 필드")
        return parsed
    except json.JSONDecodeError as e:
        # 파싱 실패 시 빈 딕셔너리 반환
        print(f"[DEBUG] ❌ JSON 파싱 오류: {e}")
        print(f"[DEBUG] 파싱 시도한 내용 (처음 500자):\n{cleaned_content[:500]}")
        return {}


def load_category_documents(category: str) -> Dict[str, Dict[str, Any]]:
    """
    카테고리에 해당하는 폴더의 모든 문서를 로드합니다.
    
    Returns:
        Dict[str, Dict]: {파일명: {필드명: 필드정보}} 형태의 딕셔너리
    """
    folder_name = CATEGORY_FOLDER_MAP.get(category)
    if not folder_name:
        return {}
    
    folder_path = os.path.join(DOCS_BASE_PATH, folder_name)
    
    if not os.path.exists(folder_path):
        return {}
    
    documents = {}
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt') or filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    parsed = parse_json_with_comments(content)
                    if parsed:
                        # 파일명에서 확장자 제거
                        doc_name = os.path.splitext(filename)[0]
                        documents[doc_name] = parsed
            except Exception as e:
                print(f"파일 로드 오류 ({filename}): {e}")
    
    return documents


def extract_field_descriptions(content: str) -> Dict[str, str]:
    """
    원본 텍스트에서 각 필드의 주석(설명)을 추출합니다.
    """
    descriptions = {}
    lines = content.split('\n')
    
    for line in lines:
        # "field.name": "value", //설명 패턴 매칭
        match = re.search(r'"([^"]+)":\s*"[^"]*",?\s*//(.+)$', line)
        if match:
            field_name = match.group(1)
            description = match.group(2).strip()
            descriptions[field_name] = description
    
    return descriptions


def load_category_documents_with_descriptions(category: str) -> Dict[str, Dict[str, Any]]:
    """
    카테고리에 해당하는 폴더의 모든 문서를 로드하고, 필드 설명도 함께 반환합니다.
    
    Returns:
        Dict[str, Dict]: {
            파일명: {
                "fields": {필드명: 기본값},
                "descriptions": {필드명: 설명}
            }
        }
    """
    folder_name = CATEGORY_FOLDER_MAP.get(category)
    print(f"[DEBUG] 카테고리: {category} → 폴더: {folder_name}")
    
    if not folder_name:
        print(f"[DEBUG] 카테고리 '{category}'에 매핑된 폴더가 없습니다.")
        return {}
    
    folder_path = os.path.join(DOCS_BASE_PATH, folder_name)
    print(f"[DEBUG] 폴더 경로: {folder_path}")
    print(f"[DEBUG] 폴더 존재 여부: {os.path.exists(folder_path)}")
    
    if not os.path.exists(folder_path):
        print(f"[DEBUG] 폴더가 존재하지 않습니다: {folder_path}")
        return {}
    
    documents = {}
    
    files = os.listdir(folder_path)
    print(f"[DEBUG] 폴더 내 파일 목록: {files}")
    
    for filename in files:
        if filename.endswith('.txt') or filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            print(f"[DEBUG] 파일 처리 중: {filename}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f"[DEBUG] 파일 내용 길이: {len(content)}")
                    parsed = parse_json_with_comments(content)
                    print(f"[DEBUG] 파싱 결과: {len(parsed)} 필드")
                    descriptions = extract_field_descriptions(content)
                    
                    if parsed:
                        doc_name = os.path.splitext(filename)[0]
                        documents[doc_name] = {
                            "fields": parsed,
                            "descriptions": descriptions
                        }
                        print(f"[DEBUG] ✅ '{doc_name}' 문서 로드 성공")
                    else:
                        print(f"[DEBUG] ❌ '{filename}' 파싱 결과가 비어있습니다.")
            except Exception as e:
                print(f"[DEBUG] ❌ 파일 로드 오류 ({filename}): {e}")
                import traceback
                traceback.print_exc()
    
    print(f"[DEBUG] 최종 로드된 문서 수: {len(documents)}")
    print(f"[DEBUG] 문서 이름들: {list(documents.keys())}")
    
    return documents


# 세션별 폼 작성 상태 저장소
form_session_store: Dict[str, Dict[str, Any]] = {}

# 세션별 대화 히스토리 저장소
chat_history_store: Dict[str, InMemoryChatMessageHistory] = {}


def get_chat_history(session_id: str) -> BaseChatMessageHistory:
    """세션 ID에 해당하는 대화 히스토리를 반환하거나 새로 생성"""
    if session_id not in chat_history_store:
        chat_history_store[session_id] = InMemoryChatMessageHistory()
    return chat_history_store[session_id]


def init_form_session(session_id: str, category: str) -> Dict[str, Any]:
    """
    폼 작성 세션을 초기화합니다.
    카테고리에 해당하는 문서들을 로드하고 빈 폼 상태를 생성합니다.
    """
    documents = load_category_documents_with_descriptions(category)
    
    # 각 문서별로 빈 폼 상태 초기화
    form_state = {
        "category": category,
        "documents": {},
        "current_document": None,
        "completed": False
    }
    
    for doc_name, doc_data in documents.items():
        form_state["documents"][doc_name] = {
            "fields": {field: "" for field in doc_data["fields"].keys()},
            "descriptions": doc_data["descriptions"],
            "template": doc_data["fields"],  # 원본 템플릿 저장
            "filled_count": 0,
            "total_count": len(doc_data["fields"])
        }
    
    # 첫 번째 문서를 현재 문서로 설정
    if documents:
        form_state["current_document"] = list(documents.keys())[0]
    
    form_session_store[session_id] = form_state
    return form_state


def get_form_session(session_id: str) -> Optional[Dict[str, Any]]:
    """세션의 폼 상태를 반환합니다."""
    return form_session_store.get(session_id)


def update_form_field(session_id: str, document_name: str, field_name: str, value: str) -> bool:
    """
    특정 필드의 값을 업데이트합니다.
    """
    session = form_session_store.get(session_id)
    if not session:
        return False
    
    doc = session["documents"].get(document_name)
    if not doc:
        return False
    
    if field_name in doc["fields"]:
        old_value = doc["fields"][field_name]
        doc["fields"][field_name] = value
        
        # 채워진 필드 수 업데이트
        if old_value == "" and value != "":
            doc["filled_count"] += 1
        elif old_value != "" and value == "":
            doc["filled_count"] -= 1
        
        return True
    
    return False


def get_unfilled_fields(session_id: str, document_name: str = None) -> List[Dict[str, str]]:
    """
    아직 채워지지 않은 필드 목록을 반환합니다.
    """
    session = form_session_store.get(session_id)
    if not session:
        return []
    
    unfilled = []
    
    docs_to_check = [document_name] if document_name else session["documents"].keys()
    
    for doc_name in docs_to_check:
        doc = session["documents"].get(doc_name)
        if not doc:
            continue
            
        for field_name, value in doc["fields"].items():
            if value == "":
                unfilled.append({
                    "document": doc_name,
                    "field": field_name,
                    "description": doc["descriptions"].get(field_name, field_name)
                })
    
    return unfilled


def close_form_session(session_id: str) -> Optional[Dict[str, Any]]:
    """
    폼 세션을 종료하고 최종 결과를 반환합니다.
    """
    session = form_session_store.pop(session_id, None)
    chat_history_store.pop(session_id, None)
    return session


# 폼 작성 유도 프롬프트
form_filling_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 행정복지센터 키오스크 상담원입니다.
사용자가 {category} 관련 서류를 작성하도록 도와주고 있습니다.

현재 작성 중인 서류: {current_document}

아직 채워지지 않은 필드들:
{unfilled_fields}

⚠️ 중요한 규칙:
1. 위에 "아직 채워지지 않은 필드들" 목록이 비어있지 않으면, 절대로 "작성 완료", "완료되었습니다", "끝났습니다" 같은 말을 하지 마세요!
2. 미작성 필드가 남아있는 한, 계속해서 정보를 수집해야 합니다.
3. 사용자가 "필요없어", "모르겠어", "아니" 같은 말을 해도, 친절하게 설명하고 다시 질문하세요.
4. 모든 필드가 실제로 채워질 때까지 작성은 끝나지 않습니다.

대화 원칙:
1. 한 번에 1-2개의 정보만 자연스럽게 물어보세요.
2. 사용자가 제공한 정보를 확인하고, 다음 필요한 정보를 물어보세요.
3. 이미 채워진 정보는 다시 묻지 마세요.
4. 친절하고 자연스러운 대화체를 사용하세요.
5. 필드명을 직접 언급하지 말고, 자연스러운 질문으로 정보를 수집하세요.
6. 답변은 300자 이내로 간결하게 작성하세요.
7. 사용자가 정보 제공을 거부하면, 왜 필요한지 간단히 설명하고 다시 요청하세요.

예시:
- "성함이 어떻게 되시나요?" (이름 수집)
- "생년월일을 알려주시겠어요?" (생년월일 수집)
- "현재 거주하시는 주소가 어떻게 되시나요?" (주소 수집)
- 사용자가 거부하면: "서류 작성을 위해 꼭 필요한 정보입니다. 주소를 알려주시겠어요?"
"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{user_input}")
])

# 정보 추출 프롬프트
extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", """사용자의 응답에서 다음 필드들에 해당하는 정보를 추출하세요.
추출할 필드 목록:
{target_fields}

추출 규칙:
1. 사용자가 명시적으로 언급한 정보만 추출하세요.
2. 추측하지 마세요.
3. 날짜는 "YYYY-MM-DD" 형식으로 변환하세요.
4. 전화번호는 "010-XXXX-XXXX" 또는 "02-XXX-XXXX" 형식으로 변환하세요.
5. 체크박스 필드는 해당되면 "V", 아니면 빈 문자열로 표시하세요.

JSON 형식으로만 응답하세요. 추출된 필드만 포함하세요.
예시: {{"field.name": "홍길동", "field.birthdate": "1990-01-15"}}
추출할 정보가 없으면 빈 객체를 반환하세요: {{}}
"""),
    ("human", "사용자 응답: {user_response}")
])


# 정보 추출을 위한 Pydantic 모델
class ExtractedInfo(BaseModel):
    """사용자 응답에서 추출된 정보"""
    extracted_fields: Dict[str, str] = Field(
        default_factory=dict,
        description="추출된 필드와 값의 딕셔너리"
    )


def create_form_chain(session_id: str):
    """폼 작성용 대화 체인을 생성합니다."""
    chain = form_filling_prompt | llm
    
    return RunnableWithMessageHistory(
        chain,
        get_chat_history,
        input_messages_key="user_input",
        history_messages_key="history"
    )


async def process_form_conversation(
    session_id: str,
    user_input: str,
    category: str = None
) -> Dict[str, Any]:
    """
    폼 작성 대화를 처리합니다.
    
    Args:
        session_id: 세션 ID
        user_input: 사용자 입력
        category: 카테고리 (새 세션 시작 시 필요)
    
    Returns:
        Dict containing:
        - response: LLM 응답
        - extracted_fields: 추출된 필드들
        - form_state: 현재 폼 상태
        - completed: 폼 작성 완료 여부
    """
    # 세션 확인 또는 생성
    session = get_form_session(session_id)
    
    if not session and category:
        session = init_form_session(session_id, category)
    elif not session:
        return {
            "error": "세션이 없습니다. 카테고리를 지정해주세요.",
            "response": None,
            "extracted_fields": {},
            "form_state": None,
            "completed": False
        }
    
    # 현재 문서와 채워지지 않은 필드 가져오기
    current_doc = session["current_document"]
    unfilled = get_unfilled_fields(session_id, current_doc)
    
    print(f"[TALK_TO_FILL] 현재 문서: {current_doc}")
    print(f"[TALK_TO_FILL] 미작성 필드 수: {len(unfilled)}")
    if unfilled:
        print(f"[TALK_TO_FILL] 처음 5개 미작성 필드: {[f['field'] for f in unfilled[:5]]}")
    
    # 사용자 응답에서 정보 추출
    if unfilled:
        target_fields_str = "\n".join([
            f"- {f['field']}: {f['description']}" 
            for f in unfilled[:5]  # 최대 5개 필드만 대상
        ])
        
        extraction_chain = extraction_prompt | llm
        
        try:
            print(f"[TALK_TO_FILL] 정보 추출 시작...")
            print(f"[TALK_TO_FILL] 대상 필드들: {[f['field'] for f in unfilled[:5]]}")
            
            extraction_response = extraction_chain.invoke({
                "target_fields": target_fields_str,
                "user_response": user_input
            })
            
            # 응답에서 JSON 추출
            response_text = extraction_response.content if hasattr(extraction_response, 'content') else str(extraction_response)
            print(f"[TALK_TO_FILL] LLM 추출 응답: {response_text[:200]}")
            
            # JSON 부분만 추출
            json_match = re.search(r'\{[^{}]*\}', response_text)
            if json_match:
                extracted = json.loads(json_match.group())
                print(f"[TALK_TO_FILL] ✅ 추출 성공: {extracted}")
            else:
                extracted = {}
                print(f"[TALK_TO_FILL] ⚠️ JSON을 찾을 수 없음")
                
        except Exception as e:
            print(f"[TALK_TO_FILL] ❌ 정보 추출 오류: {e}")
            extracted = {}
        
        # 추출된 정보로 폼 업데이트
        for field_name, value in extracted.items():
            if value:
                update_form_field(session_id, current_doc, field_name, value)
        
        # 사용자가 "필요없음", "해당없음" 등을 말하면 현재 질문한 필드들을 건너뛰기
        skip_keywords = ["필요없", "해당없", "해당 없", "모르겠", "없어", "아니", "건너뛰", "스킵"]
        if any(keyword in user_input for keyword in skip_keywords) and not extracted:
            print(f"[TALK_TO_FILL] ⏭️ 사용자가 필드 스킵 요청")
            # 현재 물어본 필드들(최대 5개)을 "N/A"로 채우기
            for field_info in unfilled[:5]:
                update_form_field(session_id, current_doc, field_info['field'], "N/A")
                print(f"[TALK_TO_FILL]   - {field_info['field']} → N/A")
    else:
        extracted = {}
    
    # 업데이트된 미작성 필드 목록
    unfilled = get_unfilled_fields(session_id, current_doc)
    
    # 현재 문서가 완료되었는지 확인
    if not unfilled and current_doc:
        # 다음 문서로 이동
        doc_names = list(session["documents"].keys())
        current_idx = doc_names.index(current_doc)
        
        if current_idx + 1 < len(doc_names):
            session["current_document"] = doc_names[current_idx + 1]
            current_doc = session["current_document"]
            unfilled = get_unfilled_fields(session_id, current_doc)
        else:
            # 모든 문서 완료
            session["completed"] = True
    
    # 대화 응답 생성
    unfilled_str = "\n".join([
        f"- {f['description']} ({f['field']})" 
        for f in unfilled[:5]
    ]) if unfilled else "모든 필드가 채워졌습니다."
    
    form_chain = create_form_chain(session_id)
    config = {"configurable": {"session_id": session_id}}
    
    try:
        print(f"[TALK_TO_FILL] 응답 생성 시작...")
        print(f"[TALK_TO_FILL]   - 카테고리: {session['category']}")
        print(f"[TALK_TO_FILL]   - 현재 문서: {current_doc or '없음'}")
        print(f"[TALK_TO_FILL]   - 미작성 필드 수: {len(unfilled) if unfilled else 0}")
        
        response = form_chain.invoke(
            {
                "category": session["category"],
                "current_document": current_doc or "없음",
                "unfilled_fields": unfilled_str,
                "user_input": user_input
            },
            config=config
        )
        
        response_text = response.content if hasattr(response, 'content') else str(response)
        print(f"[TALK_TO_FILL] ✅ 응답 생성 성공: {response_text[:150]}")
    except Exception as e:
        print(f"[TALK_TO_FILL] ❌ 응답 생성 오류: {e}")
        import traceback
        traceback.print_exc()
        response_text = "죄송합니다. 일시적인 오류가 발생했습니다. 다시 말씀해주시겠어요?"
    
    # 완료 여부 확인
    all_unfilled = get_unfilled_fields(session_id)
    is_completed = len(all_unfilled) == 0
    
    print(f"[TALK_TO_FILL] 완료 여부 체크:")
    print(f"[TALK_TO_FILL]   - 전체 미작성 필드 수: {len(all_unfilled)}")
    print(f"[TALK_TO_FILL]   - 완료: {is_completed}")
    
    if is_completed:
        session["completed"] = True
        print(f"[TALK_TO_FILL] 🎉 모든 서류 작성 완료!")
    
    return {
        "response": response_text[:300],
        "extracted_fields": extracted,
        "form_state": {
            "category": session["category"],
            "current_document": current_doc,
            "documents": {
                doc_name: {
                    "filled_count": doc["filled_count"],
                    "total_count": doc["total_count"],
                    "fields": doc["fields"]
                }
                for doc_name, doc in session["documents"].items()
            }
        },
        "unfilled_count": len(all_unfilled),
        "completed": is_completed
    }


def get_filled_form(session_id: str) -> Optional[Dict[str, Any]]:
    """
    완성된 폼 데이터를 반환합니다.
    """
    session = get_form_session(session_id)
    if not session:
        return None
    
    result = {
        "category": session["category"],
        "documents": {}
    }
    
    for doc_name, doc in session["documents"].items():
        result["documents"][doc_name] = doc["fields"]
    
    return result


# API 엔드포인트용 Pydantic 모델
class FormConversationRequest(BaseModel):
    """폼 대화 요청 모델"""
    session_id: str
    user_input: str
    category: Optional[str] = None


class FormConversationResponse(BaseModel):
    """폼 대화 응답 모델"""
    response: str
    extracted_fields: Dict[str, str]
    form_state: Optional[Dict[str, Any]]
    unfilled_count: int
    completed: bool
    error: Optional[str] = None

