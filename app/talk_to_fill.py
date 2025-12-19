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
from langchain_core.messages import AIMessage, HumanMessage
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

# 공통 필드 매핑: 같은 카테고리 내에서만 같은 의미의 필드들을 그룹화
# 예: 청년월세 신청 시 위임장과 대리수령 사이에서만 공통 필드 자동 채움
COMMON_FIELD_GROUPS_BY_CATEGORY = {
    # ========== 청년월세 (4_Monthly) ==========
    "청년월세": [
        # 그룹 1: 본인 이름 (위임하는 사람 = 수급자)
        {
            "delegator.name",              # 위임장: 위임하는 사람 이름
            "recipient.name",              # 대리수령: 수급자 이름
            "signature.applicant_name",    # 서명
            "signature.reporter_name"      # 서명
        },
        # 그룹 2: 본인 생년월일
        {
            "delegator.birthdate",         # 위임장
            "recipient.birthdate"          # 대리수령
        },
        # 그룹 3: 본인 전화번호
        {
            "delegator.number",            # 위임장
            "recipient.number"             # 대리수령
        },
        # 그룹 4: 본인 휴대전화
        {
            "recipient.mobile"             # 대리수령
        },
        # 그룹 5: 본인 주소
        {
            "delegator.address",           # 위임장
            "recipient.address"            # 대리수령
        },
        # 그룹 6: 대리인 이름
        {
            "delegate.name",                      # 위임장: 위임받는 사람
            "representative_recipient.name"       # 대리수령: 대리 수령인
        },
        # 그룹 7: 대리인 생년월일
        {
            "delegate.birthdate",                 # 위임장
            "representative_recipient.birthdate"  # 대리수령
        },
        # 그룹 8: 대리인 전화번호
        {
            "delegate.number",                    # 위임장
            "representative_recipient.phone",     # 대리수령
            "representative_recipient.number"     # 대리수령
        },
        # 그룹 9: 대리인 주소
        {
            "delegate.address",                   # 위임장
            "representative_recipient.address"    # 대리수령
        },
        # 그룹 10: 관계
        {
            "delegate.relationship_to_delegator",                 # 위임장
            "representative_recipient.relationship_to_recipient"  # 대리수령
        }
    ],
    
    # ========== 국민연금 (1_Welfare) ==========
    "국민연금": [
        # 그룹 1: 본인 이름
        {
            "person.name",                 # 국민연금신고서: 가입자/수급권자
            "reporter.name",               # 국민연금신고서: 신고인 (본인일 때)
            "subscriber.name",             # 국민연금가입자증명서
            "signature.applicant_name",    # 서명
            "signature.reporter_name"      # 서명
        },
        # 그룹 2: 주민등록번호
        {
            "person.resident_number",      # 국민연금신고서
            "reporter.resident_number",    # 국민연금신고서
            "subscriber.resident_number"   # 국민연금가입자증명서
        },
        # 그룹 3: 전화번호
        {
            "person.phone",                # 국민연금신고서
            "reporter.phone",              # 국민연금신고서
            "subscriber.phone"             # 국민연금가입자증명서
        },
        # 그룹 4: 휴대전화
        {
            "person.mobile",               # 국민연금신고서
            "reporter.mobile",             # 국민연금신고서
            "subscriber.mobile"            # 국민연금가입자증명서
        },
        # 그룹 5: 주소
        {
            "person.address",              # 국민연금신고서
            "reporter.address",            # 국민연금신고서
            "subscriber.address"           # 국민연금가입자증명서
        }
    ],
    
    # ========== 전입신고 (2_Report) ==========
    "전입신고": [
        # 단일 문서이므로 공통 필드 없음
    ],
    
    # ========== 토지-건축물 (3_Land) ==========
    "토지-건축물": [
        # 단일 문서이므로 공통 필드 없음
    ],
    
    # ========== 주거급여 (5_Salary) ==========
    "주거급여": [
        # 그룹 1: 본인 이름
        {
            "recipient.name",              # 근로활동및소득신고서: 수급권자
            "applicant.name",              # 사회보장급여신청서: 신청인
            "signature.applicant_name",    # 서명
            "signature.reporter_name",     # 서명
            "bank_account.name"            # 사회보장급여신청서: 예금주
        },
        # 그룹 2: 생년월일
        {
            "recipient.birthdate"          # 근로활동및소득신고서
        },
        # 그룹 3: 주민등록번호
        {
            "applicant.resident_number"    # 사회보장급여신청서
        },
        # 그룹 4: 전화번호
        {
            "applicant.phone"              # 사회보장급여신청서
        },
        # 그룹 5: 휴대전화
        {
            "applicant.mobile"             # 사회보장급여신청서
        },
        # 그룹 6: 주소
        {
            "recipient.address",                 # 근로활동및소득신고서
            "applicant.address.registered"       # 사회보장급여신청서
        },
        # 그룹 7: 은행 계좌
        {
            "bank_account.bank_name"       # 금융기관명
        },
        {
            "bank_account.account_number"  # 계좌번호
        }
    ]
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
    하위 폴더도 탐색합니다.
    
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
    
    # 하위 폴더를 포함하여 모든 파일 탐색
    for root, dirs, files in os.walk(folder_path):
        print(f"[DEBUG] 탐색 중인 폴더: {root}")
        print(f"[DEBUG] 파일 목록: {files}")
        
        for filename in files:
            # _좌표.json 파일은 건너뜀 (PDF 생성용 좌표 파일)
            if '_좌표' in filename:
                print(f"[DEBUG] ⏭️  좌표 파일 건너뜀: {filename}")
                continue
                
            # .txt 또는 .json 파일만 처리
            if filename.endswith('.txt') or filename.endswith('.json'):
                file_path = os.path.join(root, filename)
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
    
    # 히스토리가 너무 길어지면 최근 6개만 유지 (3턴 = 사용자3 + AI3)
    # 너무 긴 히스토리는 LLM이 이전 질문을 참고해서 중복 질문하게 만듦
    history = chat_history_store[session_id]
    if len(history.messages) > 6:
        # 최근 6개 메시지만 유지
        history.messages = history.messages[-6:]
        print(f"[CHAT_HISTORY] 히스토리 정리: 최근 6개만 유지")
    
    return history


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
        "completed": False,
        "guardian_checked": False,  # 후견인 존재 여부 확인 여부
        "guardian_exists": None     # 후견인 존재 여부 (True/False/None)
    }
    
    total_all_fields = 0
    for doc_name, doc_data in documents.items():
        field_count = len(doc_data["fields"])
        total_all_fields += field_count
        form_state["documents"][doc_name] = {
            "fields": {field: "" for field in doc_data["fields"].keys()},
            "descriptions": doc_data["descriptions"],
            "template": doc_data["fields"],  # 원본 템플릿 저장
            "filled_count": 0,
            "total_count": field_count
        }
        print(f"[FIELD_MEMORY] 📄 {doc_name} 문서: {field_count}개 필드")
    
    print(f"[FIELD_MEMORY] 📊 세션 초기화 완료 - 전체 필드 수: {total_all_fields}개 (모든 문서 합계)")
    
    # 첫 번째 문서를 현재 문서로 설정
    if documents:
        form_state["current_document"] = list(documents.keys())[0]
    
    # 세션을 먼저 저장 (get_unfilled_fields()가 세션을 읽어야 함)
    form_session_store[session_id] = form_state
    
    # 실제 채워야 할 필드 수 계산 (공통 필드 그룹 처리 후)
    # 세션이 생성된 직후이므로 모든 필드가 비어있음
    initial_unfilled = get_unfilled_fields(session_id)
    form_state["initial_total_fields"] = len(initial_unfilled)
    form_session_store[session_id]["initial_total_fields"] = len(initial_unfilled)  # 세션에도 저장
    print(f"[FIELD_MEMORY] 📊 실제 채워야 할 필드 수: {form_state['initial_total_fields']}개 (공통 필드 그룹 처리 후)")
    
    return form_state


def get_form_session(session_id: str) -> Optional[Dict[str, Any]]:
    """세션의 폼 상태를 반환합니다."""
    return form_session_store.get(session_id)


def update_form_field(session_id: str, document_name: str, field_name: str, value: str) -> bool:
    """
    특정 필드의 값을 업데이트합니다.
    공통 필드가 있으면 다른 문서의 같은 의미 필드도 자동으로 채웁니다.
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
        # ⚠️ 주의: 빈 문자열("")도 "채워진 값"으로 간주 (체크박스 필드에서 "체크하지 않음"을 의미)
        # 하지만 unfilled 판단은 별도 로직에서 처리 (get_unfilled_fields에서 빈 문자열 제외하지 않음)
        # 따라서 filled_count는 업데이트하지만, unfilled 목록에는 포함되지 않도록 get_unfilled_fields를 수정해야 함
        
        # 기존 로직 유지: 빈 문자열도 값으로 간주
        if old_value == "" and value != "":
            doc["filled_count"] += 1
        elif old_value != "" and value == "":
            doc["filled_count"] -= 1
        
        # 🔥 공통 필드 자동 채우기
        # 현재 필드와 같은 그룹의 필드들을 찾아서 자동으로 채움
        auto_fill_common_fields(session_id, field_name, value)
        
        # 📅 날짜 기간 자동 계산
        # 시작/종료 년월이 모두 채워지면 자동으로 기간을 계산
        auto_calculate_period(session_id, document_name, field_name)
        
        return True
    
    return False


def auto_fill_common_fields(session_id: str, source_field: str, value: str):
    """
    공통 필드 자동 채우기: 같은 카테고리 내에서 한 필드가 채워지면 같은 그룹의 다른 필드들도 자동으로 채움
    """
    session = form_session_store.get(session_id)
    if not session:
        return
    
    # 현재 세션의 카테고리 가져오기
    category = session.get("category")
    if not category:
        return
    
    # 해당 카테고리의 공통 필드 그룹 가져오기
    category_groups = COMMON_FIELD_GROUPS_BY_CATEGORY.get(category, [])
    if not category_groups:
        return  # 해당 카테고리에 공통 필드 그룹이 없으면 무시
    
    # 현재 필드가 속한 그룹 찾기
    related_fields = None
    for group in category_groups:
        if source_field in group:
            related_fields = group
            break
    
    if not related_fields:
        return  # 공통 필드가 아니면 무시
    
    print(f"[AUTO_FILL] '{source_field}' 필드가 업데이트됨 (카테고리: {category})")
    print(f"[AUTO_FILL] 관련 필드들: {related_fields}")
    print(f"[AUTO_FILL] 채울 값: {value}")
    
    # 같은 그룹의 다른 필드들을 같은 카테고리의 모든 문서에서 찾아서 채우기
    for doc_name, doc_data in session["documents"].items():
        for field in related_fields:
            if field == source_field:
                continue  # 원본 필드는 건너뜀
            
            if field in doc_data["fields"]:
                old_value = doc_data["fields"][field]
                
                # 이미 채워진 필드는 덮어쓰지 않음
                if old_value and old_value != "":
                    print(f"[AUTO_FILL] ⏭️  {doc_name}.{field} - 이미 값이 있음: {old_value}")
                    continue
                
                # 자동으로 값 채우기
                doc_data["fields"][field] = value
                doc_data["filled_count"] += 1
                print(f"[AUTO_FILL] ✅ {doc_name}.{field} = {value}")


def auto_calculate_period(session_id: str, document_name: str, field_name: str):
    """
    날짜 기간 자동 계산: 시작/종료 년월이 모두 채워지면 자동으로 기간을 계산
    
    예시:
    - receive_period.start_year: 2024
    - receive_period.start_month: 01
    - receive_period.end_year: 2024
    - receive_period.end_month: 03
    → receive_period.total_months: 2 (자동 계산)
    """
    session = form_session_store.get(session_id)
    if not session:
        return
    
    doc = session["documents"].get(document_name)
    if not doc:
        return
    
    # 날짜 필드가 업데이트되었는지 확인
    date_field_patterns = [
        "start_year", "start_month", "end_year", "end_month",
        "start_date", "end_date"
    ]
    
    if not any(pattern in field_name for pattern in date_field_patterns):
        return  # 날짜 관련 필드가 아니면 무시
    
    # 필드 이름에서 prefix 추출 (예: "receive_period.start_year" → "receive_period")
    if "." in field_name:
        prefix = field_name.rsplit(".", 1)[0]
    else:
        return
    
    # 시작/종료 년월 필드 확인
    start_year_field = f"{prefix}.start_year"
    start_month_field = f"{prefix}.start_month"
    end_year_field = f"{prefix}.end_year"
    end_month_field = f"{prefix}.end_month"
    total_months_field = f"{prefix}.total_months"
    
    # 모든 필드가 존재하는지 확인
    if not all(field in doc["fields"] for field in [
        start_year_field, start_month_field, end_year_field, end_month_field, total_months_field
    ]):
        return  # 필요한 필드가 없으면 무시
    
    # 시작/종료 년월 값 가져오기
    start_year = doc["fields"][start_year_field]
    start_month = doc["fields"][start_month_field]
    end_year = doc["fields"][end_year_field]
    end_month = doc["fields"][end_month_field]
    
    # 모든 값이 채워져 있는지 확인
    if not all([start_year, start_month, end_year, end_month]):
        return  # 값이 하나라도 없으면 계산 불가
    
    try:
        # 문자열을 정수로 변환
        start_year = int(start_year)
        start_month = int(start_month)
        end_year = int(end_year)
        end_month = int(end_month)
        
        # 개월 수 계산
        total_months = (end_year - start_year) * 12 + (end_month - start_month)
        
        # 기간이 음수면 0으로 설정
        if total_months < 0:
            total_months = 0
        
        # total_months 필드 자동 채우기
        old_value = doc["fields"][total_months_field]
        doc["fields"][total_months_field] = str(total_months)
        
        # 채워진 필드 수 업데이트 (이전에 비어있었다면)
        if not old_value or old_value == "":
            doc["filled_count"] += 1
        
        print(f"[AUTO_CALC] 📅 기간 자동 계산: {start_year}.{start_month:02d} ~ {end_year}.{end_month:02d} = {total_months}개월")
        print(f"[AUTO_CALC] ✅ {document_name}.{total_months_field} = {total_months}")
        
    except (ValueError, TypeError) as e:
        print(f"[AUTO_CALC] ❌ 기간 계산 실패: {e}")
        return


def get_unfilled_fields(session_id: str, document_name: str = None) -> List[Dict[str, str]]:
    """
    아직 채워지지 않은 필드 목록을 반환합니다.
    자동 계산 필드는 제외됩니다.
    공통 필드 그룹을 고려하여 같은 의미의 필드는 하나만 반환합니다.
    
    주의: document_name 파라미터는 무시되고 항상 모든 문서를 체크합니다.
    공통 필드 그룹 처리를 위해 모든 문서를 함께 확인해야 합니다.
    
    후견인 필드의 경우, 먼저 후견인 존재 여부를 확인합니다.
    """
    session = form_session_store.get(session_id)
    if not session:
        return []
    
    # 자동 계산되는 필드 패턴 (사용자에게 묻지 않음)
    auto_calculated_patterns = [
        "total_months",  # 수령 기간 (개월 수)
        "period",        # 기간
        "duration",      # 기간
        "total_days",    # 총 일수
    ]
    
    category = session.get("category")
    category_groups = COMMON_FIELD_GROUPS_BY_CATEGORY.get(category, []) if category else []
    
    # 후견인 필드 패턴 (guardian.으로 시작하는 필드)
    guardian_field_pattern = "guardian."
    
    # 후견인 필드가 있는지 확인
    has_guardian_fields = False
    guardian_fields = []
    for doc_name, doc_data in session["documents"].items():
        for field_name, value in doc_data["fields"].items():
            if guardian_field_pattern in field_name:
                has_guardian_fields = True
                if value == "":
                    guardian_fields.append({
                        "document": doc_name,
                        "field": field_name,
                        "description": doc_data["descriptions"].get(field_name, field_name)
                    })
    
    # 후견인 존재 여부 확인이 필요한 경우 (후견인 필드가 있고 아직 확인되지 않음)
    guardian_checked = session.get("guardian_checked", False)
    if has_guardian_fields and not guardian_checked and guardian_fields:
        # 먼저 후견인 존재 여부 확인을 위한 특별한 필드 반환
        return [{
            "document": guardian_fields[0]["document"],
            "field": "__guardian_exists__",  # 특별한 필드명
            "description": "후견인이 있으신가요?"
        }]
    
    # 후견인이 없다고 확인된 경우, 후견인 필드는 제외
    if guardian_checked and session.get("guardian_exists") == False:
        # 후견인 필드는 건너뛰기 (이미 N/A로 채워짐)
        pass
    
    # 공통 필드 그룹에서 이미 채워진 필드 추적 (모든 문서에서 확인)
    filled_groups = set()  # 이미 채워진 그룹의 인덱스
    
    # 공통 필드 그룹 매핑 생성 및 채워진 그룹 확인
    for group_idx, group in enumerate(category_groups):
        # 그룹 내 필드 중 하나라도 채워져 있으면 해당 그룹은 제외
        for field in group:
            for doc_name, doc_data in session["documents"].items():
                if field in doc_data["fields"]:
                    field_value = doc_data["fields"][field]
                    if field_value and field_value != "" and field_value != "N/A":
                        filled_groups.add(group_idx)
                        break
            if group_idx in filled_groups:
                break
    
    # 모든 문서의 미작성 필드를 먼저 수집
    all_unfilled_fields = []  # (doc_name, field_name, description, is_common_field, group_idx)
    
    # 모든 문서를 체크 (document_name 파라미터 무시)
    for doc_name, doc_data in session["documents"].items():
        for field_name, value in doc_data["fields"].items():
            # 자동 계산 필드는 제외
            if any(pattern in field_name for pattern in auto_calculated_patterns):
                continue
            
            # 후견인이 없다고 확인된 경우, 후견인 필드는 제외
            if guardian_checked and session.get("guardian_exists") == False:
                if guardian_field_pattern in field_name:
                    continue
            
            if value == "":
                # 공통 필드 그룹에 속하는지 확인
                is_common_field = False
                found_group_idx = None
                for group_idx, group in enumerate(category_groups):
                    if field_name in group:
                        is_common_field = True
                        found_group_idx = group_idx
                        break
                
                description = doc_data["descriptions"].get(field_name, field_name)
                all_unfilled_fields.append({
                    "document": doc_name,
                    "field": field_name,
                    "description": description,
                    "is_common_field": is_common_field,
                    "group_idx": found_group_idx
                })
    
    # 공통 필드 그룹 처리: 같은 그룹의 필드 중 하나만 선택
    unfilled = []
    processed_common_groups = set()  # 이미 처리된 공통 필드 그룹
    
    for field_info in all_unfilled_fields:
        if field_info["is_common_field"]:
            group_idx = field_info["group_idx"]
            # 이미 채워진 그룹이면 제외
            if group_idx in filled_groups:
                continue
            # 같은 그룹의 필드가 이미 처리되었으면 제외 (하나만 반환)
            if group_idx in processed_common_groups:
                continue
            # 첫 번째로 발견된 그룹의 필드만 추가
            processed_common_groups.add(group_idx)
            unfilled.append({
                "document": field_info["document"],
                "field": field_info["field"],
                "description": field_info["description"]
            })
        else:
            # 공통 필드가 아닌 경우 그대로 추가
            unfilled.append({
                "document": field_info["document"],
                "field": field_info["field"],
                "description": field_info["description"]
            })
    
    # 디버깅: 전체 필드 통계 출력
    total_fields_count = 0
    auto_calculated_count = 0
    filled_fields_count = 0
    
    for doc_name, doc_data in session["documents"].items():
        for field_name, value in doc_data["fields"].items():
            total_fields_count += 1
            if any(pattern in field_name for pattern in auto_calculated_patterns):
                auto_calculated_count += 1
            elif value and value != "" and value != "N/A":
                filled_fields_count += 1
    
    unfilled_fields_count = len(unfilled)
    
    print(f"[FIELD_MEMORY] 📊 필드 통계 (모든 문서):")
    print(f"[FIELD_MEMORY]   - 전체 필드: {total_fields_count}개")
    print(f"[FIELD_MEMORY]   - 자동 계산 필드: {auto_calculated_count}개 (제외됨)")
    print(f"[FIELD_MEMORY]   - 채워진 필드: {filled_fields_count}개")
    print(f"[FIELD_MEMORY]   - 채워야 할 필드: {unfilled_fields_count}개 (공통 필드 그룹 처리 후)")
    
    # 디버깅: 채워야 할 필드 목록 로그 출력 (전체)
    if unfilled:
        print(f"[FIELD_MEMORY] 📋 채워야 할 필드 목록 ({len(unfilled)}개):")
        for idx, field_info in enumerate(unfilled, 1):
            print(f"[FIELD_MEMORY]   {idx}. {field_info['document']}.{field_info['field']} - {field_info['description']}")
    else:
        print(f"[FIELD_MEMORY] ✅ 채워야 할 필드 없음 (모든 필드 채워짐)")
    
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
    ("system", """반드시 한국어로만 응답하세요. 중국어, 한자, 영어 사용 금지.

당신은 {category} 신청을 도와주는 상담원입니다.

[방금 사용자가 입력한 정보]
{just_extracted}

[이미 수집 완료 - 다시 묻지 마세요]
{filled_info}

[아직 필요한 정보 - 첫 번째만 질문하세요]
{unfilled_fields}

규칙:
1. 반드시 한국어만 사용하세요.
2. "이미 수집 완료" 목록에 있는 정보는 절대 다시 묻지 마세요.
3. "아직 필요한 정보" 목록의 첫 번째 항목만 질문하세요.
4. 응답 형식: "네, OOO 확인했습니다. (질문)"
5. 반드시 물음표(?)로 끝나는 질문을 하세요.
6. 한 번에 1개 정보만 물어보세요.
7. "완료", "감사합니다", "끝" 같은 말 하지 마세요.
8. "위와 같음", "상동", "동일" 같은 표현 사용 금지.
9. 사용자에게 "필요한 게 있나요?" 묻지 마세요. 당신이 직접 질문하세요.
10. "후견인이 있으신가요?" 같은 질문이 나오면, 사용자가 "없다"고 답하면 후견인 관련 모든 필드는 N/A로 처리되고 더 이상 묻지 않습니다. "있다"고 답하면 후견인 관련 필드들을 순차적으로 질문하세요."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{user_input}")
])

# 정보 추출 프롬프트
extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", """사용자 응답에서 정보를 추출하세요.

추출 대상 필드:
{target_fields}

규칙:
1. 사용자가 직접 말한 정보만 추출하세요. 추측 금지.
2. 날짜: YYYY-MM-DD 형식
3. 전화번호: 010-XXXX-XXXX 형식
4. 긍정 답변(네, 예, 원해요): "V"
5. 부정 답변(아니오, 필요없어): "N/A"
6. "위와 같음", "상동", "동일"은 유효한 값이 아닙니다. 무시하세요.

JSON만 반환하세요.
예: {{"delegator.name": "홍길동", "delegator.address": "서울시 강남구"}}
추출할 정보 없으면: {{}}"""),
    ("human", "사용자: {user_response}\n질문: {last_question}")
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
    
    # 최종 확인 단계에서 사용자가 수정을 원하는 경우 처리
    if session.get("final_confirmation_shown") and not session.get("completed"):
        # 사용자가 부정적인 답변을 한 경우 또는 확인 요청한 경우 수정 모드로 전환
        negative_keywords = ["아니", "아뇨", "아니요", "싫어", "수정", "바꿔", "고쳐", "틀렸", "잘못", "보여줘", "보여줘", "확인", "다시", "보기", "체크"]
        if any(keyword in user_input for keyword in negative_keywords):
            # 수정 모드 활성화
            session["final_confirmation_shown"] = False  # 최종 확인 플래그 초기화
            print(f"[TALK_TO_FILL] 🔄 수정 모드 진입 - 사용자가 변경 요청")
            
            return {
                "response": "알겠습니다! 어떤 정보를 수정하시겠어요? 수정하실 내용을 말씀해주세요.",
                "extracted_fields": {},
                "form_state": {
                    "category": session["category"],
                    "current_document": session["current_document"],
                    "total_fields": session.get("initial_total_fields", 0),
                    "documents": {
                        doc_name: {
                            "filled_count": doc["filled_count"],
                            "total_count": doc["total_count"],
                            "fields": doc["fields"]
                        }
                        for doc_name, doc in session["documents"].items()
                    }
                },
                "unfilled_count": 0,
                "completed": False,
                "edit_mode": True
            }
    
    # 현재 문서와 채워지지 않은 필드 가져오기
    current_doc = session["current_document"]
    unfilled = get_unfilled_fields(session_id, current_doc)
    
    print(f"[TALK_TO_FILL] 현재 문서: {current_doc}")
    print(f"[TALK_TO_FILL] 미작성 필드 수: {len(unfilled)}")
    if unfilled:
        print(f"[TALK_TO_FILL] 처음 5개 미작성 필드: {[f['field'] for f in unfilled[:5]]}")
    
    # 후견인 존재 여부 확인 단계 처리
    if unfilled and len(unfilled) > 0 and unfilled[0]["field"] == "__guardian_exists__":
        # 후견인 존재 여부 질문에 대한 사용자 응답 처리
        negative_keywords = ["없", "아니", "아뇨", "아니요", "필요없", "해당없", "해당 없", "없어요", "없습니다"]
        positive_keywords = ["있", "예", "네", "있어요", "있습니다", "있어"]
        
        user_input_lower = user_input.lower()
        has_negative = any(keyword in user_input for keyword in negative_keywords)
        has_positive = any(keyword in user_input for keyword in positive_keywords)
        
        if has_negative and not has_positive:
            # 후견인이 없는 경우: 모든 후견인 필드를 N/A로 채우기
            print(f"[TALK_TO_FILL] 🔍 후견인이 없다고 확인됨 - 모든 후견인 필드를 N/A로 채움")
            session["guardian_checked"] = True
            session["guardian_exists"] = False
            
            # 모든 문서에서 후견인 필드 찾아서 N/A로 채우기
            guardian_fields_filled = 0
            for doc_name, doc_data in session["documents"].items():
                for field_name in doc_data["fields"].keys():
                    if "guardian." in field_name:
                        old_value = doc_data["fields"][field_name]
                        if old_value == "":
                            doc_data["fields"][field_name] = "N/A"
                            doc_data["filled_count"] += 1
                            guardian_fields_filled += 1
                            print(f"[TALK_TO_FILL]   ✅ {doc_name}.{field_name} = N/A")
            
            print(f"[TALK_TO_FILL] ✅ 후견인 필드 {guardian_fields_filled}개를 N/A로 채움")
            
            # 다음 필드로 진행
            updated_unfilled = get_unfilled_fields(session_id)
            if updated_unfilled:
                next_field_desc = updated_unfilled[0]['description']
                return {
                    "response": f"알겠습니다. 후견인 관련 정보는 제외하겠습니다. {next_field_desc}는 어떻게 되시나요?",
                    "extracted_fields": {},
                    "form_state": {
                        "category": session["category"],
                        "current_document": current_doc,
                        "total_fields": session.get("initial_total_fields", 0),
                        "documents": {
                            doc_name: {
                                "filled_count": doc["filled_count"],
                                "total_count": doc["total_count"],
                                "fields": doc["fields"]
                            }
                            for doc_name, doc in session["documents"].items()
                        }
                    },
                    "unfilled_count": len(updated_unfilled),
                    "completed": False
                }
        elif has_positive:
            # 후견인이 있는 경우: 후견인 필드들을 순차적으로 질문
            print(f"[TALK_TO_FILL] 🔍 후견인이 있다고 확인됨 - 후견인 필드들을 질문하도록 설정")
            session["guardian_checked"] = True
            session["guardian_exists"] = True
            
            # 다음 필드로 진행 (후견인 필드 중 첫 번째)
            updated_unfilled = get_unfilled_fields(session_id)
            if updated_unfilled:
                next_field_desc = updated_unfilled[0]['description']
                return {
                    "response": f"알겠습니다. {next_field_desc}는 어떻게 되시나요?",
                    "extracted_fields": {},
                    "form_state": {
                        "category": session["category"],
                        "current_document": current_doc,
                        "total_fields": session.get("initial_total_fields", 0),
                        "documents": {
                            doc_name: {
                                "filled_count": doc["filled_count"],
                                "total_count": doc["total_count"],
                                "fields": doc["fields"]
                            }
                            for doc_name, doc in session["documents"].items()
                        }
                    },
                    "unfilled_count": len(updated_unfilled),
                    "completed": False
                }
        else:
            # 명확하지 않은 응답: 다시 질문
            return {
                "response": "후견인이 있으신가요, 없으신가요?",
                "extracted_fields": {},
                "form_state": {
                    "category": session["category"],
                    "current_document": current_doc,
                    "total_fields": session.get("initial_total_fields", 0),
                    "documents": {
                        doc_name: {
                            "filled_count": doc["filled_count"],
                            "total_count": doc["total_count"],
                            "fields": doc["fields"]
                        }
                        for doc_name, doc in session["documents"].items()
                    }
                },
                "unfilled_count": len(unfilled),
                "completed": False
            }
    
    # 사용자 응답에서 정보 추출
    if unfilled:
        target_fields_str = "\n".join([
            f"- {f['field']}: {f['description']}" 
            for f in unfilled[:5]  # 최대 5개 필드만 대상
        ])
        
        # 이전 질문 가져오기 (대화 히스토리에서 마지막 AI 메시지)
        history = get_chat_history(session_id)
        last_question = ""
        if history.messages and len(history.messages) > 0:
            # 히스토리 순서: [HumanMessage1, AIMessage1, HumanMessage2, AIMessage2, ...]
            # 가장 최신 AI 메시지를 찾기 위해 역순으로 순회
            # 또는 인덱스로 직접 접근: 마지막이 HumanMessage면 -2, AIMessage면 -1
            # 안전하게 역순 순회로 처리
            for i in range(len(history.messages) - 1, -1, -1):
                msg = history.messages[i]
                if isinstance(msg, AIMessage):
                    last_question = msg.content if hasattr(msg, 'content') else str(msg)
                    print(f"[TALK_TO_FILL] 최신 AI 질문 찾음 (인덱스 {i}): {last_question[:50]}...")
                    break
        
        extraction_chain = extraction_prompt | llm
        
        try:
            print(f"[TALK_TO_FILL] 정보 추출 시작...")
            print(f"[TALK_TO_FILL] 대상 필드들: {[f['field'] for f in unfilled[:5]]}")
            print(f"[TALK_TO_FILL] 현재 사용자 입력: {user_input[:100] if user_input else '(없음)'}")
            print(f"[TALK_TO_FILL] 이전 AI 질문: {last_question[:100] if last_question else '(없음)'}")
            print(f"[TALK_TO_FILL] 히스토리 메시지 수: {len(history.messages) if history.messages else 0}")
            
            extraction_response = extraction_chain.invoke({
                "target_fields": target_fields_str,
                "user_response": user_input,  # 현재 턴의 사용자 입력 (함수 파라미터)
                "last_question": last_question if last_question else "처음 질문"
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
        # 빈 문자열("")도 유효한 값 (체크박스 필드에서 "체크하지 않음"을 의미)
        print(f"[TALK_TO_FILL] 📝 필드 업데이트 시작 - 추출된 필드 수: {len(extracted)}")
        print(f"[FIELD_MEMORY] 🔄 업데이트 전 상태:")
        print(f"[FIELD_MEMORY]   - 채워야 할 필드: {len(unfilled)}개")
        
        for field_name, value in extracted.items():
            if value is not None:  # None이 아니면 업데이트 (빈 문자열 포함)
                # 먼저 현재 문서에서 시도
                success = update_form_field(session_id, current_doc, field_name, value)
                if success:
                    print(f"[TALK_TO_FILL] ✅ 필드 업데이트 성공: {current_doc}.{field_name} = {value}")
                    print(f"[FIELD_MEMORY]   ✅ {current_doc}.{field_name} = '{value}' (채워짐)")
                else:
                    # 현재 문서에 없으면 다른 모든 문서에서 찾아서 업데이트
                    found = False
                    for doc_name in session["documents"].keys():
                        if doc_name != current_doc:
                            success = update_form_field(session_id, doc_name, field_name, value)
                            if success:
                                print(f"[TALK_TO_FILL] ✅ 필드 업데이트 성공 (다른 문서): {doc_name}.{field_name} = {value}")
                                print(f"[FIELD_MEMORY]   ✅ {doc_name}.{field_name} = '{value}' (채워짐)")
                                found = True
                                break
                    if not found:
                        print(f"[TALK_TO_FILL] ⚠️ 필드를 찾을 수 없음: {field_name}")
                        print(f"[FIELD_MEMORY]   ⚠️ 필드를 찾을 수 없음: {field_name}")
        
        # 업데이트 후 상태 출력
        updated_unfilled = get_unfilled_fields(session_id)  # 모든 문서 체크
        print(f"[FIELD_MEMORY] 🔄 업데이트 후 상태:")
        print(f"[FIELD_MEMORY]   - 채워야 할 필드: {len(updated_unfilled)}개 (이전: {len(unfilled)}개)")
        
        # 사용자가 "필요없음", "해당없음" 등을 말하면 현재 질문한 필드들을 건너뛰기
        # 단, 후견인 존재 여부 질문(__guardian_exists__)은 제외 (별도 처리됨)
        skip_keywords = ["필요없", "해당없", "해당 없", "모르겠", "없어", "아니", "건너뛰", "스킵"]
        if any(keyword in user_input for keyword in skip_keywords) and not extracted:
            # 후견인 존재 여부 질문은 스킵하지 않음 (별도 처리됨)
            if unfilled and len(unfilled) > 0 and unfilled[0]["field"] != "__guardian_exists__":
                print(f"[TALK_TO_FILL] ⏭️ 사용자가 필드 스킵 요청")
                # 현재 물어본 필드들(최대 5개)을 "N/A"로 채우기
                for field_info in unfilled[:5]:
                    if field_info['field'] != "__guardian_exists__":
                        update_form_field(session_id, current_doc, field_info['field'], "N/A")
                        print(f"[TALK_TO_FILL]   - {field_info['field']} → N/A")
    else:
        extracted = {}
    
    # 업데이트된 미작성 필드 목록 (모든 문서 체크)
    unfilled = get_unfilled_fields(session_id)
    
    # 모든 필드가 채워졌는지 확인
    # 공통 필드 그룹 처리로 모든 문서의 필드를 함께 관리하므로
    # 문서별 순차 처리는 더 이상 필요 없음
    
    # 대화 응답 생성
    # ⚠️ 중요: "모든 필드가 채워졌습니다" 같은 메시지를 LLM에게 보내지 않기!
    if unfilled and len(unfilled) > 0:
        # 필드명(field)을 숨기고 설명(description)만 보여주기
        unfilled_str = "\n".join([
            f"- {f['description']}" 
            for f in unfilled[:5]
        ])
    else:
        # 모든 필드가 채워진 경우 → 최종 확인 단계
        
        # 최종 확인이 이미 표시되었는지 체크
        if not session.get("final_confirmation_shown"):
            # 첫 번째: 입력된 정보 요약 제공 + 최종 확인 요청
            session["final_confirmation_shown"] = True
            
            # 입력된 정보 요약 생성 (주요 정보만)
            summary_items = []
            for doc_name, doc_data in session["documents"].items():
                for field_name, field_value in list(doc_data["fields"].items())[:10]:  # 처음 10개만
                    if field_value and field_value != "" and field_value != "N/A":
                        field_desc = doc_data["descriptions"].get(field_name, field_name)
                        # 긴 값은 축약
                        display_value = field_value[:30] + "..." if len(field_value) > 30 else field_value
                        summary_items.append(f"• {field_desc}: {display_value}")
            
            summary_text = "\n".join(summary_items[:8])  # 최대 8개 항목만 표시
            more_count = len(summary_items) - 8
            if more_count > 0:
                summary_text += f"\n... 외 {more_count}개 항목"
            
            confirmation_message = (
                f"모든 정보가 입력되었습니다! 📝\n\n"
                f"입력하신 주요 내용:\n{summary_text}\n\n"
                f"이대로 제출하시겠습니까?"
            )
            
            print(f"[TALK_TO_FILL] 📋 최종 확인 단계 - 요약 표시")
            
            return {
                "response": confirmation_message,
                "extracted_fields": {},
                "form_state": {
                    "category": session["category"],
                    "current_document": current_doc,
                    "total_fields": session.get("initial_total_fields", 0),
                    "documents": {
                        doc_name: {
                            "filled_count": doc["filled_count"],
                            "total_count": doc["total_count"],
                            "fields": doc["fields"]
                        }
                        for doc_name, doc in session["documents"].items()
                    }
                },
                "unfilled_count": 0,
                "completed": False,  # 아직 확인 중이므로 False
                "awaiting_confirmation": True  # 최종 확인 대기 중
            }
        else:
            # 두 번째: 사용자가 확인 후 제출
            session["completed"] = True
            print(f"[TALK_TO_FILL] ✅ 사용자 확인 완료 - 제출 처리")
            
            return {
                "response": "감사합니다. 제출이 완료되었습니다!",
                "extracted_fields": {},
                "form_state": {
                    "category": session["category"],
                    "current_document": current_doc,
                    "total_fields": session.get("initial_total_fields", 0),
                    "documents": {
                        doc_name: {
                            "filled_count": doc["filled_count"],
                            "total_count": doc["total_count"],
                            "fields": doc["fields"]
                        }
                        for doc_name, doc in session["documents"].items()
                    }
                },
                "unfilled_count": 0,
                "completed": True
            }
    
    # 실제 채워야 할 필드 수는 세션의 initial_total_fields 사용
    # 이 값은 세션 초기화 시 unfilled_count로 설정됨
    actual_total_fields = session.get("initial_total_fields", 0)
    
    # 이미 채워진 정보 수집 (LLM이 중복 질문하지 않도록)
    filled_info_list = []
    filled_field_descriptions = []  # 필드 설명만 저장 (검증용)
    filled_field_keywords = []  # 검증용 키워드 (더 포괄적)
    
    # 디버깅: 채워진 필드 목록 수집
    filled_fields_detail = []
    
    for doc_name, doc_data in session["documents"].items():
        for field_name, field_value in doc_data["fields"].items():
            if field_value and field_value != "" and field_value != "N/A":
                # 설명 가져오기
                field_desc = doc_data["descriptions"].get(field_name, field_name)
                filled_info_list.append(f"- {field_desc}: {field_value}")
                filled_fields_detail.append({
                    "document": doc_name,
                    "field": field_name,
                    "description": field_desc,
                    "value": field_value
                })
                filled_field_descriptions.append(field_desc)
                
                # 검증용 키워드 추출 (더 포괄적인 매칭을 위해)
                # "위임하는 사람 이름" → ["이름", "성함", "성명"]
                keywords = [field_desc]
                if "이름" in field_desc:
                    keywords.extend(["이름", "성함", "성명"])
                if "생년월일" in field_desc:
                    keywords.extend(["생년월일", "생일", "출생"])
                if "주소" in field_desc:
                    keywords.extend(["주소", "거주지", "사는 곳"])
                if "전화" in field_desc or "번호" in field_desc:
                    keywords.extend(["전화", "연락처", "번호", "핸드폰", "휴대폰"])
                if "관계" in field_desc:
                    keywords.extend(["관계", "어떤 사이"])
                filled_field_keywords.extend(keywords)
    
    # 중복 제거
    filled_field_keywords = list(set(filled_field_keywords))
    
    # 디버깅: 채워진 필드 목록 로그 출력 (전체)
    if filled_fields_detail:
        print(f"[FIELD_MEMORY] ✅ 채워진 필드 ({len(filled_fields_detail)}개):")
        for idx, field_info in enumerate(filled_fields_detail, 1):
            print(f"[FIELD_MEMORY]   {idx}. {field_info['document']}.{field_info['field']} = '{field_info['value']}' ({field_info['description']})")
    else:
        print(f"[FIELD_MEMORY] 📝 채워진 필드 없음 (아직 입력 전)")
    
    if filled_info_list:
        # 모든 채워진 정보를 전달 (제한 없이)
        filled_info_str = "\n".join(filled_info_list)
        filled_info_str += f"\n\n🚨🚨🚨 위 {len(filled_info_list)}개 정보는 이미 수집 완료! 절대 다시 묻지 마세요! 🚨🚨🚨"
    else:
        filled_info_str = "(아직 없음)"
    
    # 방금 추출된 정보 정리 (사용자 답변 확인용)
    just_extracted_str = ""
    if extracted:
        just_extracted_items = []
        for field_name, field_value in extracted.items():
            # 필드 설명 찾기
            field_desc = "정보"
            if current_doc and current_doc in session["documents"]:
                field_desc = session["documents"][current_doc]["descriptions"].get(field_name, field_name)
            just_extracted_items.append(f"- {field_desc}: {field_value}")
        just_extracted_str = "\n".join(just_extracted_items)
    else:
        just_extracted_str = "(방금 추출된 정보 없음 - 사용자가 일반 대화를 하고 있거나 질문에 답하지 않았을 수 있음)"
    
    form_chain = create_form_chain(session_id)
    config = {"configurable": {"session_id": session_id}}
    
    try:
        print(f"[TALK_TO_FILL] 응답 생성 시작...")
        print(f"[TALK_TO_FILL]   - 카테고리: {session['category']}")
        print(f"[TALK_TO_FILL]   - 현재 문서: {current_doc or '없음'}")
        print(f"[TALK_TO_FILL]   - 미작성 필드 수: {len(unfilled)}")
        print(f"[TALK_TO_FILL]   - 미작성 필드 (처음 5개): {[f['field'] for f in unfilled[:5]]}")
        print(f"[TALK_TO_FILL]   - 이미 채워진 정보 수: {len(filled_info_list)}")
        print(f"[TALK_TO_FILL]   - 방금 추출된 정보: {extracted}")
        
        response = form_chain.invoke(
            {
                "category": session["category"],
                "just_extracted": just_extracted_str,
                "filled_info": filled_info_str,
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
    
    # 응답 검증 시스템
    if not is_completed and response_text:
        original_response = response_text
        validation_failed = False
        
        # ========== 검증 1: 완료 메시지 & 역할 혼동 체크 ==========
        completion_keywords = [
            "작성 완료", "완료되었습니다", "완료했습니다", "끝났습니다",
            "모든 정보가 입력", "서류가 완성", "다 되었습니다", "마무리되었습니다",
            "작성이 끝", "입력이 완료", "모두 작성", "감사합니다", "수고하셨습니다",
            "제출하시겠어요", "제출하실", "확인하셨나요", "확인하실",
            "추가로 필요한", "더 필요한", "필요하신 게", "필요한 사항이"
        ]
        
        contains_completion = any(keyword in response_text for keyword in completion_keywords)
        
        if contains_completion:
            print(f"[TALK_TO_FILL] ❌ 검증 실패 (1): 잘못된 완료 메시지 또는 역할 혼동!")
            validation_failed = True
        
        # ========== 검증 2: 질문으로 끝나는지 체크 ==========
        # 응답의 마지막 줄이 물음표(?)로 끝나야 함
        
        # 마지막 줄만 추출 (여러 줄 응답 대응)
        lines = response_text.strip().split('\n')
        last_line = lines[-1].strip() if lines else ""
        
        # 마크다운 굵은 글씨 제거: **텍스트** → 텍스트
        last_line = re.sub(r'\*\*([^*]+)\*\*', r'\1', last_line)
        
        # 마지막 줄이 물음표로 끝나는지 체크
        if not last_line.endswith('?'):
            print(f"[TALK_TO_FILL] ❌ 검증 실패 (2/3): 질문으로 끝나지 않음!")
            print(f"[TALK_TO_FILL]   - 원본 마지막 줄: {lines[-1] if lines else '(없음)'}")
            print(f"[TALK_TO_FILL]   - 정제된 마지막 줄: {last_line}")
            validation_failed = True
        
        # ========== 검증 3: 이미 채워진 필드를 다시 물어보는지 체크 ==========
        # filled_field_keywords를 사용하여 더 포괄적으로 검증
        if filled_field_keywords:
            # 질문 패턴들
            ask_suffixes = [
                "이 어떻게", "은 어떻게", "는 어떻게",
                "을 알려", "를 알려", "을 말씀", "를 말씀",
                "이 뭐", "은 뭐", "는 뭐",
                "을 입력", "를 입력",
                "이요", "요?",  # "이름이요?", "주소요?"
                "을 여쭤", "를 여쭤",
                "이 무엇", "은 무엇", "는 무엇"
            ]
            
            for keyword in filled_field_keywords:
                if len(keyword) < 2:  # 너무 짧은 키워드는 건너뜀
                    continue
                for suffix in ask_suffixes:
                    pattern = f"{keyword}{suffix}"
                    if pattern in response_text:
                        print(f"[TALK_TO_FILL] ❌ 검증 실패 (3/3): 이미 채워진 필드를 다시 물어봄!")
                        print(f"[TALK_TO_FILL]   - 감지된 패턴: '{pattern}'")
                        print(f"[TALK_TO_FILL]   - 원본: {response_text[:200]}")
                        validation_failed = True
                        break
                if validation_failed:
                    break
        
        # ========== 검증 실패 시 응답 자동 수정 ==========
        if validation_failed:
            print(f"[TALK_TO_FILL] 🔧 응답 자동 수정 중...")
            print(f"[TALK_TO_FILL]   - 남은 필드 수: {len(all_unfilled)}")
            
            # 다음 필드로 질문 생성
            if unfilled and len(unfilled) > 0:
                next_field_desc = unfilled[0]['description']
                response_text = f"알겠습니다. {next_field_desc}는 어떻게 되시나요?"
                print(f"[TALK_TO_FILL]   - ✅ 수정된 응답: {response_text}")
            else:
                response_text = "다음 정보를 알려주시겠어요?"
                print(f"[TALK_TO_FILL]   - ✅ 수정된 응답: {response_text}")
    
    if is_completed:
        session["completed"] = True
        print(f"[TALK_TO_FILL] 모든 서류 작성 완료!")
    else:
        print(f"[TALK_TO_FILL] 아직 {len(all_unfilled)}개 필드가 남아있습니다.")
        if all_unfilled:
            print(f"[TALK_TO_FILL]   - 다음 필드들: {[f['field'] for f in all_unfilled[:3]]}")
    
    # 최종 응답 반환 전 검증
    print(f"[TALK_TO_FILL] 최종 응답:")
    print(f"[TALK_TO_FILL]   - completed: {is_completed}")
    print(f"[TALK_TO_FILL]   - unfilled_count: {len(all_unfilled)}")
    print(f"[TALK_TO_FILL]   - response: {response_text[:100]}")
    
    # 실제 채워야 할 필드 수 (세션에 저장된 초기값 사용)
    actual_total_fields = session.get("initial_total_fields", len(all_unfilled))
    
    return {
        "response": response_text[:500],  # 300 → 500으로 확장 (자연스러운 응답을 위해)
        "extracted_fields": extracted,
        "form_state": {
            "category": session["category"],
            "current_document": current_doc,
            "total_fields": actual_total_fields,  # 실제 채워야 할 필드 수 (공통 필드 그룹 처리 후)
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


def fill_common_fields_for_pdf(session_id: str):
    """
    PDF 생성 전에 공통 필드 매핑을 참조하여 모든 문서의 필드를 채웁니다.
    한 문서에만 채워진 공통 필드 값을 다른 문서의 대응 필드에도 자동으로 입력합니다.
    """
    session = form_session_store.get(session_id)
    if not session:
        return
    
    category = session.get("category")
    if not category:
        return
    
    # 해당 카테고리의 공통 필드 그룹 가져오기
    category_groups = COMMON_FIELD_GROUPS_BY_CATEGORY.get(category, [])
    if not category_groups:
        return
    
    print(f"[PDF_FILL] 📝 PDF 생성 전 공통 필드 채우기 시작 - 카테고리: {category}")
    
    # 각 공통 필드 그룹을 순회
    for group_idx, group in enumerate(category_groups):
        # 그룹 내에서 채워진 값 찾기
        filled_value = None
        filled_field = None
        
        for doc_name, doc_data in session["documents"].items():
            for field_name in group:
                if field_name in doc_data["fields"]:
                    value = doc_data["fields"][field_name]
                    if value and value != "" and value != "N/A":
                        filled_value = value
                        filled_field = field_name
                        break
            if filled_value:
                break
        
        # 찾은 값으로 그룹 내 다른 필드들을 채우기
        if filled_value:
            print(f"[PDF_FILL] 🔄 그룹 {group_idx + 1}: '{filled_field}' = '{filled_value}' → 다른 필드에 복사")
            for doc_name, doc_data in session["documents"].items():
                for field_name in group:
                    if field_name in doc_data["fields"]:
                        current_value = doc_data["fields"][field_name]
                        # 비어있는 필드만 채우기
                        if not current_value or current_value == "":
                            doc_data["fields"][field_name] = filled_value
                            print(f"[PDF_FILL]   ✅ {doc_name}.{field_name} = {filled_value}")
    
    print(f"[PDF_FILL] ✅ 공통 필드 채우기 완료")


def get_filled_form(session_id: str) -> Optional[Dict[str, Any]]:
    """
    완성된 폼 데이터를 반환합니다.
    PDF 생성 전에 공통 필드를 채웁니다.
    """
    session = get_form_session(session_id)
    if not session:
        return None
    
    # PDF 생성 전에 공통 필드 채우기
    fill_common_fields_for_pdf(session_id)
    
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

