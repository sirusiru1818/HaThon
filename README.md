# HaThon - 행정복지센터 키오스크 AI 상담 시스템

행정복지센터 키오스크를 위한 AI 기반 카테고리 분류 및 상담 시스템입니다.

## 프로젝트 구조

```
HaThon/
├── app/                    # 메인 애플리케이션 코드
│   ├── __init__.py        # 패키지 초기화 파일
│   ├── category.py        # 카테고리 분류 및 FastAPI 앱 메인 로직
│   └── main.py            # 앱 래퍼 파일
├── docs/                   # 문서 파일
│   └── flowchart.rtf      # 플로우차트 문서
├── scripts/                # 유틸리티 스크립트
│   └── start_server.sh    # 서버 실행 스크립트
├── tests/                  # 테스트 파일
│   ├── 02-llm_application.ipynb
│   └── 03-llm_application.ipynb
├── main.py                 # uvicorn 실행을 위한 루트 레벨 파일
├── requirements.txt        # Python 의존성 패키지
└── README.md              # 프로젝트 설명서
```

## 설치

1. 가상 환경 생성 및 활성화:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2. 의존성 설치:
```bash
pip install -r requirements.txt
```

3. 환경 변수 설정:
`.env` 파일을 생성하고 다음 변수들을 설정하세요:
```
AWS_MODEL_ID=your_model_id
AWS_REGION=your_region
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
```

## 실행

### 방법 1: 스크립트 사용
```bash
./scripts/start_server.sh
```

### 방법 2: 직접 실행
```bash
uvicorn main:app --reload
```

서버가 실행되면 `http://127.0.0.1:8000`에서 접근할 수 있습니다.

## API 엔드포인트

### POST /process
사용자 질문을 받아 카테고리 분류 및 답변을 제공합니다.

**Request Body:**
```json
{
  "text": "사용자 질문 텍스트",
  "session_id": "세션 ID (선택사항)"
}
```

**Response:**
```json
{
  "question": "사용자 질문",
  "category": "카테고리 (국민연금, 전입신고, 토지-건축물, 청년월세, 주거급여, etc)",
  "answer": "답변",
  "message": "처리 상태 메시지",
  "reason": "카테고리 분류 근거",
  "session_id": "세션 ID",
  "is_guidance": true  // etc 카테고리일 때만
}
```

## 주요 기능

- **카테고리 자동 분류**: 사용자 질문을 6개 카테고리 중 하나로 자동 분류
- **세션 기반 대화 유도**: etc 카테고리일 때 세션을 유지하며 다른 카테고리로 자연스럽게 유도
- **대화 히스토리 관리**: `RunnableWithMessageHistory`를 활용한 세션별 대화 히스토리 관리

## 카테고리

1. **국민연금**: 국민연금 가입, 납부, 수급 관련
2. **전입신고**: 전입신고, 주소변경, 주민등록 관련
3. **토지-건축물**: 토지, 건축물, 부동산 등기 관련
4. **청년월세**: 청년월세 지원금, 청년주거급여 관련
5. **주거급여**: 주거급여 신청, 자격, 절차 관련
6. **etc**: 위 카테고리와 전혀 관련 없는 일반 대화

## 기술 스택

- FastAPI
- LangChain
- AWS Bedrock
- Pydantic

## 개발

테스트 파일은 `tests/` 폴더의 Jupyter 노트북을 참고하세요.

## 웹
http://127.0.0.1:8000
http://localhost:8000/docs#/
http://127.0.0.1:8000/online


