#텍스트 임베딩 및 최종 출력

import json
import time
from aws_config import get_bedrock_client

def generate_vector(text: str) -> list:
    """
    Bedrock Titan Embeddings 모델을 사용하여 텍스트를 벡터로 변환합니다.
    """
    client = get_bedrock_client()
    if not client:
        return []

    # --- 1. Titan Embeddings 모델 호출 파라미터 구성 ---
    
    # 모델 ID (Titan Embeddings G1 - Text)
    model_id = "amazon.titan-embed-text-v1" 
    
    body = json.dumps({
        "inputText": text
    })
    
    try:
        response = client.invoke_model(
            contentType='application/json',
            accept='application/json',
            modelId=model_id,
            body=body
        )
        
        # --- 2. 응답 파싱 및 벡터 추출 ---
        response_body = json.loads(response.get('body').read())
        # 임베딩 벡터 (list 형태)
        embedding = response_body.get('embedding', []) 
        
        return embedding
    
    except Exception as e:
        print(f"Titan Embeddings API 호출 오류: {e}")
        return []

def generate_vector_and_query_json(minwon_text: str) -> dict:
    """
    텍스트를 임베딩하고 다음 팀에게 전달할 최종 JSON 쿼리 구조를 생성합니다.
    """
    print("🚀 텍스트 임베딩 시작...")
    
    # 1. 벡터 생성 (Step 4)
    query_vector = generate_vector(minwon_text)
    
    if not query_vector:
        return {"error": "임베딩 벡터 생성 실패", "text": minwon_text}

    print(f"✅ 임베딩 벡터 생성 성공 (차원: {len(query_vector)})")

    # 2. 최종 JSON 구조 정의 (Step 5)
    final_query_data = {
        "user_query_text": minwon_text,
        "user_query_vector": query_vector, # 다음 팀의 벡터 검색 입력
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "processing_status": "READY_FOR_LLM_CLASSIFICATION"
    }
    
    return final_query_data

if __name__ == '__main__':
    test_text = "인감증명서를 발급받으려면 어떤 서류를 준비해야 하나요?"
    final_output = generate_vector_and_query_json(test_text)
    print(json.dumps(final_output, indent=2, ensure_ascii=False))