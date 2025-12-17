import boto3
import os

# 🚨 환경 변수에서 AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION을 읽어옵니다.

def get_bedrock_client():
    """
    Amazon Bedrock 런타임 클라이언트 (임베딩에 사용)를 반환합니다.
    """
    try:
        # Bedrock 클라이언트는 Titan Embeddings 호출에 사용됩니다.
        bedrock_client = boto3.client(
            service_name='bedrock-runtime'
            # region_name은 환경 변수에서 자동 로드됨
        )
        return bedrock_client
    except Exception as e:
        print(f"AWS Bedrock 클라이언트 생성 오류: {e}")
        return None

def get_transcribe_client():
    """
    Amazon Transcribe 서비스 클라이언트를 반환합니다.
    """
    try:
        # Transcribe 클라이언트 생성
        transcribe_client = boto3.client(
            service_name='transcribe'
            # region_name은 환경 변수에서 자동 로드됨
        )
        return transcribe_client
    except Exception as e:
        print(f"AWS Transcribe 클라이언트 생성 오류: {e}")
        return None

if __name__ == '__main__':
    if get_bedrock_client() and get_transcribe_client():
        print("AWS 클라이언트(Bedrock, Transcribe) 생성 성공.")