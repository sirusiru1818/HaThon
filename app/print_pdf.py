"""
PDF 생성 모듈
- PdfGenerator: PDF 템플릿에 데이터를 채워넣는 클래스
- PdfManager: 문서 타입별 PDF 생성을 관리하는 클래스
"""

import io
import os
import json
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from pypdf import PdfReader, PdfWriter


class PdfGenerator:
    """PDF 템플릿에 텍스트를 오버레이하여 새 PDF를 생성하는 클래스"""
    
    def __init__(self, template_path, font_path):
        self.template_path = template_path
        self.font_name = 'CustomFont'
        
        # 1. 폰트 파일 존재 여부 확인
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"폰트 파일을 찾을 수 없습니다: {font_path}")
        
        # 2. 폰트 등록 (이미 등록돼있으면 건너뛰기)
        try:
            pdfmetrics.getFont(self.font_name)
        except:
            pdfmetrics.registerFont(TTFont(self.font_name, font_path))

    def _create_overlay(self, data, coords, debug=False, invert_y=True):
        """
        데이터와 좌표를 받아 투명한 PDF 레이어를 메모리에 생성합니다.
        
        Args:
            data: 채울 데이터 딕셔너리 {"field_name": "value"}
            coords: 좌표 딕셔너리 {"field_name": {"x": 100, "y": 200}}
            debug: True시 빨간 박스로 영역 표시
            invert_y: True시 y좌표 반전 (위에서부터 측정한 좌표인 경우)
        """
        packet = io.BytesIO()
        # A4 사이즈 캔버스 생성
        can = canvas.Canvas(packet, pagesize=A4)
        
        # 폰트 설정
        can.setFont(self.font_name, 11)

        # 페이지 높이 (좌표 반전 계산용, 약 842pt)
        page_height = A4[1]

        for key, value in data.items():
            if key in coords:
                x = coords[key]['x']
                raw_y = coords[key]['y']
                
                # [좌표 시스템 주의사항]
                # PDF는 왼쪽 아래가 (0,0)입니다.
                # 만약 가져오신 좌표가 '맨 위에서부터 잰 길이'라면 invert_y=True 사용
                
                # 옵션에 따라 계산 방식 변경
                if invert_y:
                    y = page_height - raw_y  # 뒤집기 (위 -> 아래 좌표계)
                else:
                    y = raw_y  # 그대로 사용

                # --- [디버그 모드] 빨간 박스 그리기 ---
                if debug:
                    can.saveState()
                    can.setStrokeColor(colors.red)
                    can.setLineWidth(1)
                    # 텍스트 영역 박스
                    can.rect(x, y, 150, 20, fill=0)
                    # 키 이름 표시 (작게)
                    can.setFont("Helvetica", 8)
                    can.setFillColor(colors.red)
                    can.drawString(x, y + 22, f"{key}")
                    can.restoreState()
                # ------------------------------------

                # 텍스트 쓰기 (None값 방지)
                text_val = str(value) if value is not None else ""
                
                # 좌표 (x, y) 위치에 글자 쓰기 (y+5는 박스 중앙 정렬을 위한 미세 조정)
                can.drawString(x, y + 5, text_val)
        
        can.save()
        packet.seek(0)
        return packet

    def create_pdf(self, data_json, coord_json, output_path, debug=False, invert_y=True):
        """
        원본 PDF와 텍스트 레이어를 합쳐서 저장합니다.
        
        Args:
            data_json: 채울 데이터
            coord_json: 좌표 정보
            output_path: 저장할 파일 경로
            debug: 디버그 모드 (빨간 박스 표시)
            invert_y: y좌표 반전 여부
            
        Returns:
            생성된 PDF 파일 경로
        """
        # 1. 텍스트 오버레이(투명 레이어) 생성
        overlay_packet = self._create_overlay(data_json, coord_json, debug, invert_y)
        new_pdf_layer = PdfReader(overlay_packet)
        
        # 2. 원본 템플릿 읽기
        existing_pdf = PdfReader(open(self.template_path, "rb"))
        output = PdfWriter()

        # 3. 페이지 합치기 (현재는 1페이지만 처리)
        # 템플릿의 첫 페이지 가져오기
        page = existing_pdf.pages[0]
        # 그 위에 텍스트 레이어 덮어쓰기
        page.merge_page(new_pdf_layer.pages[0])
        output.add_page(page)

        # 4. 저장 경로 폴더가 없으면 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 5. 파일 저장
        with open(output_path, "wb") as f:
            output.write(f)
        
        return output_path


class PdfManager:
    """문서 타입별 PDF 생성을 관리하는 클래스"""
    
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.assets_dir = os.path.join(base_dir, 'assets')
        
        # [핵심] 문서 등록부 (Registry)
        # 백엔드에서 보내주는 'doc_type' 키값과 실제 파일명을 연결합니다.
        self.doc_registry = {
            # "문서타입ID": { "template": "PDF파일명", "coords": "좌표JSON파일명" }
            
            # 청년 월세 위임장
            "youth_rent_apply": { 
                "template": "wiimjang.pdf", 
                "coords": "wiimjang.json",
                "invert_y": True
            },
            # 청년 월세 지원 신청서 (대리수령)
            "appeal_request": { 
                "template": "청년월세지원대리수령신청서.pdf", 
                "coords": "청년월세지원대리수령신청서.json",
                "invert_y": False
            }
            # 필요한 서류가 늘어나면 여기에 계속 추가하면 됩니다.
        }

    def process_request(self, doc_type, user_data, output_filename, debug=False):
        """
        문서 타입에 맞는 PDF를 생성합니다.
        
        Args:
            doc_type: "youth_rent_apply" 같은 문서 식별자
            user_data: 채울 데이터 { "name": "홍길동"... }
            output_filename: 저장할 파일명 (전체 경로)
            debug: 디버그 모드 (빨간 박스 표시)
            
        Returns:
            생성된 PDF 파일 경로
        """
        
        # 1. 등록된 문서인지 확인
        if doc_type not in self.doc_registry:
            raise ValueError(f"지원하지 않는 문서 타입입니다: {doc_type}")

        # 2. 파일 경로 설정
        doc_info = self.doc_registry[doc_type]
        
        template_path = os.path.join(self.assets_dir, 'templates', doc_info['template'])
        coords_path = os.path.join(self.assets_dir, 'coords', doc_info['coords'])

        # 설정값 가져오기
        invert_option = doc_info.get('invert_y', True)
        
        # 3. 좌표 파일 로딩
        if not os.path.exists(coords_path):
            raise FileNotFoundError(f"좌표 파일이 없습니다: {coords_path}")
            
        with open(coords_path, 'r', encoding='utf-8') as f:
            coords_data = json.load(f)

        # 4. Generator 호출 (기존에 만든 클래스 사용)
        # 폰트 경로는 공통이므로 여기서 지정
        font_path = os.path.join(self.assets_dir, 'fonts', 'NanumGothic-Regular.ttf')
        
        generator = PdfGenerator(template_path, font_path)
        
        # 5. PDF 생성
        result_path = generator.create_pdf(user_data, coords_data, output_filename, debug, invert_y=invert_option)
        return result_path


# ===== 테스트 코드 =====
def main():
    """PDF 생성 테스트"""
    print("--- 멀티 서류 처리 시스템 테스트 ---")
    
    # 프로젝트 루트 경로
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Manager 초기화
    manager = PdfManager(project_root)
    output_dir = os.path.join(project_root, 'output')

    # --- 시나리오 1: 청년 월세 위임장 ---
    print("\n[Case 1] 사용자가 '청년월세 위임장'을 원함")
    
    req_type_1 = "youth_rent_apply"
    data_1 = {
        # 위임하는 사람
        "delegator.name": "홍길동",
        "delegator.birthdate": "1990-01-01",
        "delegator.address": "서울특별시 강남구 테헤란로 123",
        "delegator.number": "010-1234-5678",

        # 위임받는 사람
        "delegate.name": "김철수",
        "delegate.birthdate": "1995-05-12",
        "delegate.relationship_to_delegator": "친구",
        "delegate.number": "010-9876-5432",
        "delegate.address": "서울특별시 마포구 월드컵북로 45",

        # 민원내용 체크박스
        "civil_service_items.rent_request": "V",
        "civil_service_items.appeal_request": "",
        "civil_service_items.certificate_issuance": ""
    }
    
    try:
        filename = os.path.join(output_dir, "result_위임장.pdf")
        # debug=True 로 설정하면 빨간 박스가 보입니다.
        manager.process_request(req_type_1, data_1, filename, debug=True)
        print(f" -> ✅ 성공: {filename}")
    except Exception as e:
        print(f" -> ❌ 실패: {e}")


    # --- 시나리오 2: 대리수령 신청서 ---
    print("\n[Case 2] 사용자가 '대리수령 신청서'를 원함")
    
    req_type_2 = "appeal_request"
    data_2 = {
        "recipient.name": "홍길동",
        "recipient.birthdate": "1958-03-12",
        "recipient.gender": "남",

        "recipient.phone": "02-345-6789",
        "recipient.mobile": "010-1234-5678",
        "recipient.address": "서울특별시 강남구 테헤란로 123",

        "application_reason.guardianship_no_account": "해당",
        "application_reason.seized_claim": "",
        "application_reason.dementia_or_immobility": "",

        "receive_period.start_year": "2024",
        "receive_period.start_month": "01",
        "receive_period.end_year": "2024",
        "receive_period.end_month": "12",
        "receive_period.total_months": "12",

        "guardian.name": "김철수",
        "guardian.birthdate": "1985-07-21",
        "guardian.phone": "02-987-6543",
        "guardian.mobile": "010-9876-5432",
        "guardian.address": "서울특별시 서초구 서초대로 45",

        "proxy_receiver.name": "이영희",
        "proxy_receiver.birthdate": "1990-11-05",
        "proxy_receiver.phone": "031-222-3333",
        "proxy_receiver.mobile": "010-2222-3333",
        "proxy_receiver.address": "경기도 성남시 분당구 판교로 88",
        "proxy_receiver.relationship_to_recipient": "자녀",

        "bank_account.bank_name": "국민은행",
        "bank_account.account_number": "123456-01-987654",

        "signature.date.year": "2024",
        "signature.date.month": "09",
        "signature.date.day": "30",
        "signature.name": "홍길동"
    }
    
    try:
        filename = os.path.join(output_dir, "result_대리수령.pdf")
        manager.process_request(req_type_2, data_2, filename, debug=True)
        print(f" -> ✅ 성공: {filename}")
    except Exception as e:
        print(f" -> ❌ 실패: {e}")


if __name__ == "__main__":
    main()
