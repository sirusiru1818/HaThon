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

# 카테고리와 폴더 매핑 (talk_to_fill.py와 동일)
CATEGORY_FOLDER_MAP = {
    "국민연금": "1_Welfare",
    "전입신고": "2_Report", 
    "토지-건축물": "3_Land",
    "청년월세": "4_Monthly",
    "주거급여": "5_Salary"
}


class PdfGenerator:
    """PDF 템플릿에 텍스트를 오버레이하여 새 PDF를 생성하는 클래스"""
    
    def __init__(self, template_path, font_path):
        self.template_path = template_path
        self.font_name = 'CustomFont'
        
        # 1. 폰트 파일 찾기 (여러 경로 시도)
        font_paths_to_try = [
            font_path,  # 지정된 경로
            '/System/Library/Fonts/Supplemental/AppleGothic.ttf',  # macOS 시스템 폰트
            '/Library/Fonts/AppleGothic.ttf',  # macOS 폰트
            '/System/Library/Fonts/AppleSDGothicNeo.ttc',  # macOS 최신 폰트
            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',  # Linux 폰트
        ]
        
        actual_font_path = None
        for try_path in font_paths_to_try:
            if os.path.exists(try_path):
                actual_font_path = try_path
                print(f"[PDF] 폰트 발견: {try_path}")
                break
        
        if not actual_font_path:
            print(f"[PDF] ⚠️ 한글 폰트를 찾을 수 없습니다. 기본 폰트(Helvetica)를 사용합니다.")
            print(f"[PDF]    시도한 경로: {font_paths_to_try}")
            self.font_name = 'Helvetica'  # 기본 폰트 사용 (한글 깨짐 가능)
            return
        
        # 2. 폰트 등록 (이미 등록돼있으면 건너뛰기)
        try:
            pdfmetrics.getFont(self.font_name)
            print(f"[PDF] 폰트 '{self.font_name}'은 이미 등록되어 있습니다.")
        except KeyError:
            try:
                pdfmetrics.registerFont(TTFont(self.font_name, actual_font_path))
                print(f"[PDF] 폰트 '{self.font_name}' 등록 완료: {actual_font_path}")
            except Exception as e:
                print(f"[PDF] ❌ 폰트 등록 실패: {e}")
                self.font_name = 'Helvetica'  # 폴백

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
        self.docs_dir = os.path.join(base_dir, 'docs')
        
        # 폰트 경로 (고정)
        self.font_path = os.path.join(base_dir, 'assets', 'fonts', 'NanumGothic-Regular.ttf')

    def find_document_files(self, category_folder, document_name):
        """
        카테고리 폴더에서 문서명에 해당하는 파일들을 찾습니다.
        
        Args:
            category_folder: "4_Monthly" 같은 카테고리 폴더명
            document_name: "위임장", "대리수령" 같은 문서명
            
        Returns:
            dict: {
                "template": "템플릿 PDF 경로",
                "coords": "좌표 JSON 경로",
                "invert_y": True/False
            }
        """
        category_path = os.path.join(self.docs_dir, category_folder)
        
        if not os.path.exists(category_path):
            raise FileNotFoundError(f"카테고리 폴더가 없습니다: {category_path}")
        
        # 하위 폴더 탐색
        for root, dirs, files in os.walk(category_path):
            folder_name = os.path.basename(root)
            
            # 문서명과 일치하는 폴더 찾기
            if folder_name == document_name:
                template_path = None
                coords_path = None
                
                for file in files:
                    if file.endswith('.pdf') and '_좌표' not in file:
                        template_path = os.path.join(root, file)
                    elif file.endswith('_좌표.json'):
                        coords_path = os.path.join(root, file)
                
                if template_path and coords_path:
                    return {
                        "template": template_path,
                        "coords": coords_path,
                        "invert_y": True  # 기본값
                    }
        
        raise FileNotFoundError(f"'{document_name}' 문서의 템플릿 또는 좌표 파일을 찾을 수 없습니다.")
    
    def process_request(self, category_folder, document_name, user_data, output_filename, debug=False):
        """
        문서 타입에 맞는 PDF를 생성합니다.
        
        Args:
            category_folder: "4_Monthly" 같은 카테고리 폴더명
            document_name: "위임장", "대리수령" 같은 문서명
            user_data: 채울 데이터 { "delegator.name": "홍길동"... }
            output_filename: 저장할 파일명 (전체 경로)
            debug: 디버그 모드 (빨간 박스 표시)
            
        Returns:
            생성된 PDF 파일 경로
        """
        print(f"[PDF_MANAGER] PDF 생성 시작")
        print(f"[PDF_MANAGER]   - 카테고리: {category_folder}")
        print(f"[PDF_MANAGER]   - 문서명: {document_name}")
        
        # 1. 문서 파일 찾기
        doc_info = self.find_document_files(category_folder, document_name)
        print(f"[PDF_MANAGER]   - 템플릿: {doc_info['template']}")
        print(f"[PDF_MANAGER]   - 좌표: {doc_info['coords']}")
        
        # 2. 좌표 파일 로딩
        with open(doc_info['coords'], 'r', encoding='utf-8') as f:
            coords_data = json.load(f)
        print(f"[PDF_MANAGER]   - 좌표 필드 수: {len(coords_data)}")
        
        # 3. Generator 초기화 및 PDF 생성
        generator = PdfGenerator(doc_info['template'], self.font_path)
        
        result_path = generator.create_pdf(
            user_data, 
            coords_data, 
            output_filename, 
            debug, 
            invert_y=doc_info['invert_y']
        )
        
        print(f"[PDF_MANAGER] ✅ PDF 생성 완료: {result_path}")
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
    os.makedirs(output_dir, exist_ok=True)

    # --- 시나리오 1: 청년 월세 위임장 ---
    print("\n[Case 1] 사용자가 '청년월세 위임장'을 원함")
    
    category_1 = "4_Monthly"
    doc_name_1 = "위임장"
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
        manager.process_request(category_1, doc_name_1, data_1, filename, debug=True)
        print(f" -> ✅ 성공: {filename}")
    except Exception as e:
        print(f" -> ❌ 실패: {e}")
        import traceback
        traceback.print_exc()


    # --- 시나리오 2: 대리수령 신청서 ---
    print("\n[Case 2] 사용자가 '대리수령 신청서'를 원함")
    
    category_2 = "4_Monthly"
    doc_name_2 = "대리수령"
    data_2 = {
        "recipient.name": "홍길동",
        "recipient.birthdate": "1950-01-01",
        "recipient.gender": "남",
        "recipient.number": "02-123-4567",
        "recipient.mobile": "010-1234-5678",
        "recipient.address": "서울시 강남구 테헤란로 123, 101동 101호",

        "application_reason.guardianship_no_account": "",
        "application_reason.seized_claim": "",
        "application_reason.dementia_or_immobility": "V",

        "receive_period.start_year": "2024",
        "receive_period.start_month": "01",
        "receive_period.end_year": "2024",
        "receive_period.end_month": "02",
        "receive_period.total_months": "1",

        "guardian.name": "",
        "guardian.birthdate": "",
        "guardian.number": "",
        "guardian.mobile": "",
        "guardian.address": "",

        "representative_recipient.name": "홍길순",
        "representative_recipient.birthdate": "1980-05-05",
        "representative_recipient.phone": "02-987-6543",
        "representative_recipient.number": "010-9876-5432",
        "representative_recipient.address": "경기도 성남시 분당구 판교로 456",
        "representative_recipient.relationship_to_recipient": "자녀",

        "bank_account.bank_name": "KB국민은행",
        "bank_account.account_number": "123-456-78-901234"
    }
    
    try:
        filename = os.path.join(output_dir, "result_대리수령.pdf")
        manager.process_request(category_2, doc_name_2, data_2, filename, debug=True)
        print(f" -> ✅ 성공: {filename}")
    except Exception as e:
        print(f" -> ❌ 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
