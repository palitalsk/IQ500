import io
import json
import os
import re
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import easyocr
from pyzbar.pyzbar import decode
from typing import Optional, Dict, Tuple, List

# ====================== Bank Detection Configuration ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# NOTE: Template files ใช้สำหรับกำหนดตำแหน่งของข้อมูลในสลิปแต่ละธนาคาร
# TODO: ถ้าต้องการเพิ่มธนาคารใหม่ ให้สร้าง template file แล้วเพิ่มที่นี่
# Template file ต้องอยู่ในโฟลเดอร์ template/ และใช้รูปแบบ JSON (VIA annotation format)
BANK_TEMPLATES = {
    'kbank': os.path.join(BASE_DIR, 'template', 'kbank_template.json'),
    'scb': os.path.join(BASE_DIR, 'template', 'scb_template.json'),
    'gsb': os.path.join(BASE_DIR, 'template', 'gsb_template.json'),
    'ktb': os.path.join(BASE_DIR, 'template', 'ktb_template.json'),
    'bbl': os.path.join(BASE_DIR, 'template', 'bbl_template.json'),
    'uob': os.path.join(BASE_DIR, 'template', 'uob_template.json'),
    'bay': os.path.join(BASE_DIR, 'template', 'bay_template.json')
}

BANK_KEYWORDS = {
    'kbank': [
        'กสิกรไทย', 'กสิกร', 'kbank', 'kasikorn', 'ธ.กสิกรไทย',
        'kasikornbank', 'กสิกรไทย จำกัด'
    ],
    'scb': [
        'ไทยพาณิชย์', 'SCB','scb', 'siam commercial', 'ธนาคารไทยพาณิชย์',
        'พาณิชย์', 'siamcommercial'
    ],
    'gsb': [
        'ออมสิน', 'gsb', 'government savings', 'ธนาคารออมสิน',
        'กอส', 'savings bank', 'mymemo'
    ],
    'ktb': [
        'กรุงไทย', 'ktb', 'krung thai', 'ธนาคารกรุงไทย',
        'krungthai', 'กรุงไทย จำกัด'
    ],
    'bbl': [
        'กรุงเทพ', 'bbl', 'bangkok bank', 'ธนาคารกรุงเทพ',
        'bangkokbank', 'กรุงเทพ จำกัด'
    ],
    'uob': [
        'ยูโอบี', 'uob', 'united overseas bank', 'ธนาคารยูโอบี',
        'uobthailand', 'ยูโอบี ไทย'
    ],
    'bay': [
        'กรุงศรี', 'bay', 'krungsri', 'ธนาคารกรุงศรีอยุธยา',
        'krungsri bank', 'กรุงศรีอยุธยา'
    ]
}

# ====================== QR-based Bank Detection ======================
# NOTE: QR Bank Code คือตัวเลข 3 หลักที่อยู่ใน QR Code ของสลิป (ตำแหน่ง index 18-20)
# อ้างอิง: รายการ Bank Code ทั้งหมดสามารถดูได้ที่
# https://www.bot.or.th/Thai/FinancialInstitutions/Standard/Pages/QRCode.aspx
# TODO: ถ้าต้องการเพิ่มธนาคารใหม่ ให้ดู Bank Code จากลิงก์ด้านบนแล้วเพิ่ม mapping ตรงนี้
QR_BANK_CODE_MAP = {
    "004": "kbank",  # ธนาคารกสิกรไทย
    "014": "scb",    # ธนาคารไทยพาณิชย์
    "006": "ktb",    # ธนาคารกรุงไทย
    "030": "gsb",    # ธนาคารออมสิน
    "002": "bbl",    # ธนาคารกรุงเทพ
    "025": "bay",    # ธนาคารกรุงศรีอยุธยา
    "024": "uob",    # ธนาคารยูโอบี
}


def extract_qr_from_image(image_path: str) -> Optional[str]:
    """ดึง raw QR data จากภาพทั้งใบ"""
    try:
        img = Image.open(image_path)
        decoded_objects = decode(img)
        if not decoded_objects:
            print("No QR code found in image")
            return None

        raw_qr = decoded_objects[0].data.decode("utf-8")
        print(f"QR code extracted successfully, length: {len(raw_qr)}")
        return raw_qr
    except NameError as e:
        print(f"pyzbar decode not available: {str(e)}")
        return None
    except Exception as e:
        print(f"Error extracting QR from image: {str(e)}")
        return None


def extract_bank_code_from_qr(qr_raw: str) -> Optional[str]:
    """ดึง bank code 3 หลักจาก raw QR (ตำแหน่ง index 18-20)"""
    if not qr_raw:
        return None
    if len(qr_raw) < 21:
        return None
    bank_code = qr_raw[18:21]
    return bank_code


def detect_bank_from_qr(image_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    ตรวจหาธนาคารจาก QR Code ในภาพ
    
    Returns:
        detected_bank: โค้ดธนาคารภายในระบบ เช่น 'kbank', 'scb'
        numeric_bank_code: bank code ตัวเลขจาก QR เช่น '004', '014'
    """
    qr_raw = extract_qr_from_image(image_path)
    if not qr_raw:
        return None, None

    numeric_bank_code = extract_bank_code_from_qr(qr_raw)
    if not numeric_bank_code:
        return None, None

    detected_bank = QR_BANK_CODE_MAP.get(numeric_bank_code)
    return detected_bank, numeric_bank_code

# ====================== CNN Model ======================
class EnhancedSlipCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(EnhancedSlipCNN, self).__init__()
        from torchvision import models
        from torchvision.models import EfficientNet_B0_Weights
        self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.features[8:].parameters():
            param.requires_grad = True
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

def load_cnn_model(model_path, device='cpu'):
    model = EnhancedSlipCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_inference_transform(img_size=224):
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()
    ])
    return transform

def classify_image_with_cnn(image_path, model, transform, device):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed = transform(image=image)['image']
    input_tensor = transformed.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)
    return predicted_idx.item(), confidence.item()

# ====================== JSON Template Functions ======================
def load_template_from_json(template_json_path: str) -> List[Dict]:
    """
    โหลด template จากไฟล์ JSON และแปลงเป็น list of regions
    
    Returns:
        List[Dict]: รายการ regions พร้อมข้อมูล shape และ field name
    """
    try:
        with open(template_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # ดึง metadata ของรูปแรก (สมมติว่ามีรูปเดียวใน template)
        img_metadata = list(data.get('_via_img_metadata', {}).values())[0]
        regions = img_metadata.get('regions', [])
        
        # แปลง regions เป็น format ที่ใช้งานง่าย
        processed_regions = []
        for region in regions:
            shape_attrs = region.get('shape_attributes', {})
            region_attrs = region.get('region_attributes', {})
            
            # หา field name ที่ไม่ใช่ค่าว่าง
            field_name = None
            for key, value in region_attrs.items():
                if value and value.strip():
                    field_name = value
                    break
            
            if field_name and shape_attrs.get('name') == 'rect':
                processed_regions.append({
                    'field_name': field_name,
                    'x': shape_attrs.get('x', 0),
                    'y': shape_attrs.get('y', 0),
                    'width': shape_attrs.get('width', 0),
                    'height': shape_attrs.get('height', 0)
                })
        
        return processed_regions
        
    except Exception as e:
        print(f"Error loading template JSON: {str(e)}")
        return []


def extract_sender_bank_field(image_path: str, template_json_path: str) -> Optional[str]:
    """
    แยกเฉพาะ field sender_bank จาก template ที่กำหนด
    """
    reader = easyocr.Reader(['th', 'en'])
    img = Image.open(image_path)
    
    try:
        regions = load_template_from_json(template_json_path)
        
        # หา sender_bank field
        sender_bank_region = None
        for region in regions:
            if region['field_name'] == 'sender_bank':
                sender_bank_region = region
                break
        
        if not sender_bank_region:
            return None
        
        # Crop และ OCR เฉพาะ sender_bank area
        x = sender_bank_region['x']
        y = sender_bank_region['y']
        width = sender_bank_region['width']
        height = sender_bank_region['height']
        
        cropped_img = img.crop((x, y, x+width, y+height))
        img_bytes = io.BytesIO()
        cropped_img.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        result = reader.readtext(img_bytes, detail=0)
        text = ' '.join(result).strip()
        
        return text
        
    except Exception as e:
        print(f"Error extracting sender_bank: {str(e)}")
        return None


def detect_bank_from_sender_field(image_path: str) -> Tuple[Optional[str], float]:
    """
    ตรวจสอบธนาคารโดยการเช็ค sender_bank field จากทุก template
    """
    bank_scores = {}
    
    for bank_code, template_path in BANK_TEMPLATES.items():
        if not os.path.exists(template_path):
            continue
            
        try:
            sender_text = extract_sender_bank_field(image_path, template_path)
            
            if not sender_text:
                bank_scores[bank_code] = 0
                continue
                
            score = calculate_bank_score(sender_text.lower(), bank_code)
            bank_scores[bank_code] = score
            
            print(f"Bank: {bank_code}, Sender text: '{sender_text}', Score: {score}")
            
        except Exception as e:
            print(f"Error processing {bank_code}: {str(e)}")
            bank_scores[bank_code] = 0
    
    if not bank_scores:
        return None, 0.0
        
    best_bank = max(bank_scores, key=bank_scores.get)
    best_score = bank_scores[best_bank]
    
    if best_score >= 0.5:
        return best_bank, best_score
    
    return None, best_score


def calculate_bank_score(sender_text: str, bank_code: str) -> float:
    """คำนวณคะแนนความตรงกับธนาคารจาก sender_bank text"""
    if not sender_text:
        return 0.0
        
    keywords = BANK_KEYWORDS.get(bank_code, [])
    if not keywords:
        return 0.0
    
    matches = 0
    for keyword in keywords:
        if keyword.lower() in sender_text:
            matches += 1
    
    base_score = matches / len(keywords)
    
    # ให้ bonus กับ keyword หลัก
    main_keyword = keywords[0].lower()
    if main_keyword in sender_text:
        base_score += 0.3
    
    return min(base_score, 1.0)


def get_bank_template_path(bank_code: str) -> Optional[str]:
    """รับ path ของ template ตามรหัสธนาคาร"""
    template_path = BANK_TEMPLATES.get(bank_code)
    if template_path and os.path.exists(template_path):
        return template_path
    return None


def ocr_with_auto_template(image_path: str) -> Tuple[Dict, Optional[str], Optional[str]]:
    """
    OCR โดยตรวจหาธนาคารจาก QR Code เท่านั้น และใช้ template ที่เหมาะสม
    
    Returns:
        (extracted_data, detected_bank, error_message)
    """
    # ตรวจหาธนาคารจาก QR Code
    detected_bank, numeric_bank_code = detect_bank_from_qr(image_path)
    
    print(f"Bank detection from QR: {detected_bank} (numeric code: {numeric_bank_code})")
    
    if not detected_bank:
        if numeric_bank_code:
            error_msg = f"ไม่รู้จักธนาคารนี้ (Bank Code: {numeric_bank_code})"
            print(f"Error: {error_msg}")
            return {}, None, error_msg
        else:
            error_msg = "ไม่พบ QR Code ในภาพสลิป"
            print(f"Error: {error_msg}")
            return {}, None, error_msg
    
    # ตรวจสอบว่ามี template
    template_path = get_bank_template_path(detected_bank)
    if not template_path:
        error_msg = f"ไม่สามารถถอดข้อความได้เนื่องจากไม่สามารถระบุเทมเพลตธนาคารที่ใช้ได้ (ธนาคาร: {detected_bank})"
        print(f"Error: {error_msg}")
        return {}, detected_bank, error_msg
    
    # ทำ OCR ด้วย template
    print(f"Using template: {template_path} for bank: {detected_bank}")
    extracted_data = ocr_with_template(image_path, template_path)
    
    return extracted_data, detected_bank, None


def ocr_with_template(image_path: str, template_json_path: str) -> Dict:
    """
    OCR ด้วย JSON template ที่กำหนด
    """
    reader = easyocr.Reader(['th', 'en'])
    img = Image.open(image_path)
    extracted_data = {}
    
    # โหลด regions จาก JSON template
    regions = load_template_from_json(template_json_path)
    
    for region in regions:
        try:
            field_name = region['field_name']
            x = region['x']
            y = region['y']
            width = region['width']
            height = region['height']
            
            # Crop image
            cropped_img = img.crop((x, y, x+width, y+height))
            
            # OCR
            img_bytes = io.BytesIO()
            cropped_img.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()
            result = reader.readtext(img_bytes, detail=0)
            text = ' '.join(result).strip()
            extracted_data[field_name] = text
            
            # QR Code detection
            if field_name == 'qr_code':
                qr_data = decode(cropped_img)
                if qr_data:
                    extracted_data['qr_code'] = qr_data[0].data.decode('utf-8')
                    
        except Exception as e:
            print(f"Error processing field {field_name}: {str(e)}")
            extracted_data[field_name] = ""
    
    return extracted_data

# ====================== AI Result Processing ======================
def process_ocr_to_ai_result(ocr_result: Dict, bank_detected: Optional[str] = None) -> Dict:
    """
    ประมวลผล OCR result ให้เป็น structured data สำหรับ ai_result
    """
    def clean_text(text: str) -> str:
        if not text or text.strip() == "":
            return ""
        text = " ".join(text.split())
        return text.strip()
    
    def clean_amount(amount_str: str) -> Optional[float]:
        if not amount_str or amount_str.strip() == "":
            return None
        try:
            cleaned = amount_str.replace(",", "").replace(" ", "").strip()
            cleaned = re.sub(r'[^\d.]', '', cleaned)
            if cleaned:
                return float(cleaned)
        except (ValueError, AttributeError):
            pass
        return None
    
    def clean_account(account_str: str) -> str:
        if not account_str or account_str.strip() == "":
            return ""
        cleaned = re.sub(r'[^\dxX\-]', '', account_str)
        return cleaned.strip()
    
    def clean_name(name_str: str) -> str:
        if not name_str or name_str.strip() == "":
            return ""
        cleaned = " ".join(name_str.split())
        cleaned = re.sub(r'[^\w\sก-๙\.]', '', cleaned, flags=re.UNICODE)
        return cleaned.strip()
    
    def clean_date(date_str: str) -> str:
        if not date_str or date_str.strip() == "":
            return ""
        return clean_text(date_str)
    
    def clean_time(time_str: str) -> str:
        if not time_str or time_str.strip() == "":
            return ""
        cleaned = re.sub(r'[^\d:]', '', time_str)
        return cleaned.strip()
    
    def clean_reference(ref_str: str) -> str:
        if not ref_str or ref_str.strip() == "":
            return ""
        cleaned = re.sub(r'[^\w\|\-]', '', ref_str)
        return cleaned.strip()
    
    ai_result = {
        "amount": clean_amount(ocr_result.get("amount", "")),
        "sender": {
            "bank": clean_text(ocr_result.get("sender_bank", "")),
            "account": clean_account(ocr_result.get("sender_account", "")),
            "name": clean_name(ocr_result.get("sender_name", ""))
        },
        "receiver": {
            "bank": clean_text(ocr_result.get("reciever_bank", "")),
            "account": clean_account(ocr_result.get("reciever_account", "")),
            "name": clean_name(ocr_result.get("reciever_name", ""))
        },
        "transfer_date": clean_date(ocr_result.get("transfer_date", "")),
        "transfer_time": clean_time(ocr_result.get("transfer_time", "")),
        "reference_number": clean_reference(ocr_result.get("reference_number", "")),
        "fee": clean_amount(ocr_result.get("fee", "")) if ocr_result.get("fee") else None,
        "qr_code": clean_text(ocr_result.get("qr_code", "")) if ocr_result.get("qr_code") else None
    }
    
    if bank_detected:
        ai_result["detected_bank"] = bank_detected
    
    return ai_result