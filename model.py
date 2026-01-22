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
import pandas as pd
from pyzbar.pyzbar import decode
from typing import Optional, Dict, Tuple

# ====================== Bank Detection Configuration ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BANK_TEMPLATES = {
    'kbank': os.path.join(BASE_DIR, 'template', 'kbank_template.csv'),
    'scb': os.path.join(BASE_DIR, 'template', 'scb_template.csv'),
    'gsb': os.path.join(BASE_DIR, 'template', 'gsb_template.csv'),
    'ktb': os.path.join(BASE_DIR, 'template', 'ktb_template.csv')
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
    ]
}

# ====================== QR-based Bank Detection ======================
# bank code (ตัวเลขจาก QR)
QR_BANK_CODE_MAP = {
    "004": "kbank",  # ธนาคารกสิกรไทย
    "014": "scb",    # ธนาคารไทยพาณิชย์
    "006": "ktb",    # ธนาคารกรุงไทย
    "030": "gsb",    # ธนาคารออมสิน
    "002": "bbl",    # ธนาคารกรุงเทพ
    "025": "bay",    # ธนาคารกรุงศรีอยูธยา
}


def extract_qr_from_image(image_path: str) -> Optional[str]:
    """
    ดึง raw QR data จากภาพทั้งใบ
    """
    try:
        img = Image.open(image_path)
        decoded_objects = decode(img)
        if not decoded_objects:
            print("No QR code found in image")
            return None

        # ใช้ QR ตัวแรกที่เจอ
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
    """
    ดึง bank code 3 หลักจาก raw QR (ตำแหน่ง index 18-20)
    """
    if not qr_raw:
        return None

    if len(qr_raw) < 21:
        return None

    bank_code = qr_raw[18:21]  # index 18–20
    return bank_code


def detect_bank_from_qr(image_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    ตรวจหาธนาคารจาก QR Code ในภาพ

    Returns:
        detected_bank (str | None): โค้ดธนาคารภายในระบบ เช่น 'kbank', 'scb', 'gsb', 'ktb'
        numeric_bank_code (str | None): bank code ตัวเลขจาก QR เช่น '004', '014'
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

# ====================== Bank Detection Functions ======================
def extract_sender_bank_field(image_path: str, template_csv_path: str) -> Optional[str]:
    """
    แยกเฉพาะ field sender_bank จาก template ที่กำหนด
    """
    reader = easyocr.Reader(['th', 'en'])
    img = Image.open(image_path)
    df = pd.read_csv(template_csv_path)
    
    try:
        # หา sender_bank field ใน template
        sender_bank_row = df[df['region_attributes'].str.contains('"name":"sender_bank"', na=False)]
        
        if sender_bank_row.empty:
            return None
            
        # ดึงพิกัดของ sender_bank
        row = sender_bank_row.iloc[0]
        shape_attrs = json.loads(row['region_shape_attributes'].replace('""', '"'))
        x, y = shape_attrs['x'], shape_attrs['y']
        width, height = shape_attrs['width'], shape_attrs['height']
        
        # Crop และ OCR เฉพาะ sender_bank area
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
            # ดึง sender_bank text จาก template นี้
            sender_text = extract_sender_bank_field(image_path, template_path)
            
            if not sender_text:
                bank_scores[bank_code] = 0
                continue
                
            # คำนวณคะแนนจากการตรงกับ keywords
            score = calculate_bank_score(sender_text.lower(), bank_code)
            bank_scores[bank_code] = score
            
            print(f"Bank: {bank_code}, Sender text: '{sender_text}', Score: {score}")
            
        except Exception as e:
            print(f"Error processing {bank_code}: {str(e)}")
            bank_scores[bank_code] = 0
    
    # เลือกธนาคารที่ได้คะแนนสูงสุด
    if not bank_scores:
        return None, 0.0
        
    best_bank = max(bank_scores, key=bank_scores.get)
    best_score = bank_scores[best_bank]
    
    # กำหนด threshold สำหรับการยอมรับ
    if best_score >= 0.5:  # ต้องตรงกับ keyword อย่างน้อย 50%
        return best_bank, best_score
    
    return None, best_score

def calculate_bank_score(sender_text: str, bank_code: str) -> float:
    """
    คำนวณคะแนนความตรงกับธนาคารจาก sender_bank text
    """
    if not sender_text:
        return 0.0
        
    keywords = BANK_KEYWORDS.get(bank_code, [])
    if not keywords:
        return 0.0
    
    matches = 0
    for keyword in keywords:
        if keyword.lower() in sender_text:
            matches += 1
    
    # คะแนน = (จำนวน keyword ที่ตรง / จำนวน keyword ทั้งหมด) + bonus ถ้าตรงกับ keyword หลัก
    base_score = matches / len(keywords)
    
    # ให้ bonus กับ keyword หลัก (keyword แรก)
    main_keyword = keywords[0].lower()
    if main_keyword in sender_text:
        base_score += 0.3
    
    return min(base_score, 1.0)  # จำกัดไม่เกิน 1.0

def get_bank_template_path(bank_code: str) -> Optional[str]:
    """
    รับ path ของ template ตามรหัสธนาคาร
    """
    template_path = BANK_TEMPLATES.get(bank_code)
    if template_path and os.path.exists(template_path):
        return template_path
    return None

def ocr_with_auto_template(image_path: str) -> Tuple[Dict, Optional[str], Optional[str]]:
    """
    OCR โดยตรวจหาธนาคารจาก QR Code เท่านั้น และใช้ template ที่เหมาะสม
    
    Returns:
        (extracted_data, detected_bank, error_message)
        - extracted_data: ข้อมูล OCR ที่ได้
        - detected_bank: โค้ดธนาคารที่ตรวจพบ
        - error_message: ข้อความ error ถ้ามี (None ถ้าสำเร็จ)
    """
    # 1. ตรวจหาธนาคารจาก QR Code เท่านั้น
    detected_bank, numeric_bank_code = detect_bank_from_qr(image_path)
    
    print(f"Bank detection from QR: {detected_bank} (numeric code: {numeric_bank_code})")
    
    # 2. ถ้าไม่เจอ QR Code หรืออ่าน QR Code ไม่ได้
    if not detected_bank:
        if numeric_bank_code:
            # มี QR code แต่ bank code ไม่รู้จัก
            error_msg = f"ไม่รู้จักธนาคารนี้ (Bank Code: {numeric_bank_code})"
            print(f"Error: {error_msg}")
            return {}, None, error_msg
        else:
            # ไม่พบ QR Code ในภาพ
            error_msg = "ไม่พบ QR Code ในภาพสลิป"
            print(f"Error: {error_msg}")
            return {}, None, error_msg
    
    # 3. ตรวจสอบว่ามี template สำหรับธนาคารนี้หรือไม่
    template_path = get_bank_template_path(detected_bank)
    if not template_path:
        error_msg = f"ไม่สามารถถอดข้อความได้เนื่องจากไม่สามารถระบุเทมเพลตธนาคารที่ใช้ได้ (ธนาคาร: {detected_bank})"
        print(f"Error: {error_msg}")
        return {}, detected_bank, error_msg
    
    # 4. ทำ OCR ด้วย template ที่เหมาะสม
    print(f"Using template: {template_path} for bank: {detected_bank}")
    extracted_data = ocr_with_template(image_path, template_path)
    
    return extracted_data, detected_bank, None

def ocr_with_template(image_path, template_csv_path):
    """
    OCR ด้วย template ที่กำหนด
    """
    reader = easyocr.Reader(['th', 'en'])
    img = Image.open(image_path)
    df = pd.read_csv(template_csv_path)
    extracted_data = {}

    for _, row in df.iterrows():
        try:
            shape_attrs = json.loads(row['region_shape_attributes'].replace('""', '"'))
            x, y = shape_attrs['x'], shape_attrs['y']
            width, height = shape_attrs['width'], shape_attrs['height']
            region_attrs = json.loads(row['region_attributes'].replace('""', '"'))
            field_name = region_attrs['name']
            
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
    
    Args:
        ocr_result: ข้อมูล OCR ที่ได้จาก ocr_with_template
        bank_detected: ธนาคารที่ตรวจพบ (optional)
    
    Returns:
        Dict ที่มีโครงสร้างข้อมูลที่ประมวลผลแล้ว
    """
    def clean_text(text: str) -> str:
        """ทำความสะอาดข้อความ"""
        if not text or text.strip() == "":
            return ""
        # ลบ whitespace ที่ไม่จำเป็น
        text = " ".join(text.split())
        return text.strip()
    
    def clean_amount(amount_str: str) -> Optional[float]:
        """แปลงจำนวนเงินเป็น float"""
        if not amount_str or amount_str.strip() == "":
            return None
        try:
            # ลบ comma และ whitespace
            cleaned = amount_str.replace(",", "").replace(" ", "").strip()
            # ลบตัวอักษรที่ไม่ใช่ตัวเลขและจุดทศนิยม
            cleaned = re.sub(r'[^\d.]', '', cleaned)
            if cleaned:
                return float(cleaned)
        except (ValueError, AttributeError):
            pass
        return None
    
    def clean_account(account_str: str) -> str:
        """ทำความสะอาดเลขบัญชี"""
        if not account_str or account_str.strip() == "":
            return ""
        # ลบ whitespace และตัวอักษรพิเศษที่ไม่ใช่ตัวเลขและ x
        cleaned = re.sub(r'[^\dxX\-]', '', account_str)
        return cleaned.strip()
    
    def clean_name(name_str: str) -> str:
        """ทำความสะอาดชื่อ"""
        if not name_str or name_str.strip() == "":
            return ""
        # ลบ whitespace ที่ไม่จำเป็น
        cleaned = " ".join(name_str.split())
        # ลบตัวอักษรพิเศษที่อาจเป็น noise
        cleaned = re.sub(r'[^\w\sก-๙\.]', '', cleaned, flags=re.UNICODE)
        return cleaned.strip()
    
    def clean_date(date_str: str) -> str:
        """ทำความสะอาดวันที่"""
        if not date_str or date_str.strip() == "":
            return ""
        return clean_text(date_str)
    
    def clean_time(time_str: str) -> str:
        """ทำความสะอาดเวลา"""
        if not time_str or time_str.strip() == "":
            return ""
        # ลบตัวอักษรที่ไม่ใช่ตัวเลขและ :
        cleaned = re.sub(r'[^\d:]', '', time_str)
        return cleaned.strip()
    
    def clean_reference(ref_str: str) -> str:
        """ทำความสะอาด reference number"""
        if not ref_str or ref_str.strip() == "":
            return ""
        # ลบ whitespace และตัวอักษรพิเศษที่ไม่จำเป็น
        cleaned = re.sub(r'[^\w\|\-]', '', ref_str)
        return cleaned.strip()
    
    # สร้าง structured data
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
    
    # เพิ่ม bank_detected ถ้ามี
    if bank_detected:
        ai_result["detected_bank"] = bank_detected
    
    return ai_result


