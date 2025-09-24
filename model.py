# AI-model/model.py 
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
BANK_TEMPLATES = {
    'kbank': 'IQ500/template/kbank_template.csv',
    'scb': 'IQ500/template/scb_template.csv',
    'gsb': 'IQ500/template/gsb_template.csv',
    'ktb': 'IQ500/template/ktb_template.csv'
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

def ocr_with_auto_template(image_path: str) -> Tuple[Dict, Optional[str]]:
    """
    OCR โดยตรวจหาธนาคารอัตโนมัติจาก sender_bank field และใช้ template ที่เหมาะสม
    """
    # 1. ตรวจหาธนาคารจาก sender_bank fields
    detected_bank, confidence = detect_bank_from_sender_field(image_path)
    
    print(f"Bank detection result: {detected_bank} (confidence: {confidence})")
    
    if not detected_bank:
        return {}, None
    
    # 2. ใช้ template ของธนาคารที่ตรวจพบ
    template_path = get_bank_template_path(detected_bank)
    if not template_path:
        return {}, detected_bank
    
    # 3. ทำ OCR ด้วย template ที่เหมาะสม
    extracted_data = ocr_with_template(image_path, template_path)
    
    return extracted_data, detected_bank

def ocr_with_template(image_path, template_csv_path):
    """
    OCR ด้วย template ที่กำหนด (ฟังก์ชันเดิม)
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

# ====================== Fallback OCR ======================
def ocr_fallback_all_banks(image_path: str) -> Dict:
    """
    ถ้าตรวจหาธนาคารไม่ได้ ลองใช้ template ทุกธนาคารแล้วเลือกผลที่ดีที่สุด
    """
    best_result = {}
    best_score = 0
    best_bank = None
    
    for bank_code, template_path in BANK_TEMPLATES.items():
        if not os.path.exists(template_path):
            continue
            
        try:
            result = ocr_with_template(image_path, template_path)
            
            # คำนวณคะแนนจากจำนวน field ที่มีข้อมูล
            score = sum(1 for v in result.values() if v and str(v).strip())
            
            if score > best_score:
                best_score = score
                best_result = result
                best_bank = bank_code
                
        except Exception as e:
            print(f"Error with {bank_code} template: {str(e)}")
            continue
    
    return best_result, best_bank