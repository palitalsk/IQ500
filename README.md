# IQ500 - Slip Detection & OCR Service

บริการตรวจสอบสลิปและถอดข้อความจากสลิปธนาคารด้วย AI

## 📋 สารบัญ

- [ความต้องการของระบบ](#ความต้องการของระบบ)
- [การติดตั้ง](#การติดตั้ง)
- [การรัน](#การรัน)
- [API Endpoints](#api-endpoints)
- [การตั้งค่าธนาคาร](#การตั้งค่าธนาคาร)
- [การเพิ่มธนาคารใหม่](#การเพิ่มธนาคารใหม่)
- [Troubleshooting](#troubleshooting)

## 🔧 ความต้องการของระบบ

- Python 3.8 หรือสูงกว่า
- pip
- ไฟล์โมเดล CNN (`best_cnn_model.pth`)

## 📦 การติดตั้ง

### 1. Clone repository

```bash
git clone <repository-url>
cd IQ500
```

### 2. สร้าง virtual environment (แนะนำ)

```bash
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. ติดตั้ง dependencies

```bash
pip install -r requirements.txt
```

### 4. เตรียมไฟล์โมเดล

วางไฟล์โมเดล `best_cnn_model.pth` ไว้ที่:
```
IQ500/models/best_cnn_model.pth
```

**หมายเหตุ:** ไฟล์โมเดลนี้ใช้สำหรับตรวจสอบว่าภาพที่ส่งมาเป็นสลิปจริงหรือไม่

## 🚀 การรัน

### รัน API Server

```bash
python api.py
```

API จะรันที่ `http://0.0.0.0:5555`

### ตรวจสอบว่า API ทำงาน

```bash
curl http://localhost:5555/predict-slip
```

## 📡 API Endpoints

### POST `/predict-slip`

ตรวจสอบว่าภาพเป็นสลิปหรือไม่ และถอดข้อความจากสลิป

**Request:**
```json
{
  "image": "<base64 string หรือ URL ของรูปภาพ>"
}
```

**Response ตัวอย่าง (สำเร็จ):**
```json
{
  "is_slip": true,
  "confidence": 0.98,
  "bank_detected": "kbank",
  "ocr_result": {
    "sender_name": "นายทดสอบ ระบบ",
    "sender_account": "1234567890",
    "amount": "1000.00",
    ...
  },
  "ai_result": {
    "amount": 1000.0,
    "sender": {
      "bank": "กสิกรไทย",
      "account": "1234567890",
      "name": "นายทดสอบ ระบบ"
    },
    ...
  },
  "detection_method": "qr_code"
}
```

**Response ตัวอย่าง (ไม่ใช่สลิป):**
```json
{
  "is_slip": false,
  "confidence": 0.15,
  "bank_detected": null,
  "ocr_result": null,
  "ai_result": null
}
```

## 🏦 การตั้งค่าธนาคาร

### ธนาคารที่รองรับ

ระบบรองรับธนาคารต่อไปนี้:

| ธนาคาร | Code | QR Bank Code | Template File |
|--------|------|--------------|---------------|
| กสิกรไทย | `kbank` | `004` | `kbank_template.json` |
| ไทยพาณิชย์ | `scb` | `014` | `scb_template.json` |
| กรุงไทย | `ktb` | `006` | `ktb_template.json` |
| ออมสิน | `gsb` | `030` | `gsb_template.json` |
| กรุงเทพ | `bbl` | `002` | `bbl_template.json` |
| กรุงศรีอยุธยา | `bay` | `025` | `bay_template.json` |
| ยูโอบี | `uob` | `024` | `uob_template.json` |

### จุดสำคัญในการตั้งค่า

#### 1. **ตั้งค่า Confidence Threshold** (ไฟล์: `api.py`)

```python
CONFIDENCE_THRESHOLD = 0.85  # ⚠️ จุดนี้คือค่า threshold สำหรับตัดสินว่าเป็นสลิปหรือไม่
```

**หมายเหตุ:** 
- ถ้า confidence >= 0.85 จะถือว่าเป็นสลิปและทำ OCR
- ถ้า confidence < 0.85 จะถือว่าไม่ใช่สลิป
- **TODO:** ถ้าต้องการเปลี่ยนค่า threshold ให้แก้ที่บรรทัดนี้

#### 2. **ตั้งค่า Bank Templates** (ไฟล์: `model.py`)

```python
BANK_TEMPLATES = {
    'kbank': os.path.join(BASE_DIR, 'template', 'kbank_template.json'),
    'scb': os.path.join(BASE_DIR, 'template', 'scb_template.json'),
    # ... เพิ่มธนาคารใหม่ที่นี่
}
```

**หมายเหตุ:** 
- แต่ละธนาคารต้องมี template file อยู่ในโฟลเดอร์ `template/`
- Template file ใช้รูปแบบ JSON (VIA annotation format)

#### 3. **ตั้งค่า QR Bank Code Mapping** (ไฟล์: `model.py`)

```python
QR_BANK_CODE_MAP = {
    "004": "kbank",  # ธนาคารกสิกรไทย
    "014": "scb",    # ธนาคารไทยพาณิชย์
    "006": "ktb",    # ธนาคารกรุงไทย
    "030": "gsb",    # ธนาคารออมสิน
    "002": "bbl",    # ธนาคารกรุงเทพ
    "025": "bay",    # ธนาคารกรุงศรีอยุธยา
    "024": "uob",    # ธนาคารยูโอบี
}
```

**หมายเหตุ:**
- QR Bank Code คือตัวเลข 3 หลักที่อยู่ใน QR Code ของสลิป (ตำแหน่ง index 18-20)
- **อ้างอิง:** รายการ Bank Code ทั้งหมดสามารถดูได้ที่ [Bank of Thailand - QR Code Standard](https://www.bot.or.th/Thai/FinancialInstitutions/Standard/Pages/QRCode.aspx)
- **NOTE:** ถ้าเพิ่มธนาคารใหม่ ต้องเพิ่ม mapping ตรงนี้ด้วย

#### 4. **ตั้งค่า Bank Keywords** (ไฟล์: `model.py`)

```python
BANK_KEYWORDS = {
    'kbank': [
        'กสิกรไทย', 'กสิกร', 'kbank', 'kasikorn', ...
    ],
    # ... เพิ่ม keywords สำหรับธนาคารใหม่
}
```

**หมายเหตุ:**
- Keywords ใช้สำหรับการตรวจสอบธนาคารจากข้อความในสลิป (fallback method)
- แต่ตอนนี้ระบบใช้ QR Code เป็นหลัก

## ➕ การเพิ่มธนาคารใหม่

### ขั้นตอนการเพิ่มธนาคารใหม่

1. **หา QR Bank Code ของธนาคาร**
   - ดูจาก QR Code ในสลิป (ตัวเลขตำแหน่งที่ 18-20)
   - หรือดูจาก [Bank of Thailand - QR Code Standard](https://www.bot.or.th/Thai/FinancialInstitutions/Standard/Pages/QRCode.aspx)

2. **สร้าง Template File**
   - ใช้ VIA (VGG Image Annotator) เพื่อสร้าง template
   - บันทึกเป็น JSON format ไว้ที่ `template/<bank_code>_template.json`
   - Template ต้องมี fields: `sender_name`, `sender_account`, `receiver_name`, `receiver_account`, `amount`, `transfer_date`, `transfer_time`, `reference_number`, `qr_code`, `sender_bank`, `receiver_bank`, `fee`

3. **เพิ่มใน `BANK_TEMPLATES`** (ไฟล์: `model.py` บรรทัด ~18)
   ```python
   BANK_TEMPLATES = {
       # ... ธนาคารเดิม
       'new_bank': os.path.join(BASE_DIR, 'template', 'new_bank_template.json'),
   }
   ```

4. **เพิ่มใน `QR_BANK_CODE_MAP`** (ไฟล์: `model.py` บรรทัด ~58)
   ```python
   QR_BANK_CODE_MAP = {
       # ... mapping เดิม
       "XXX": "new_bank",  # XXX คือ QR Bank Code (3 หลัก)
   }
   ```

5. **เพิ่มใน `BANK_KEYWORDS`** (ไฟล์: `model.py` บรรทัด ~28)
   ```python
   BANK_KEYWORDS = {
       # ... keywords เดิม
       'new_bank': [
           'ชื่อธนาคารภาษาไทย', 'ชื่อย่อ', 'ชื่อภาษาอังกฤษ', ...
       ],
   }
   ```

6. **ทดสอบ**
   - ส่งรูปสลิปของธนาคารใหม่มาทดสอบ
   - ตรวจสอบว่า OCR ทำงานถูกต้อง

## 🔍 Troubleshooting

### ปัญหา: "ไม่พบ QR Code ในภาพสลิป"

**สาเหตุ:**
- ภาพไม่ชัด
- QR Code ถูกบดบัง
- ภาพไม่ใช่สลิปจริง

**วิธีแก้:**
- ตรวจสอบว่าภาพชัดเจนและเห็น QR Code ชัด
- ลองใช้รูปสลิปอื่น

### ปัญหา: "ไม่รู้จักธนาคารนี้ (Bank Code: XXX)"

**สาเหตุ:**
- QR Bank Code ไม่ได้ถูกเพิ่มใน `QR_BANK_CODE_MAP`

**วิธีแก้:**
- เพิ่ม mapping ใน `QR_BANK_CODE_MAP` ตามขั้นตอนใน [การเพิ่มธนาคารใหม่](#การเพิ่มธนาคารใหม่)

### ปัญหา: "ไม่สามารถระบุเทมเพลตธนาคารที่ใช้ได้"

**สาเหตุ:**
- ไม่มี template file สำหรับธนาคารนั้น
- Template file อยู่ผิดตำแหน่ง

**วิธีแก้:**
- ตรวจสอบว่ามี template file ใน `template/` folder
- ตรวจสอบว่า path ใน `BANK_TEMPLATES` ถูกต้อง

### ปัญหา: API ไม่ทำงาน

**วิธีแก้:**
- ตรวจสอบว่า port 5555 ไม่ถูกใช้งาน
- ตรวจสอบว่า dependencies ติดตั้งครบ
- ตรวจสอบว่าไฟล์โมเดล `best_cnn_model.pth` อยู่ถูกตำแหน่ง

## 📝 หมายเหตุเพิ่มเติม

- **GPU Support:** หากต้องการใช้ GPU ให้ติดตั้ง PyTorch ที่รองรับ CUDA
- **macOS zbar:** บน macOS อาจต้องตั้งค่า `DYLD_LIBRARY_PATH` สำหรับ pyzbar (โค้ดจัดการให้อัตโนมัติแล้ว)
- **Template Format:** Template files ใช้ VIA (VGG Image Annotator) JSON format

## 📚 อ้างอิง

- [Bank of Thailand - QR Code Standard](https://www.bot.or.th/Thai/FinancialInstitutions/Standard/Pages/QRCode.aspx) - รายการ Bank Code ทั้งหมด
- [VIA (VGG Image Annotator)](https://www.robots.ox.ac.uk/~vgg/software/via/) - เครื่องมือสำหรับสร้าง template annotations
