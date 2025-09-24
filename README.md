# IQ500

## Requirements

- Python 3.8+
- pip

## การติดตั้ง

1. **Clone repo นี้**
   ```bash
   git clone https://github.com/palitalsk/IQ500.git
   cd IQ500
   ```

2. **ติดตั้ง dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **เตรียมไฟล์โมเดล**
   - วางไฟล์โมเดล `best_cnn_model.pth` ไว้ที่ `IQ500/models/best_cnn_model.pth`

## การรัน API

```bash
python api.py
```

API จะรันที่ `http://0.0.0.0:5555`

## ตัวอย่างการเรียกใช้งาน API

### Endpoint: `/predict-slip` (POST)

**Request JSON:**
```json
{
  "image": "<base64 string หรือ URL ของรูป>"
}
```

**Response ตัวอย่าง:**
```json
{
  "is_slip": true, //true, false
  "confidence": 0.98, // ตอนนี้เขียนดักไว้ที่มากกว่า > 0.85 ถึงจะถือว่าเป็น slip >> ส่งไป ocr
  "bank_detected": "KBANK", // ตอนนี้ทำเทมเพลตไว้แค่ Kbank, GSB, KTB (BBL, SCB ยังไม่มแ่นตรวจไม่ได้)
  "ocr_result": {...},
}
```

## หมายเหตุ

- รองรับทั้ง base64 และ URL ของรูปภาพ
- หากต้องการรันบน GPU ให้ติดตั้ง PyTorch ที่รองรับ CUDA