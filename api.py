from flask import Flask, request, jsonify
import base64
from PIL import Image
import io
import torch
import requests
import os
import tempfile
import logging
from model import (
    load_cnn_model, 
    get_inference_transform, 
    classify_image_with_cnn, 
    ocr_with_auto_template,
    ocr_fallback_all_banks
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# โหลดโมเดลตอน start server
MODEL_PATH = 'IQ500/models/best_cnn_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn_model = load_cnn_model(MODEL_PATH, device=DEVICE)
transform = get_inference_transform()

CONFIDENCE_THRESHOLD = 0.85

def decode_base64_image(img_input):
    """แปลง base64 เป็น PIL Image"""
    try:
        # ถ้าเป็น data URL format
        if img_input.startswith("data:image/"):
            img_input = img_input.split(",")[1]
        
        # decode base64
        image_bytes = base64.b64decode(img_input)
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except Exception as e:
        raise ValueError(f"Invalid base64 image: {str(e)}")

@app.route('/predict-slip', methods=['POST'])
def predict_slip():
    temp_file = None
    
    try:
        logger.info("Received prediction request")
        
        # Validate input
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        img_input = data.get('image')
        if not img_input:
            return jsonify({"error": "No image provided"}), 400

        # Load image
        if img_input.startswith("http://") or img_input.startswith("https://"):
            # โหลดรูปจาก URL
            logger.info("Loading image from URL")
            resp = requests.get(img_input, timeout=30)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content))
        else:
            # แปลง base64
            logger.info("Decoding base64 image")
            image = decode_base64_image(img_input)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            temp_file = tmp.name
            image.save(temp_file)
            logger.info(f"Image saved to {temp_file}")

        # Step 1: ตรวจสอบว่าเป็น slip หรือไม่
        pred_class, confidence = classify_image_with_cnn(temp_file, cnn_model, transform, DEVICE)
        logger.info(f"CNN prediction: class={pred_class}, confidence={confidence}")

        # Initialize response
        response = {
            "is_slip": False, 
            "confidence": confidence, 
            "bank_detected": None,
            "ocr_result": None,
            "detection_method": None
        }

        # Step 2: ถ้าเป็น slip ทำ OCR
        if pred_class == 1 and confidence >= CONFIDENCE_THRESHOLD:
            logger.info("Image classified as slip, starting OCR...")
            
            # Step 3: ตรวจหาธนาคารและทำ OCR โดยใช้ sender_bank field
            ocr_result, detected_bank = ocr_with_auto_template(temp_file)
            
            if detected_bank and ocr_result:
                # พบธนาคารและได้ข้อมูล OCR
                logger.info(f"Bank detected from sender_bank field: {detected_bank}")
                response.update({
                    "is_slip": True,
                    "bank_detected": detected_bank,
                    "ocr_result": ocr_result,
                    "detection_method": "sender_bank_field"
                })
                
            else:
                # ไม่พบธนาคารจาก sender_bank ลอง fallback
                logger.info("Sender bank detection failed, trying fallback method...")
                fallback_result, fallback_bank = ocr_fallback_all_banks(temp_file)
                
                if fallback_result:
                    logger.info(f"Fallback successful with bank: {fallback_bank}")
                    response.update({
                        "is_slip": True,
                        "bank_detected": fallback_bank,
                        "ocr_result": fallback_result,
                        "detection_method": "fallback_best_match"
                    })
                else:
                    logger.warning("All OCR methods failed")
                    response.update({
                        "is_slip": True,
                        "bank_detected": None,
                        "ocr_result": {},
                        "detection_method": "failed"
                    })
        else:
            logger.info("Image is not a slip or confidence too low")

        return jsonify(response)

    except requests.RequestException as e:
        logger.error(f"Network error: {str(e)}")
        return jsonify({"error": f"Failed to load image from URL: {str(e)}"}), 400
        
    except ValueError as e:
        logger.error(f"Image processing error: {str(e)}")
        return jsonify({"error": str(e)}), 400
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500
        
    finally:
        # ลบไฟล์ชั่วคราว
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                logger.info(f"Temporary file {temp_file} removed")
            except Exception as e:
                logger.warning(f"Failed to remove temp file: {str(e)}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, debug=False)