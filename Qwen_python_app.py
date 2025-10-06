from flask import Flask, request, jsonify, send_from_directory
import os
import requests
import easyocr
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import io
import hashlib
import time

app = Flask(__name__)
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

# โหลด OCR ครั้งเดียวตอนเริ่ม
print("กำลังโหลด EasyOCR (อาจใช้เวลา 10-30 วินาที)...")
reader = easyocr.Reader(['ja', 'ko', 'en', 'zh-cn', 'zh-tw'], gpu=False)
print("โหลด OCR เสร็จสิ้น")

# ฟอนต์ภาษาไทย (ดาวน์โหลดมาแล้ววางในโฟลเดอร์เดียวกัน)
FONT_PATH = "NotoSansThai-Regular.ttf"
if not os.path.exists(FONT_PATH):
    # ดาวน์โหลดฟอนต์ถ้ายังไม่มี
    print("กำลังดาวน์โหลดฟอนต์ภาษาไทย...")
    r = requests.get("https://github.com/googlefonts/noto-fonts/raw/main/unhinted/ttf/NotoSansThai/NotoSansThai-Regular.ttf")
    with open(FONT_PATH, "wb") as f:
        f.write(r.content)

def translate_text(text, target_lang="th"):
    try:
        # ใช้ LibreTranslate สาธารณะ (ฟรี)
        res = requests.post("https://libretranslate.com/translate", json={
            "q": text,
            "source": "auto",
            "target": target_lang,
            "format": "text"
        }, timeout=10)
        if res.status_code == 200:
            return res.json().get("translatedText", text)
        else:
            return text
    except:
        return text

@app.route('/translate', methods=['POST'])
def translate_image():
    data = request.json
    image_url = data.get('image_url')
    target_lang = data.get('target_lang', 'th')

    if not image_url:
        return jsonify({"error": "ต้องระบุ image_url"}), 400

    try:
        # ดาวน์โหลดรูป
        img_resp = requests.get(image_url, timeout=15)
        img_resp.raise_for_status()
        img_pil = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
        img_cv = np.array(img_pil)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        # OCR
        results = reader.readtext(img_cv)

        # ประมวลผลแต่ละกล่องข้อความ
        output_img = img_cv.copy()
        pil_output = Image.fromarray(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_output)

        for (bbox, text, prob) in results:
            if prob < 0.3:  # ความมั่นใจต่ำเกินไป ข้าม
                continue

            # ลบข้อความเดิม: ปิดทับด้วยค่าเฉลี่ยพื้นหลัง
            pts = np.array(bbox, dtype=np.int32)
            x_min = min(p[0] for p in pts)
            y_min = min(p[1] for p in pts)
            x_max = max(p[0] for p in pts)
            y_max = max(p[1] for p in pts)

            # ขยายกล่องเล็กน้อย
            pad = 2
            x_min = max(0, x_min - pad)
            y_min = max(0, y_min - pad)
            x_max = min(img_cv.shape[1], x_max + pad)
            y_max = min(img_cv.shape[0], y_max + pad)

            roi = output_img[y_min:y_max, x_min:x_max]
            if roi.size > 0:
                avg_color = np.mean(roi, axis=(0,1)).astype(int)
                cv2.rectangle(output_img, (x_min, y_min), (x_max, y_max), avg_color.tolist(), -1)

            # แปลข้อความ
            translated = translate_text(text, target_lang)

            # วาดคำแปลใหม่
            try:
                font_size = max(14, int((y_max - y_min) * 0.8))
                font = ImageFont.truetype(FONT_PATH, font_size)
                draw.text((x_min, y_min), translated, fill=(0, 0, 0), font=font)
            except:
                # fallback
                draw.text((x_min, y_min), translated, fill=(0, 0, 0))

        # บันทึกไฟล์
        filename = hashlib.md5(f"{image_url}_{target_lang}_{time.time()}".encode()).hexdigest() + ".jpg"
        filepath = os.path.join(STATIC_DIR, filename)
        pil_output.save(filepath, quality=90)

        return jsonify({
            "image_url": request.url_root + "static/" + filename
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))