from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, jsonify, send_file
import torch
import io

app = Flask(__name__)

print("🔁 Loading model...")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
print("✅ Model loaded!")

@app.route('/detect-image', methods=['POST'])
def detect_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file sent"}), 400

    image_file = request.files['image']
    image = Image.open(image_file.stream).convert("RGB")
    draw = ImageDraw.Draw(image)

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]

    print(f"[INFO] Deteksi ditemukan: {len(results['scores'])}")

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_text = f"{model.config.id2label[label.item()]} {round(score.item()*100)}%"
        box = [round(i, 2) for i in box.tolist()]

        print(f"[DETECTED] {label_text} di {box}")

        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0] + 5, box[1] + 5), label_text, fill="red", font=font)

    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)

    return send_file(buf, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
