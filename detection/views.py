# views.py
from django.shortcuts import render
from .form import ImageUploadForm
from ultralytics import YOLO
import os
import cv2
import base64
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent


def index(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_instance = form.save()
            imagepath = os.path.join(BASE_DIR, image_instance.image.name)
            wei = os.path.join(BASE_DIR, 'best.pt')
            # output_dir = "../Django/helmet/output"
            # os.makedirs(output_dir, exist_ok=True)
            # results = model.predict(
            #     source=imagepath, conf=0.35, save=False, project=output_dir, name="yyy")
            model = YOLO(wei)
            results = model.predict(
                source=imagepath, conf=0.35, save=False)

            image = cv2.imread(imagepath)
            class_names = ["without helmet", "with helmet"]
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    class_id = int(box.cls[0])
                    class_name = class_names[class_id]
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'{class_name}: {confidence:.2f}'
                    cv2.putText(image, label, (x1, y2 + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode('.jpg', image_rgb)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            context = {
                'image_base64': image_base64,
            }
            return render(request, 'image.html', context)
    else:
        form = ImageUploadForm()
    return render(request, 'index.html', {'form': form})
