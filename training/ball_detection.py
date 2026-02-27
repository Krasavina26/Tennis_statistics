from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo26n.pt")

    results = model.train(
        data='data.yaml',
        epochs=200,
        project='runs',
        name='train_ball',
        batch=16,
        device=0,
        imgsz=896,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10.0, translate=0.1, scale=0.5, shear=2.0,
        val=True,
        exist_ok=True,
        workers=4,
        patience=50,
        save_period=5,
    )
