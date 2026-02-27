from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('yolov8s.pt')

    results = model.train(
        data='data.yaml',
        epochs=50,
        project='runs',
        name='player_net_detection',
        batch=16,
        device=0,
        imgsz=800,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10.0, translate=0.1, scale=0.5, shear=2.0,
        val=False,
        exist_ok=True,
        workers=4,
        patience=15,
        save_period=10
    )
