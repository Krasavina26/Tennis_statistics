from ultralytics import YOLO

def continue_train_court():
    data_yml = r"C:\\Users\\krask\\Downloads\\Tennis Detection new version.v1i.yolov8\\data.yaml"

    model = YOLO('yolo26s-pose')
    
    results = model.train(
    data=data_yml,
    epochs=100,
    imgsz=1280,
    batch=4,
    workers=0,
    device=0,
    patience=20,
    
    augment=True,
    hsv_h=0.01, hsv_s=0.3, hsv_v=0.3,
    degrees=3.0,
    translate=0.1,
    scale=0.5,
    shear=3.0,
    fliplr=0.0,
    mosaic=0.1,
    mixup=0.02,
    
    close_mosaic=5,
    
    # box=7.0,
    # cls=0.5,
    # dfl=1.5,
    # pose=30.0,
    # kobj=1.5,
    
    amp=True,
    cache=True,
    
    name="court_yolo11m_pose_v558",
    project="runs/yolo11m_pose",
    exist_ok=True,
    # resume = True,
    # freeze=10,
    
    val=True,
    save_period=1
)

if __name__ == '__main__':
    continue_train_court()
