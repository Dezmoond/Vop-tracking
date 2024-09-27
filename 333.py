from ultralytics import YOLO

# Загрузка модели YOLOv8
model = YOLO('D:/Dow/best (10).pt')  # Замените на свою модель, если нужно
model.to('cuda')
# Выполнение трекинга на видео и сохранение результатов
results = model.track(source='F:/УЧЕБА/Воп.mp4', show=True, save=True, tracker='botsort.yaml')  # Или 'bytetrack.yaml' проверить ямл файл
