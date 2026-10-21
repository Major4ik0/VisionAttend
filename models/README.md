# Офлайн-модели InsightFace для VisionAttend

В закрытом контуре (Air-gapped) модели должны быть скачаны заранее и размещены в этой папке.

## Путь размещения файлов:
`./models/insightface/models/buffalo_l/`

## Необходимые файлы:
1. `det_10g.onnx` (Детекция лиц SCRFD 10G — **Обязательно**)
2. `w600k_r50.onnx` (Извлечение векторов ArcFace ResNet-50 — **Обязательно**)
3. `1k3d68.onnx` (3D ключевые точки — опционально)
4. `2d106det.onnx` (2D ключевые точки — опционально)
5. `genderage.onnx` (Оценка пола/возраста — опционально)

## Где скачать:
Официальный архив `buffalo_l.zip`:
- GitHub: https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip

Распакуйте содержимое архива прямо в папку `models/insightface/models/buffalo_l/`.
