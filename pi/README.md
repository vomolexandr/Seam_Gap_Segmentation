# Pi-частина: розгортання на Raspberry Pi 5

Цю теку копіюють на Pi разом із `best.onnx` з
`runs/segment/<назва_запуску>/weights/`:

```bash
scp -r pi/ runs/segment/seam_gap_v2/weights/best.onnx pi@<ip>:~/seam/
```

## Встановлення на Pi

```bash
sudo apt install -y python3-picamera2 python3-opencv
pip install onnxruntime pymodbus numpy
```

## Сценарії

| Скрипт | Призначення |
|---|---|
| `capture_dataset.py` | збір 50–100 кадрів для fine-tuning (ENTER або інтервал) |
| `seam_infer.py` | декодер моделі + екстракція кромок; можна ганяти по файлах |
| `run_live.py` | живий цикл: камера → модель → кромки → Modbus TCP |

Перший запуск на Pi (перевірка без робота):

```bash
python3 run_live.py --onnx best.onnx
```

З Modbus-сервером (робот читає holding-регістри, мапа регістрів у шапці `run_live.py`):

```bash
python3 run_live.py --onnx best.onnx --modbus 502
```

## Продуктивність

onnxruntime на CPU Pi 5 дасть орієнтовно 3–7 FPS на 640×640. Цього досить для
першого налагодження. Повна швидкість досягається після компіляції під Hailo:

1. На ноуті/ПК з Linux (або WSL2): встановити **Hailo Dataflow Compiler**
   (реєстрація на hailo.ai → Developer Zone → Software Downloads).
2. `hailomz compile yolov8n_seg --ckpt best.onnx --hw-arch hailo8l
   --calib-path <тека з ~64 реальними кадрами>` → результат `.hef`.
3. На Pi: `sudo apt install hailo-all`, запуск через HailoRT/GStreamer.
   У `seam_infer.py` замінюється лише клас `SeamDetector` (виклик моделі);
   декодування і кромки не змінюються.

## Калібрування «пікселі → міліметри»

Для робота координати треба переводити в мм. Найпростіше: зняти кадр із
лінійкою/шаблоном відомої ширини на робочій відстані, порахувати px/мм.
Якщо відстань від камери до деталі фіксована, це один коефіцієнт.
