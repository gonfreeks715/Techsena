from ultralytics import YOLO
import time, numpy as np

model = YOLO("../runs/cargo/xray_v1/weights/best.pt")
metrics = model.val(data="../datasets/cargo_final/data.yaml", split="test", verbose=False)

print("\n══════════════════════════════════════")
print("  CARGOVISION AI — MODEL RESULTS")
print("══════════════════════════════════════")
print(f"  mAP@50    : {metrics.box.map50:.3f}   (need > 0.75)")
print(f"  Precision : {metrics.box.mp:.3f}   (need > 0.80)")
print(f"  Recall    : {metrics.box.mr:.3f}   (need > 0.75)")
print("──────────────────────────────────────")
for i, name in enumerate(model.names.values()):
    try:
        bar = "█" * int(metrics.box.ap50[i] * 20)
        print(f"  {name:<22} {metrics.box.ap50[i]:.2f}  {bar}")
    except: pass

# Speed test
dummy = np.zeros((640,640,3), dtype=np.uint8)
times = [(__import__('time').time(), model.predict(dummy, verbose=False), __import__('time').time()) for _ in range(10)]
avg = sum(c-a for a,_,c in times)/10*1000
print(f"\n  Inference : {avg:.0f}ms per image  ({1000/avg:.0f} FPS)")
print("══════════════════════════════════════")