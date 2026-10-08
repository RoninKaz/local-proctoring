"""
Локальный прокторинг: прототип (YOLO + MediaPipe + защита среды).
Клавиши в окне:  g = защита ВКЛ/ВЫКЛ,  c = перекалибровка,  q = выход
"""
import os, time, csv, ctypes
import cv2
import keyboard
import mediapipe as mp
from ultralytics import YOLO
try:
    import psutil
except ImportError:
    psutil = None

WIN = "Local Proctoring"
YOLO_MODEL = "yolov8n.pt"      # можно "yolo11n.pt"
CONF = 0.40                    # порог уверенности YOLO
YOLO_EVERY = 3                 # YOLO на каждом 3-м кадре (экономит CPU)
SECS_DOWN = 3                  # взгляд вниз дольше N секунд
SECS_SIDE = 2                  # взгляд влево/вправо дольше N секунд
SECS_NOFACE = 3                # нет человека дольше N секунд
SECS_MULTI = 1                 # второе лицо дольше N секунд
SECS_PHONE = 1                 # телефон в кадре дольше N секунд
YAW_T = 0.12                   # порог поворота головы
PITCH_T = 0.08                 # порог наклона головы
IRIS_T = 0.12                  # порог смещения зрачка по горизонтали
PITCH_SIGN = 1                 
ALLOWED_HOTKEYS = ["alt+tab", "ctrl+c", "ctrl+v", "print screen"]
WEIGHTS = {"PHONE": 30, "PHONE_RAISED": 40, "GAZE_DOWN": 10, "GAZE_SIDE": 10,
           "NO_PERSON": 15, "SECOND_FACE": 25, "KEY_BLOCKED": 5, "FOREIGN_WINDOW": 10}
# -------------------------------------------------------------------------

os.makedirs("logs/evidence", exist_ok=True)
LOG = "logs/events.csv"
if not os.path.exists(LOG):
    with open(LOG, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["time", "event", "detail"])

events, score, LAST = [], [0], [None]
user32 = ctypes.windll.user32


def log(ev, detail=""):
    t = time.strftime("%H:%M:%S")
    events.append((t, ev, detail))
    score[0] = min(100, score[0] + WEIGHTS.get(ev, 5))
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([t, ev, detail])
    if LAST[0] is not None:  # кадр-доказательство
        cv2.imwrite(f"logs/evidence/{time.strftime('%H%M%S')}_{ev}.jpg", LAST[0])
    print(f"[{t}] {ev} {detail}")


# ---------------- 2.3 ЗАЩИТА СРЕДЫ ----------------
guard_on = [False]


def guard(on):
    keyboard.unhook_all()
    guard_on[0] = on
    if on:
        for hk in ALLOWED_HOTKEYS:
            keyboard.add_hotkey(hk, lambda h=hk: log("KEY_BLOCKED", h), suppress=True)
        for k in ("left windows", "right windows"):
            keyboard.block_key(k)


def fg_title():
    try:
        h = user32.GetForegroundWindow()
        n = user32.GetWindowTextLengthW(h)
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(h, buf, n + 1)
        return buf.value
    except Exception:
        return ""


# ---------------- ВСПОМОГАТЕЛЬНОЕ ----------------
state = {}


def sustain(name, cond, secs, cooldown=5):
    now = time.time()
    s = state.setdefault(name, {"since": None, "last": 0})
    if cond:
        if s["since"] is None:
            s["since"] = now
        if now - s["since"] >= secs and now - s["last"] >= cooldown:
            s["last"] = now
            return True
    else:
        s["since"] = None
    return False


def head_eye(lm):
    P = lambda i: (lm[i].x, lm[i].y)
    nose, L, R, top, chin = P(1), P(234), P(454), P(10), P(152)
    yaw = (nose[0] - L[0]) / (R[0] - L[0] + 1e-6)
    pitch = (nose[1] - top[1]) / (chin[1] - top[1] + 1e-6)
    r = lambda i, a, b: (lm[i].x - lm[a].x) / (lm[b].x - lm[a].x + 1e-6)
    iris_h = (r(468, 33, 133) + r(473, 362, 263)) / 2
    return yaw, pitch, iris_h


def text(img, s, org, col=(255, 255, 255), sc=0.55, th=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th + 2)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th)


# ---------------- ЗАПУСК ----------------
model = YOLO(YOLO_MODEL)
face = mp.solutions.face_mesh.FaceMesh(max_num_faces=2, refine_landmarks=True,
                                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("Камера не открылась")
cv2.namedWindow(WIN)

base, calib, frame_i = None, [], 0
boxes = []          # последние детекции YOLO: (cls, x1, y1, x2, y2, conf)
fps, t_prev = 0.0, time.time()
fps_list, cpu_list, ram_list = [], [], []
last_fg, last_fg_t = "", 0

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        LAST[0] = frame.copy()
        frame_i += 1

        # ---- 2.1 YOLO: телефон и люди ----
        if frame_i % YOLO_EVERY == 0:
            r = model(frame, imgsz=320, conf=CONF, classes=[0, 67], verbose=False)[0]
            boxes = [(int(b.cls), *map(int, b.xyxy[0]), float(b.conf)) for b in r.boxes]
        phones = [b for b in boxes if b[0] == 67]
        persons = [b for b in boxes if b[0] == 0]
        for cls, x1, y1, x2, y2, cf in boxes:
            col = (0, 165, 255) if cls == 67 else (255, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            text(frame, f"{'cell phone' if cls == 67 else 'person'} {cf:.2f}", (x1, max(15, y1 - 6)), col)

        # телефон поднят/направлен на экран: крупный бокс в верхней части кадра
        raised = any(((b[3] - b[1]) * (b[4] - b[2]) > 0.04 * w * h) and ((b[2] + b[4]) / 2 < 0.6 * h)
                     for b in phones)

        # ---- 2.2 MediaPipe: лицо, взгляд, голова ----
        res = face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        faces = res.multi_face_landmarks or []
        n_faces = len(faces)
        gaze = "NO FACE"
        down = side = False
        if n_faces:
            yaw, pitch, iris_h = head_eye(faces[0].landmark)
            if base is None:
                calib.append((yaw, pitch, iris_h))
                gaze = f"CALIBRATING {len(calib)}/40"
                if len(calib) >= 40:
                    base = tuple(sum(v[i] for v in calib) / len(calib) for i in range(3))
            else:
                dy, dp, di = yaw - base[0], (pitch - base[1]) * PITCH_SIGN, iris_h - base[2]
                down = dp > PITCH_T
                side = abs(dy) > YAW_T or abs(di) > IRIS_T
                if down:
                    gaze = "DOWN"
                elif side:
                    gaze = "RIGHT" if (dy + di) > 0 else "LEFT"
                else:
                    gaze = "CENTER"
                text(frame, f"yaw {dy:+.2f} pitch {dp:+.2f} iris {di:+.2f}", (10, h - 12), (180, 180, 180), 0.45)
            for lm in faces[0].landmark[::6]:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (255, 255, 0), -1)

        calibrated = base is not None
        # ---- события ----
        if sustain("phone", bool(phones), SECS_PHONE):
            log("PHONE", "телефон в кадре")
        if sustain("raised", raised, SECS_PHONE):
            log("PHONE_RAISED", "телефон поднят / направлен на экран")
        if calibrated and sustain("down", down, SECS_DOWN):
            log("GAZE_DOWN", f"взгляд вниз > {SECS_DOWN}с")
        if calibrated and sustain("side", side and not down, SECS_SIDE):
            log("GAZE_SIDE", f"взгляд в сторону > {SECS_SIDE}с ({gaze})")
        if calibrated and sustain("noface", n_faces == 0 and not persons, SECS_NOFACE):
            log("NO_PERSON", "в кадре нет человека")
        if sustain("multi", n_faces > 1 or len(persons) > 1, SECS_MULTI):
            log("SECOND_FACE", "второй человек в кадре")

        # ---- 2.3 посторонние окна ----
        if guard_on[0] and time.time() - last_fg_t > 0.5:
            last_fg_t = time.time()
            t = fg_title()
            if t and t != WIN and t != last_fg:
                log("FOREIGN_WINDOW", t[:60])
            last_fg = t

        # ---- панель ----
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1 / max(now - t_prev, 1e-6)) if fps else 15.0
        t_prev = now
        fps_list.append(fps)
        if psutil and frame_i % 15 == 0:
            cpu_list.append(psutil.cpu_percent())
            ram_list.append(psutil.Process().memory_info().rss / 1e6)
        cv2.rectangle(frame, (0, 0), (330, 150), (20, 20, 20), -1)
        text(frame, f"FPS {fps:4.1f}   faces {n_faces}   persons {len(persons)}", (10, 20), (0, 255, 0))
        text(frame, f"GAZE: {gaze}", (10, 42), (0, 255, 255))
        text(frame, f"GUARD: {'ON' if guard_on[0] else 'OFF'} (g)", (10, 64), (0, 200, 0) if guard_on[0] else (0, 0, 255))
        sc = score[0]
        text(frame, f"RISK SCORE: {sc}/100", (10, 86), (0, 0, 255) if sc >= 60 else (0, 255, 255))
        for i, (t, ev, d) in enumerate(events[-3:]):
            text(frame, f"{t} {ev}", (10, 108 + i * 15), (255, 255, 255), 0.42)
        cv2.imshow(WIN, frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("g"):
            guard(not guard_on[0])
            log("GUARD", "ON" if guard_on[0] else "OFF")
        if k == ord("c"):
            base, calib = None, []
finally:
    keyboard.unhook_all()
    cap.release()
    cv2.destroyAllWindows()
    if fps_list:
        print(f"\nСредний FPS: {sum(fps_list) / len(fps_list):.1f}")
    if cpu_list:
        print(f"Средняя нагрузка CPU: {sum(cpu_list) / len(cpu_list):.0f}%  |  RAM процесса: {sum(ram_list) / len(ram_list):.0f} МБ")
