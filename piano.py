import cv2
import mediapipe as mp
import numpy as np
import wave
import os
import pygame
import time

# ─────────────────────────────────────────
# GENERATE PIANO NOTE WAV FILES
# ─────────────────────────────────────────
def generate_piano_note(filename, frequency, duration=4.0, sample_rate=44100):
    if os.path.exists(filename):
        return
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Fundamental tone + rich harmonic series for piano timbre
    wave_data = (
        1.00 * np.sin(2 * np.pi * frequency * 1 * t) +
        0.60 * np.sin(2 * np.pi * frequency * 2 * t) +
        0.30 * np.sin(2 * np.pi * frequency * 3 * t) +
        0.15 * np.sin(2 * np.pi * frequency * 4 * t) +
        0.08 * np.sin(2 * np.pi * frequency * 5 * t) +
        0.04 * np.sin(2 * np.pi * frequency * 6 * t)
    )
    
    # Natural decay envelope
    envelope = 0.5 * np.exp(-3.0 * t) + 0.5 * np.exp(-0.8 * t)
    wave_data = wave_data * envelope
    
    # Soft end fade-out to prevent popping/clicking
    fade_samples = int(sample_rate * 0.05)
    wave_data[-fade_samples:] *= np.linspace(1, 0, fade_samples)

    # Normalize audio
    wave_data = wave_data / np.max(np.abs(wave_data))
    wave_data = (wave_data * 32767).astype(np.int16)
    
    with wave.open(filename, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(wave_data.tobytes())

FREQUENCIES = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
NOTE_FILES  = [f"note_{i}.wav" for i in range(8)]

print("Generating piano notes...")
for i, freq in enumerate(FREQUENCIES):
    generate_piano_note(NOTE_FILES[i], freq)
print("Done!")

# ─────────────────────────────────────────
# PYGAME INIT
# ─────────────────────────────────────────
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
pygame.mixer.set_num_channels(32)
sounds = [pygame.mixer.Sound(f) for f in NOTE_FILES]

# ─────────────────────────────────────────
# MEDIAPIPE INIT (Tracking up to 2 hands)
# ─────────────────────────────────────────
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ─────────────────────────────────────────
# WEBCAM + KEY LAYOUT (Top Placement)
# ─────────────────────────────────────────
cap = cv2.VideoCapture(0)
ret, test_frame = cap.read()
FRAME_H, FRAME_W = test_frame.shape[:2]

NUM_KEYS    = 8
GAP         = 6
KEY_H       = 160
KEY_W       = (FRAME_W - GAP * (NUM_KEYS + 1)) // NUM_KEYS
KEY_Y       = 20  # Positioned at top of screen
total_width = NUM_KEYS * KEY_W + (NUM_KEYS - 1) * GAP
START_X     = (FRAME_W - total_width) // 2

NOTE_NAMES = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C2']

keys = []
for i in range(NUM_KEYS):
    x1 = START_X + i * (KEY_W + GAP)
    x2 = x1 + KEY_W
    keys.append((x1, KEY_Y, x2, KEY_Y + KEY_H))

# ─────────────────────────────────────────
# STATE
# ─────────────────────────────────────────
last_pressed_keys = set()

# ─────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame   = cv2.flip(frame, 1)
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    finger_positions = []
    fingertip_ids = [8]  # Index fingertip ID on each hand

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for tip_id in fingertip_ids:
                tip = hand_landmarks.landmark[tip_id]
                fx, fy = int(tip.x * FRAME_W), int(tip.y * FRAME_H)
                finger_positions.append((fx, fy))
                cv2.circle(frame, (fx, fy), 9, (0, 255, 0), -1)

    # ── Detect which keys are currently touched by any finger ──
    current_pressed_keys = set()
    for fx, fy in finger_positions:
        for i, (x1, y1, x2, y2) in enumerate(keys):
            if x1 < fx < x2 and y1 < fy < y2:
                current_pressed_keys.add(i)

    # ── Polyphonic Sound Trigger Logic ──
    newly_pressed_keys = current_pressed_keys - last_pressed_keys

    for key_idx in newly_pressed_keys:
        sounds[key_idx].play()

    last_pressed_keys = current_pressed_keys

    # ── Draw Keys at Top ──
    for i, (x1, y1, x2, y2) in enumerate(keys):
        touching = i in current_pressed_keys
        color    = (0, 255, 255) if touching else (255, 255, 255)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(frame, NOTE_NAMES[i],
                    (x1 + KEY_W // 2 - 10, y2 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ── UI Text (Moved to Bottom) ──
    cv2.putText(frame, "Polyphonic Virtual Piano", (10, FRAME_H - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
    cv2.putText(frame, "Play chords & merged notes with index fingers (Press ESC to exit)", (10, FRAME_H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    cv2.imshow("Virtual Piano", frame)

    key_pressed = cv2.waitKey(1)
    if key_pressed == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()