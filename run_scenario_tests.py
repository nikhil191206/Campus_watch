"""Scenario Test Runner for SmartRoad AI.

Runs integrate.py logic on each of Sanat's 6 test video scenarios,
records the dominant RL action per clip, and outputs results to
scenario_test_results.csv.

Usage:
    python run_scenario_tests.py
    python run_scenario_tests.py --test-dir test_data --model ppo_v2.zip

Author: Sachi (Pipeline Integration Lead) — Day 3
"""

import os
import sys
import csv
import argparse
import cv2
import torch
import numpy as np
from datetime import datetime
from collections import Counter
from PIL import Image

# ---- Model imports ----
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from ultralytics import YOLO

# ---- Project imports ----
from obs_builder import build_observation, reset_tracker, get_tracker_state
from attention_score import compute_attention_score_from_durations

# ---- Optional: PPO agent ----
try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

# ---- Constants ----
SEG_INTERVAL = 10
ACTION_LABELS = {0: "All Clear", 1: "Monitor", 2: "Violation"}

# Expected actions per scenario (from Sanat's test_scenarios.md)
EXPECTED_ACTIONS = {
    "scenario_1": 2,  # Phone while looking ahead → Violation
    "scenario_2": 2,  # Phone while looking away → Violation
    "scenario_3": 2,  # Driver sleeping / eyes closed → Violation
    "scenario_4": 1,  # Smoking → Monitor (or Violation)
    "scenario_5": 0,  # Fully attentive → All Clear
    "scenario_6": 1,  # Brief phone glance → Monitor
}


def rule_based_action(obs):
    """Fallback rule-based action when no PPO model is available."""
    phone_dur = obs[5]
    gaze_dur = obs[6]
    if phone_dur > 3.0 or gaze_dur > 4.0:
        return 2
    elif phone_dur > 1.0 or gaze_dur > 1.5:
        return 1
    else:
        return 0


def run_scenario(video_path, seg_processor, seg_model, yolo_model, ppo_agent=None):
    """Run a single scenario video through the full pipeline.

    Processes every frame of the video, collects all RL actions,
    and returns aggregate statistics.

    Args:
        video_path (str): Path to the .mp4 scenario video.
        seg_processor: HuggingFace SegformerImageProcessor.
        seg_model: HuggingFace SegformerForSemanticSegmentation.
        yolo_model: Ultralytics YOLO model.
        ppo_agent: Trained PPO model or None for rule-based fallback.

    Returns:
        dict: Results with keys:
            - dominant_action (int): Most common action across frames
            - action_counts (dict): {action_int: count}
            - max_attention_score (float): Peak attention score seen
            - avg_attention_score (float): Average attention score
            - total_frames (int): Number of frames processed
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Cannot open {video_path}"}

    reset_tracker()

    frame_count = 0
    actions_taken = []
    attention_scores = []
    seg_results = {"driver_zone": False, "steering_visible": True}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ---- SegFormer (every SEG_INTERVAL frames) ----
        if frame_count % SEG_INTERVAL == 0 or frame_count == 1:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = seg_processor(images=pil_img, return_tensors="pt")
            with torch.no_grad():
                outputs = seg_model(**inputs)
            seg_map = outputs.logits.argmax(dim=1)[0].numpy()
            seg_resized = cv2.resize(
                seg_map.astype(np.uint8),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            unique_classes = set(np.unique(seg_resized).tolist())
            driver_zone = bool(unique_classes & {12, 15, 20})
            steering_visible = bool(unique_classes & {15, 20, 135}) or driver_zone
            seg_results = {
                "driver_zone": driver_zone,
                "steering_visible": steering_visible
            }

        # ---- YOLOv8 ----
        results = yolo_model(frame, verbose=False)
        yolo_results = {}
        if results[0].boxes:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = results[0].names[cls_id]
                conf = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                if cls_name not in yolo_results or conf > yolo_results[cls_name]["conf"]:
                    yolo_results[cls_name] = {
                        "bbox": [int(b) for b in bbox],
                        "conf": round(conf, 4)
                    }

        # ---- Observation + Action ----
        obs = build_observation(yolo_results, seg_results)

        if ppo_agent is not None:
            action, _ = ppo_agent.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = rule_based_action(obs)

        actions_taken.append(action)

        # ---- Attention Score ----
        tracker = get_tracker_state()
        score = compute_attention_score_from_durations(
            tracker["phone_duration"],
            tracker["gaze_duration"],
            tracker.get("cigarette_duration", 0.0)
        )
        attention_scores.append(score)

    cap.release()

    if not actions_taken:
        return {"error": "No frames processed"}

    action_counts = Counter(actions_taken)
    dominant_action = action_counts.most_common(1)[0][0]

    return {
        "dominant_action": dominant_action,
        "action_counts": dict(action_counts),
        "max_attention_score": max(attention_scores),
        "avg_attention_score": sum(attention_scores) / len(attention_scores),
        "total_frames": frame_count
    }


def main():
    parser = argparse.ArgumentParser(description="SmartRoad AI Scenario Tester")
    parser.add_argument("--test-dir", default="test_data",
                        help="Directory containing scenario .mp4 files")
    parser.add_argument("--model", default="ppo_v2.zip",
                        help="Path to trained PPO model (.zip)")
    parser.add_argument("--output", default="scenario_test_results.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    print("=" * 70)
    print("SmartRoad AI — Scenario Test Runner")
    print("=" * 70)

    # ---- Load Models ----
    print("\n[INIT] Loading SegFormer-b0...")
    seg_processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    )
    seg_model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    )
    seg_model.eval()

    print("[INIT] Loading YOLOv8 Nano...")
    yolo_model = YOLO("yolov8n.pt")

    ppo_agent = None
    if SB3_AVAILABLE and os.path.exists(args.model):
        print(f"[INIT] Loading PPO agent from {args.model}...")
        ppo_agent = PPO.load(args.model)
    else:
        fallback = "ppo_v1.zip"
        if SB3_AVAILABLE and os.path.exists(fallback):
            print(f"[INIT] Loading fallback PPO from {fallback}...")
            ppo_agent = PPO.load(fallback)
        else:
            print("[WARN] No PPO model found — using rule-based fallback.")

    # ---- Find scenario videos ----
    if not os.path.isdir(args.test_dir):
        print(f"[ERROR] Test directory not found: {args.test_dir}")
        print("[INFO] Create test_data/ with scenario_1.mp4 ... scenario_6.mp4")
        print("[INFO] Running in dry-run mode with synthetic results.\n")
        # Dry-run for when videos aren't available yet
        write_dryrun_csv(args.output)
        return

    video_files = sorted([
        f for f in os.listdir(args.test_dir)
        if f.endswith(('.mp4', '.avi', '.mov'))
    ])

    if not video_files:
        print(f"[WARN] No video files found in {args.test_dir}/")
        write_dryrun_csv(args.output)
        return

    # ---- Run tests ----
    results_rows = []
    passed = 0
    total = 0

    for vf in video_files:
        video_path = os.path.join(args.test_dir, vf)
        scenario_key = os.path.splitext(vf)[0]  # e.g., "scenario_1"
        expected = EXPECTED_ACTIONS.get(scenario_key, -1)

        print(f"\n[TEST] Running: {vf}")
        result = run_scenario(video_path, seg_processor, seg_model,
                              yolo_model, ppo_agent)

        if "error" in result:
            print(f"  [ERROR] {result['error']}")
            continue

        actual = result["dominant_action"]
        correct = "PASS" if actual == expected else "FAIL"
        if actual == expected:
            passed += 1
        total += 1

        row = {
            "file_name": vf,
            "expected_action": ACTION_LABELS.get(expected, "Unknown"),
            "actual_action": ACTION_LABELS.get(actual, "Unknown"),
            "max_attention_score": f"{result['max_attention_score']:.2f}",
            "avg_attention_score": f"{result['avg_attention_score']:.2f}",
            "total_frames": result["total_frames"],
            "action_distribution": str(result["action_counts"]),
            "result": correct
        }
        results_rows.append(row)

        print(f"  Frames: {result['total_frames']}  |  "
              f"Expected: {ACTION_LABELS.get(expected, '?')}  |  "
              f"Actual: {ACTION_LABELS.get(actual, '?')}  |  "
              f"Attention: {result['avg_attention_score']:.1f} (avg), "
              f"{result['max_attention_score']:.1f} (max)  |  "
              f"[{correct}]")

    # ---- Write CSV ----
    write_results_csv(args.output, results_rows)

    # ---- Summary ----
    print(f"\n{'=' * 70}")
    print(f"SCENARIO TEST SUMMARY: {passed}/{total} passed")
    print(f"Results saved to: {args.output}")
    print(f"{'=' * 70}")


def write_results_csv(output_path, rows):
    """Write scenario test results to CSV."""
    headers = [
        "file_name", "expected_action", "actual_action",
        "max_attention_score", "avg_attention_score",
        "total_frames", "action_distribution", "result"
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def write_dryrun_csv(output_path):
    """Write a template CSV when no test videos are available."""
    print("[DRY-RUN] Writing template scenario_test_results.csv...")
    headers = [
        "file_name", "expected_action", "actual_action",
        "max_attention_score", "avg_attention_score",
        "total_frames", "action_distribution", "result"
    ]
    template_rows = [
        {"file_name": f"scenario_{i}.mp4",
         "expected_action": ACTION_LABELS.get(EXPECTED_ACTIONS.get(f"scenario_{i}", 0), "?"),
         "actual_action": "PENDING",
         "max_attention_score": "N/A",
         "avg_attention_score": "N/A",
         "total_frames": 0,
         "action_distribution": "{}",
         "result": "PENDING"}
        for i in range(1, 7)
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(template_rows)
    print(f"  Template saved to: {output_path}")


if __name__ == "__main__":
    main()
