"""Attention Score module for SmartRoad AI.

Computes a novel distraction severity metric (0–100) from per-activity
duration timers and provides a threshold-based action recommendation.

Author: Shakti (Attention Score & Metrics)
"""

# ==========================================================================
# Attention Score Formula
# ==========================================================================
#
#   AttentionScore = (phone_duration   * 0.4   — highest weight: most dangerous
#                  +  gaze_duration    * 0.3   — second: eyes off road is critical
#                  +  cig_duration     * 0.2   — third: manual + cognitive load
#                  +  activity_count   * 0.1)  — fourth: composite activity signal
#
#   Raw sum is normalised to 0–100 by dividing by MAX_RAW (worst-case score)
#   then clamping to [0, 100].
#
#   MAX_RAW = 30*0.4 + 30*0.3 + 30*0.2 + 10*0.1 = 12 + 9 + 6 + 1 = 28
#
# ==========================================================================

_MAX_RAW = 28.0   # Theoretical maximum of the weighted sum
_MAX_DUR = 30.0   # Duration cap in seconds (matches obs_builder FPS cap)
_MAX_ACT = 10     # Maximum activity_count value


class DurationTracker:
    """Tracks per-activity distraction durations across video frames.

    Intended for use in the evaluation pipeline where pipeline results
    are processed frame-by-frame outside the RL environment.

    Attributes:
        phone_timer (float): Cumulative phone distraction time in seconds.
        gaze_timer (float): Cumulative gaze-away time in seconds.
        cigarette_timer (float): Cumulative cigarette time in seconds.
    """

    def __init__(self):
        """Initialise all timers to zero."""
        self.phone_timer = 0.0
        self.gaze_timer = 0.0
        self.cigarette_timer = 0.0

    def update(self, detections_dict, fps=30):
        """Increment timers based on current-frame detections.

        Args:
            detections_dict (dict): Keys that trigger timer increments:
                - "phone" (bool or truthy): phone detected this frame
                - "gaze_away" (bool or truthy): driver looking away
                - "cigarette" (bool or truthy): cigarette detected
            fps (int): Frames per second — used to convert 1 frame → seconds.
        """
        frame_seconds = 1.0 / fps

        if detections_dict.get("phone"):
            self.phone_timer += frame_seconds
        if detections_dict.get("gaze_away"):
            self.gaze_timer += frame_seconds
        if detections_dict.get("cigarette"):
            self.cigarette_timer += frame_seconds

    def reset_all(self):
        """Reset all timers to zero."""
        self.phone_timer = 0.0
        self.gaze_timer = 0.0
        self.cigarette_timer = 0.0


def compute_attention_score(tracker, activity_count=0):
    """Compute the Attention Score from a DurationTracker instance.

    Args:
        tracker (DurationTracker): Tracker with current duration values.
        activity_count (int): Number of simultaneous distraction activities
            (0–10). Defaults to 0.

    Returns:
        float: Attention Score in range [0.0, 100.0].
            0  = fully attentive, no distraction.
            100 = maximum distraction across all categories.
    """
    return compute_attention_score_from_durations(
        phone_duration=tracker.phone_timer,
        gaze_duration=tracker.gaze_timer,
        cigarette_duration=tracker.cigarette_timer,
        activity_count=activity_count
    )


def compute_attention_score_from_durations(phone_duration, gaze_duration,
                                           cigarette_duration=0.0,
                                           activity_count=0):
    """Compute the Attention Score directly from duration floats.

    This is the primary function used by integrate.py and final_integrate.py
    because they already have tracker state as plain floats from
    obs_builder.get_tracker_state().

    Args:
        phone_duration (float): Seconds phone has been detected (capped at 30).
        gaze_duration (float): Seconds driver has been looking away (capped at 30).
        cigarette_duration (float): Seconds cigarette detected (capped at 30).
        activity_count (int): Number of simultaneous activities (0–10).

    Returns:
        float: Attention Score in range [0.0, 100.0].
    """
    # Cap inputs to defined maximums
    p = min(phone_duration, _MAX_DUR)
    g = min(gaze_duration, _MAX_DUR)
    c = min(cigarette_duration, _MAX_DUR)
    a = min(activity_count, _MAX_ACT)

    raw = p * 0.4 + g * 0.3 + c * 0.2 + a * 0.1
    score = (raw / _MAX_RAW) * 100.0
    return float(min(max(score, 0.0), 100.0))


def get_recommended_action(score):
    """Map an Attention Score to a recommended action label.

    Thresholds calibrated so that obvious violations score > 70 and
    safe driving scores < 30 (per Shakti's Day 3 calibration spec).

    Args:
        score (float): Attention Score in [0, 100].

    Returns:
        int: Recommended action:
            0 = ALL_CLEAR  (score < 60)
            1 = MONITOR    (60 <= score < 80)
            2 = VIOLATION  (score >= 80)
    """
    if score >= 80.0:
        return 2  # VIOLATION
    elif score >= 60.0:
        return 1  # MONITOR
    return 0      # ALL_CLEAR


def evaluate_model(predictions, ground_truth):
    """Compute precision, recall and F1 for agent predictions.

    Args:
        predictions (list[int]): Predicted actions (0, 1, or 2).
        ground_truth (list[int]): Ground-truth actions from scenarios.csv.

    Returns:
        dict: Keys 'report' (str), 'accuracy' (float).
            Body is filled on Day 2 when eval_results.csv is available.
    """
    from sklearn.metrics import classification_report, accuracy_score

    target_names = ["ALL_CLEAR", "MONITOR", "VIOLATION"]
    report = classification_report(
        ground_truth, predictions,
        target_names=target_names,
        zero_division=0
    )
    accuracy = accuracy_score(ground_truth, predictions)
    return {"report": report, "accuracy": accuracy}


# ==========================================================================
# Standalone test / smoke-check
# ==========================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("Testing attention_score.py")
    print("=" * 55)

    # ---- Test 1: DurationTracker increments correctly ----
    tracker = DurationTracker()
    fps = 30
    n_frames = 90  # ~3 seconds of phone detected

    for _ in range(n_frames):
        tracker.update({"phone": True, "gaze_away": False, "cigarette": False}, fps=fps)

    score = compute_attention_score(tracker)
    print(f"\nTest 1 — 90 frames (~3 s) of phone detected:")
    print(f"  phone_timer : {tracker.phone_timer:.2f}s")
    print(f"  gaze_timer  : {tracker.gaze_timer:.2f}s")
    print(f"  score       : {score:.2f}  (expected > 0)")
    assert score > 0, "FAILED: score should be > 0 after phone detection"
    print("  [OK]")

    # ---- Test 2: compute_attention_score_from_durations directly ----
    s = compute_attention_score_from_durations(phone_duration=10.0, gaze_duration=5.0,
                                               cigarette_duration=2.0, activity_count=3)
    print(f"\nTest 2 — phone=10s, gaze=5s, cig=2s, activity=3: score={s:.2f}")
    assert 0.0 <= s <= 100.0, "FAILED: score out of [0, 100]"
    print("  [OK]")

    # ---- Test 3: Zero input → zero score ----
    s_zero = compute_attention_score_from_durations(0, 0, 0, 0)
    print(f"\nTest 3 — all zeros: score={s_zero:.2f}  (expected 0.0)")
    assert s_zero == 0.0, "FAILED: zero input should give zero score"
    print("  [OK]")

    # ---- Test 4: Max input → 100 score ----
    s_max = compute_attention_score_from_durations(30.0, 30.0, 30.0, 10)
    print(f"\nTest 4 — all maxed: score={s_max:.2f}  (expected 100.0)")
    assert s_max == 100.0, "FAILED: max input should give 100.0"
    print("  [OK]")

    # ---- Test 5: get_recommended_action thresholds ----
    assert get_recommended_action(25.0) == 0, "Expected ALL_CLEAR"
    assert get_recommended_action(65.0) == 1, "Expected MONITOR"
    assert get_recommended_action(85.0) == 2, "Expected VIOLATION"
    print("\nTest 5 — get_recommended_action thresholds: [OK]")

    # ---- Test 6: reset_all ----
    tracker.reset_all()
    assert tracker.phone_timer == 0.0
    assert tracker.gaze_timer == 0.0
    assert tracker.cigarette_timer == 0.0
    print("Test 6 — reset_all: [OK]")

    print("\n" + "=" * 55)
    print("ALL attention_score.py TESTS PASSED")
    print("=" * 55)
