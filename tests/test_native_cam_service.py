import numpy as np

from app.services.gradcam_service import NativeCAMService


def test_energy_metrics_reward_joint_map_and_penalize_upper_femur_map():
    joint_cam = np.zeros((100, 100), dtype=np.float32)
    joint_cam[42:58, 15:85] = 1.0
    upper_cam = np.zeros((100, 100), dtype=np.float32)
    upper_cam[5:20, 20:80] = 1.0

    joint = NativeCAMService.energy_metrics(joint_cam)
    upper = NativeCAMService.energy_metrics(upper_cam)

    assert joint["peak_inside_joint"] is True
    assert upper["peak_inside_joint"] is False
    assert joint["joint_energy"] > upper["joint_energy"]
    assert joint["anatomy_score"] > upper["anatomy_score"]


def test_energy_metrics_measure_lower_tibia_leakage():
    cam = np.zeros((100, 100), dtype=np.float32)
    cam[75:90, 15:85] = 1.0

    metrics = NativeCAMService.energy_metrics(cam)

    assert metrics["lower_tibia_energy"] > 0.99
    assert metrics["peak_inside_joint"] is False
