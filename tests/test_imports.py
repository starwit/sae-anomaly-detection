import pytest

def test_anomalydetection_import():
    try:
        from anomalydetection.anomalydetection import AnomalyDetection
    except ImportError as e:
        pytest.fail(f"Failed to import AnomalyDetection: {e}")

    assert AnomalyDetection is not None, "AnomalyDetection should be imported successfully"

def test_main_import():
    try:
        from main import main
    except ImportError as e:
        pytest.fail(f"Failed to import main: {e}")

    assert main is not None, "main should be imported successfully"
    
def test_detector_import():
    try:
        from anomalydetection.detector import Detector
    except ImportError as e:
        pytest.fail(f"Failed to import Detector: {e}")

    assert Detector is not None, "Detector should be imported successfully"