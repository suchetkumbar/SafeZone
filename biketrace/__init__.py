"""BikeTrace: CCTV Stolen 2-Wheeler Forensics & Suspect Identification Toolkit."""
from .config import ForensicsConfig, TargetSpec
from .pipeline import ForensicsPipeline

__version__ = "1.0.0"
__all__ = ["ForensicsConfig", "TargetSpec", "ForensicsPipeline"]
