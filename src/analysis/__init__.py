"""
ASRC Micro Analysis - Core Analysis Modules
============================================

This package contains the main analysis pipeline for zero-shot
microstructural characterization using Foundation Models (SAM)
and physics-informed multi-modal learning.

Modules:
    01_nist_benchmark     - Validate SAM on NIST concrete dataset
    02_asrc_characterization - Extract features from ASRC samples
    03_robustness_test    - Test segmentation stability
    04_physics_fusion     - Physics-informed SEM-XRD fusion
    main_pipeline         - Orchestrate full analysis

For: Automation in Construction Journal Paper
Date: 2026-01
"""

__version__ = "1.0.0"
__author__ = "Research Team"

# Expose main functions for easy import
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
