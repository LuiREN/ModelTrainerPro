import os
import sys
from typing import Dict, Any, List, Optional
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QPushButton, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QGroupBox, QScrollArea, QWidget, QSizePolicy
)
from PyQt6.QtCore import Qt

from models.model_utils import get_model_params

class ModelConfigDialog(QDialog):
    """Диалоговое окно для настройки параметров модели."""
    
    def __init__(self, model_type: str,
