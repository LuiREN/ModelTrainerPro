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
    
    def __init__(self, model_type: str, parent=None):
        """
        Инициализация диалога настройки модели.
        
        Args:
            model_type (str): Тип модели ('random_forest' или 'knn')
            parent: Родительский виджет
        """
        super().__init__(parent)
        
        self.model_type = model_type
        self.param_widgets = {}
        
        # Получаем доступные параметры для модели
        self.model_params = get_model_params(model_type)
        
        # Настройка диалога
        self.setWindowTitle(f"Настройка параметров модели")
        self.setMinimumWidth(400)
        
        # Основной layout
        self.main_layout = QVBoxLayout(self)
        
        # Создаем скроллируемую область для параметров
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        
        # Содержимое скроллируемой области
        self.scroll_content = QWidget()
        self.params_layout = QVBoxLayout(self.scroll_content)
        
        # Создаем элементы управления для каждого параметра
        self.create_param_controls()
        
        # Устанавливаем содержимое в скроллируемую область
        self.scroll_area.setWidget(self.scroll_content)
        
        # Добавляем скроллируемую область в основной layout
        self.main_layout.addWidget(self.scroll_area)
        
        # Кнопки действий
        self.buttons_layout = QHBoxLayout()
        
        self.ok_button = QPushButton("ОК")
        self.ok_button.clicked.connect(self.accept)
        
        self.cancel_button = QPushButton("Отмена")
        self.cancel_button.clicked.connect(self.reject)
        
        self.buttons_layout.addStretch()
        self.buttons_layout.addWidget(self.ok_button)
        self.buttons_layout.addWidget(self.cancel_button)
        
        self.main_layout.addLayout(self.buttons_layout)
    
    def create_param_controls(self):
        """Создает элементы управления для настройки параметров."""
        for param_name, param_info in self.model_params.items():
            # Создаем группу для параметра
            param_group = QGroupBox(param_info['description'])
            param_layout = QVBoxLayout(param_group)
            
            # Создаем соответствующий элемент управления в зависимости от типа параметра
            if param_info['type'] == 'int':
                widget = QSpinBox()
                widget.setMinimum(param_info.get('min', 0))
                widget.setMaximum(param_info.get('max', 1000))
    
                # Специальная обработка для параметров, где None является значением по умолчанию
                default_value = param_info.get('default')
                if default_value is None and param_name == 'max_depth':
                    # Для max_depth используем 0 как представление None
                    widget.setValue(0)
                    # Добавим галочку или специальную подсказку, что 0 означает "None (без ограничений)"
                else:
                    # Для других параметров используем значение по умолчанию или 0
                    widget.setValue(default_value if default_value is not None else 0)
                
            elif param_info['type'] == 'float':
                widget = QDoubleSpinBox()
                widget.setMinimum(param_info.get('min', 0.0))
                widget.setMaximum(param_info.get('max', 1.0))
                widget.setSingleStep(0.01)
                widget.setValue(param_info.get('default', 0.0))
                
            elif param_info['type'] == 'select':
                widget = QComboBox()
                widget.addItems(param_info.get('options', []))
                default_index = param_info.get('options', []).index(param_info.get('default', '')) \
                    if param_info.get('default', '') in param_info.get('options', []) else 0
                widget.setCurrentIndex(default_index)
                
            elif param_info['type'] == 'bool':
                widget = QCheckBox("Включить")
                widget.setChecked(param_info.get('default', False))
                
            else:
                continue
            
            # Сохраняем виджет
            self.param_widgets[param_name] = widget
            
            # Добавляем виджет в layout группы
            param_layout.addWidget(widget)
            
            # Добавляем группу в основной layout параметров
            self.params_layout.addWidget(param_group)
        
        # Добавляем растягивающийся элемент в конец
        self.params_layout.addStretch()
    
    def get_params(self) -> Dict[str, Any]:
        """
        Возвращает выбранные пользователем параметры.
        
        Returns:
            Dict[str, Any]: Словарь с параметрами модели
        """
        params = {}
        
        for param_name, widget in self.param_widgets.items():
            # Получаем значение в зависимости от типа виджета
            if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                value = widget.value()
            elif isinstance(widget, QComboBox):
                value = widget.currentText()
            elif isinstance(widget, QCheckBox):
                value = widget.isChecked()
            else:
                continue
            
           # Специальная обработка для max_depth в Random Forest
            if param_name == 'max_depth' and isinstance(widget, QSpinBox) and widget.value() == 0:
                value = None
            
            # Добавляем параметр в словарь
            params[param_name] = value
        
        return params