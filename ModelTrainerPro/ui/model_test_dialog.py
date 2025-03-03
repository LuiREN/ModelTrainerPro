import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QPushButton, QLabel, QComboBox, QFileDialog, QMessageBox,
    QTableView, QScrollArea, QWidget, QLineEdit, QGroupBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from utils.data_loader import DataLoader

class PandasModel(QAbstractTableModel):
    """Модель для отображения данных pandas в QTableView."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Инициализация модели данных.
        
        Args:
            data (pd.DataFrame): Данные для отображения
        """
        super().__init__()
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data)

    def columnCount(self, parent=None):
        return len(self._data.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid():
            if role == Qt.ItemDataRole.DisplayRole:
                value = self._data.iloc[index.row(), index.column()]
                if pd.isna(value):
                    return ""
                elif isinstance(value, float):
                    return f"{value:.4f}"
                return str(value)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._data.columns[section])
            elif orientation == Qt.Orientation.Vertical:
                return str(self._data.index[section])
        return None

class ModelTestDialog(QDialog):
    """Диалоговое окно для тестирования обученной модели."""
    
    def __init__(self, model, model_info, data_loader, parent=None):
        """
        Инициализация диалога тестирования модели.
        
        Args:
            model: Обученная модель
            model_info (Dict[str, Any]): Информация о модели
            data_loader (DataLoader): Загрузчик данных
            parent: Родительский виджет
        """
        super().__init__(parent)
        
        self.model = model
        self.model_info = model_info
        self.data_loader = data_loader
        self.feature_names = model_info['feature_names']
        self.test_data = None
        
        # Настройка диалога
        self.setWindowTitle(f"Тестирование модели {model_info['model_type']}")
        self.setMinimumSize(800, 600)
        
        # Основной layout
        self.main_layout = QVBoxLayout(self)
        
        # Заголовок
        header_label = QLabel(f"Тестирование модели {model_info['model_type']}")
        header_label.setObjectName("header")
        self.main_layout.addWidget(header_label)
        
        # Информация о модели
        model_info_group = QGroupBox("Информация о модели")
        model_info_layout = QGridLayout(model_info_group)
        
        model_info_layout.addWidget(QLabel("Тип модели:"), 0, 0)
        model_info_layout.addWidget(QLabel(model_info['model_type']), 0, 1)
        
        model_info_layout.addWidget(QLabel("Дата обучения:"), 1, 0)
        model_info_layout.addWidget(QLabel(model_info.get('trained_date', 'Неизвестно')), 1, 1)
        
        model_info_layout.addWidget(QLabel("Целевая переменная:"), 2, 0)
        model_info_layout.addWidget(QLabel(model_info.get('target', 'Неизвестно')), 2, 1)
        
        # Метрики
        if 'metrics' in model_info:
            model_info_layout.addWidget(QLabel("Метрики на тестовых данных:"), 3, 0)
            
            metrics_text = f"MAE: {model_info['metrics'].get('mae', 0):.4f}, "
            metrics_text += f"RMSE: {model_info['metrics'].get('rmse', 0):.4f}, "
            metrics_text += f"R²: {model_info['metrics'].get('r2', 0):.4f}"
            
            model_info_layout.addWidget(QLabel(metrics_text), 3, 1)
        
        self.main_layout.addWidget(model_info_group)
        
        # Группа для загрузки тестовых данных
        data_group = QGroupBox("Тестовые данные")
        data_layout = QVBoxLayout(data_group)
        
        # Кнопки для загрузки данных
        data_buttons_layout = QHBoxLayout()
        
        self.load_data_btn = QPushButton("Загрузить данные")
        self.load_data_btn.clicked.connect(self.load_test_data)
        
        data_buttons_layout.addWidget(self.load_data_btn)
        data_buttons_layout.addStretch()
        
        data_layout.addLayout(data_buttons_layout)
        
        # Таблица для тестовых данных
        self.data_table = QTableView()
        self.data_table.setAlternatingRowColors(True)
        data_layout.addWidget(self.data_table)
        
        self.main_layout.addWidget(data_group)
        
        # Группа для ручного ввода
        manual_group = QGroupBox("Ручной ввод")
        manual_layout = QVBoxLayout(manual_group)
        
        # Поля для ввода значений признаков
        manual_scroll = QScrollArea()
        manual_scroll.setWidgetResizable(True)
        manual_content = QWidget()
        self.manual_content_layout = QGridLayout(manual_content)
        
        # Создаем поля ввода для каждого признака
        self.feature_inputs = {}
        
        for i, feature in enumerate(self.feature_names):
            row = i // 2
            col = (i % 2) * 2
            
            label = QLabel(f"{feature}:")
            input_field = QLineEdit()
            input_field.setPlaceholderText("Введите значение")
            
            self.manual_content_layout.addWidget(label, row, col)
            self.manual_content_layout.addWidget(input_field, row, col + 1)
            
            self.feature_inputs[feature] = input_field
        
        manual_scroll.setWidget(manual_content)
        manual_layout.addWidget(manual_scroll)
        
        # Кнопка для получения предсказания по ручному вводу
        self.predict_manual_btn = QPushButton("Получить предсказание")
        self.predict_manual_btn.clicked.connect(self.predict_manual)
        manual_layout.addWidget(self.predict_manual_btn)
        
        self.main_layout.addWidget(manual_group)
        
        # Группа для результатов предсказания
        results_group = QGroupBox("Результаты предсказания")
        results_layout = QVBoxLayout(results_group)
        
        # Кнопка для предсказания на загруженных данных
        self.predict_btn = QPushButton("Сделать предсказания на загруженных данных")
        self.predict_btn.clicked.connect(self.predict)
        self.predict_btn.setEnabled(False)
        results_layout.addWidget(self.predict_btn)
        
        # Таблица для результатов
        self.results_table = QTableView()
        self.results_table.setAlternatingRowColors(True)
        results_layout.addWidget(self.results_table)
        
        self.main_layout.addWidget(results_group)
        
        # Кнопки действий
        self.buttons_layout = QHBoxLayout()
        
        self.export_btn = QPushButton("Экспорт результатов")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        
        self.close_button = QPushButton("Закрыть")
        self.close_button.clicked.connect(self.accept)
        
        self.buttons_layout.addWidget(self.export_btn)
        self.buttons_layout.addStretch()
        self.buttons_layout.addWidget(self.close_button)
        
        self.main_layout.addLayout(self.buttons_layout)
        
        # Сохраняем переменную для результатов предсказания
        self.predictions_df = None
    
    def load_test_data(self):
        """Загружает тестовые данные из файла."""
        # Открываем диалог выбора файла
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл с тестовыми данными", "", "CSV файлы (*.csv);;Excel файлы (*.xlsx *.xls)"
        )
        
        if file_path:
            try:
                # Загрузка данных
                self.test_data = pd.read_csv(file_path) if file_path.endswith('.csv') \
                    else pd.read_excel(file_path)
                
                # Проверяем наличие необходимых признаков
                missing_features = [f for f in self.feature_names if f not in self.test_data.columns]
                
                if missing_features:
                    QMessageBox.warning(
                        self, 
                        "Отсутствуют признаки", 
                        f"В загруженных данных отсутствуют следующие признаки: {', '.join(missing_features)}"
                    )
                    return
                
                # Отображение данных в таблице
                model = PandasModel(self.test_data)
                self.data_table.setModel(model)
                
                # Активируем кнопку предсказания
                self.predict_btn.setEnabled(True)
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка загрузки данных", str(e))
    
    def predict(self):
        """Делает предсказания на загруженных данных."""
        if self.test_data is None:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите тестовые данные")
            return
        
        try:
            # Выбираем только нужные признаки
            X = self.test_data[self.feature_names]
            
            # Преобразуем данные, если необходимо
            if 'scaler' in self.model_info and self.model_info['scaler'] is not None:
                X = self.model_info['scaler'].transform(X)
            
            # Делаем предсказания
            predictions = self.model.predict(X)
            
            # Создаем DataFrame с результатами
            self.predictions_df = self.test_data.copy()
            self.predictions_df['Предсказание'] = predictions
            
            # Если знаем целевую переменную и она есть в данных, вычисляем ошибку
            target = self.model_info.get('target')
            if target and target in self.test_data.columns:
                self.predictions_df['Фактическое значение'] = self.test_data[target]
                self.predictions_df['Ошибка'] = self.predictions_df['Фактическое значение'] - predictions
            
            # Отображаем результаты
            model = PandasModel(self.predictions_df)
            self.results_table.setModel(model)
            
            # Активируем кнопку экспорта
            self.export_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка предсказания", str(e))
    
    def predict_manual(self):
        """Делает предсказание на основе ручного ввода."""
        try:
            # Собираем данные из полей ввода
            data = {}
            
            for feature, input_field in self.feature_inputs.items():
                value_str = input_field.text().strip()
                
                if not value_str:
                    QMessageBox.warning(
                        self, 
                        "Пропущены значения", 
                        f"Введите значение для признака '{feature}'"
                    )
                    return
                
                try:
                    value = float(value_str)
                    data[feature] = value
                except ValueError:
                    QMessageBox.warning(
                        self, 
                        "Некорректное значение", 
                        f"Значение '{value_str}' для признака '{feature}' не является числом"
                    )
                    return
            
            # Создаем DataFrame с одной строкой
            X = pd.DataFrame([data])
            
            # Преобразуем данные, если необходимо
            if 'scaler' in self.model_info and self.model_info['scaler'] is not None:
                X = self.model_info['scaler'].transform(X)
            
            # Делаем предсказание
            prediction = self.model.predict(X)[0]
            
            # Отображаем результат
            QMessageBox.information(
                self, 
                "Результат предсказания", 
                f"Предсказанное значение: {prediction:.4f}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка предсказания", str(e))
    
    def export_results(self):
        """Экспортирует результаты предсказания в файл."""
        if self.predictions_df is None:
            QMessageBox.warning(self, "Нет результатов", "Сначала сделайте предсказания")
            return
        
        # Открываем диалог сохранения файла
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результаты", "", "CSV файлы (*.csv);;Excel файлы (*.xlsx)"
        )
        
        if file_path:
            try:
                # Сохраняем результаты в выбранный формат
                if file_path.endswith('.csv'):
                    self.predictions_df.to_csv(file_path, index=False)
                else:
                    self.predictions_df.to_excel(file_path, index=False)
                
                QMessageBox.information(
                    self, 
                    "Экспорт успешен", 
                    f"Результаты успешно сохранены в файл:\n{file_path}"
                )
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка экспорта", str(e))
