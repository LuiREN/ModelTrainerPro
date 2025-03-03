import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QPushButton, QLabel, QComboBox, QFileDialog, QMessageBox,
    QTableView, QSplitter, QTabWidget, QTextEdit, QGroupBox,
    QLineEdit, QCheckBox, QProgressBar, QDialog, QScrollArea,
    QFrame, QSpinBox, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, QModelIndex, QAbstractTableModel, QSize, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont, QIcon

from ui.style import get_stylesheet
from ui.model_config import ModelConfigDialog
from utils.data_loader import DataLoader
from models.model_trainer import ModelTrainer
from utils.report_generator import ReportGenerator
from models.model_utils import load_model, format_metrics

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

    def rowCount(self, parent: QModelIndex = None) -> int:
        """Возвращает количество строк в модели."""
        return len(self._data)

    def columnCount(self, parent: QModelIndex = None) -> int:
        """Возвращает количество столбцов в модели."""
        return len(self._data.columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        """Возвращает данные для отображения."""
        if index.isValid():
            if role == Qt.ItemDataRole.DisplayRole:
                value = self._data.iloc[index.row(), index.column()]
                # Преобразуем в строку с учетом NaN
                if pd.isna(value):
                    return ""
                elif isinstance(value, float):
                    return f"{value:.4f}"
                return str(value)
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        """Возвращает данные заголовков."""
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._data.columns[section])
            elif orientation == Qt.Orientation.Vertical:
                return str(self._data.index[section])
        return None

class MainWindow(QMainWindow):
    """Главное окно приложения ModelTrainerPro."""
    
    def __init__(self):
        """Инициализация главного окна."""
        super().__init__()
        
        # Инициализация компонентов
        self.data_loader = DataLoader()
        self.model_trainer = ModelTrainer()
        self.report_generator = ReportGenerator()
        
        # Переменные для хранения данных
        self.data = None
        self.trained_model = None
        self.model_results = None
        
        # Настройка интерфейса
        self.setWindowTitle("ModelTrainerPro")
        self.setMinimumSize(1000, 700)
        
        # Применение стилей
        self.setStyleSheet(get_stylesheet())
        
        # Создание центрального виджета
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Главный layout
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Создание заголовка
        self.create_header()
        
        # Создание панелей инструментов
        self.create_toolbar()
        
        # Создание основного содержимого
        self.create_main_content()
        
        # Создание строки состояния
        self.statusBar().showMessage("Готово")
    
    def create_header(self):
        """Создает заголовок приложения."""
        header_layout = QHBoxLayout()
        
        # Заголовок
        header_label = QLabel("ModelTrainerPro")
        header_label.setObjectName("header")
        
        # Добавление заголовка в layout
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        
        # Добавление layout в главный layout
        self.main_layout.addLayout(header_layout)
    
    def create_toolbar(self):
        """Создает панель инструментов."""
        toolbar_layout = QHBoxLayout()
        
        # Кнопка загрузки данных
        self.load_data_btn = QPushButton("Загрузить данные")
        self.load_data_btn.clicked.connect(self.load_data)
        
        # Выбор модели
        self.model_type_label = QLabel("Модель:")
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["Random Forest", "KNN Regressor"])
        
        # Кнопка настройки модели
        self.model_config_btn = QPushButton("Настроить модель")
        self.model_config_btn.clicked.connect(self.configure_model)
        self.model_config_btn.setEnabled(False)
        
        # Кнопка обучения модели
        self.train_model_btn = QPushButton("Обучить модель")
        self.train_model_btn.clicked.connect(self.train_model)
        self.train_model_btn.setEnabled(False)
        
        # Кнопка сохранения модели
        self.save_model_btn = QPushButton("Сохранить модель")
        self.save_model_btn.clicked.connect(self.save_model)
        self.save_model_btn.setEnabled(False)
        
        # Кнопка загрузки модели
        self.load_model_btn = QPushButton("Загрузить модель")
        self.load_model_btn.clicked.connect(self.load_existing_model)
        
        # Добавление виджетов в layout
        toolbar_layout.addWidget(self.load_data_btn)
        toolbar_layout.addWidget(self.model_type_label)
        toolbar_layout.addWidget(self.model_type_combo)
        toolbar_layout.addWidget(self.model_config_btn)
        toolbar_layout.addWidget(self.train_model_btn)
        toolbar_layout.addWidget(self.save_model_btn)
        toolbar_layout.addWidget(self.load_model_btn)
        
        # Добавление layout в главный layout
        self.main_layout.addLayout(toolbar_layout)
    
    def create_main_content(self):
        """Создает основное содержимое окна."""
        # Сплиттер для разделения на две части
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Верхняя часть - данные
        self.data_frame = QFrame()
        self.data_layout = QVBoxLayout(self.data_frame)
        
        # Заголовок данных
        self.data_header = QLabel("Данные")
        self.data_header.setObjectName("subheader")
        self.data_layout.addWidget(self.data_header)
        
        # Содержимое данных
        self.data_table = QTableView()
        self.data_table.setAlternatingRowColors(True)
        self.data_layout.addWidget(self.data_table)
        
        # Нижняя часть - вкладки
        self.tabs_frame = QFrame()
        self.tabs_layout = QVBoxLayout(self.tabs_frame)
        
        # Создание вкладок
        self.tabs = QTabWidget()
        
        # Вкладка "Конфигурация"
        self.config_tab = QWidget()
        self.config_layout = QVBoxLayout(self.config_tab)
        
        # Группа выбора целевого столбца
        self.target_group = QGroupBox("Целевой столбец")
        self.target_layout = QVBoxLayout(self.target_group)
        
        self.target_combo = QComboBox()
        self.target_layout.addWidget(self.target_combo)
        
        self.config_layout.addWidget(self.target_group)
        
        # Группа разделения данных
        self.split_group = QGroupBox("Разделение данных")
        self.split_layout = QGridLayout(self.split_group)
        
        self.split_label = QLabel("Доля тестовых данных:")
        self.split_spin = QDoubleSpinBox()
        self.split_spin.setRange(0.1, 0.5)
        self.split_spin.setSingleStep(0.05)
        self.split_spin.setValue(0.2)
        
        self.random_state_label = QLabel("Random state:")
        self.random_state_spin = QSpinBox()
        self.random_state_spin.setRange(0, 100)
        self.random_state_spin.setValue(42)
        
        self.scale_check = QCheckBox("Масштабировать данные")
        self.scale_check.setChecked(True)
        
        self.split_layout.addWidget(self.split_label, 0, 0)
        self.split_layout.addWidget(self.split_spin, 0, 1)
        self.split_layout.addWidget(self.random_state_label, 1, 0)
        self.split_layout.addWidget(self.random_state_spin, 1, 1)
        self.split_layout.addWidget(self.scale_check, 2, 0, 1, 2)
        
        self.config_layout.addWidget(self.split_group)
        self.config_layout.addStretch()
        
        # Вкладка "Параметры модели"
        self.params_tab = QWidget()
        self.params_layout = QVBoxLayout(self.params_tab)
        
        self.params_scroll = QScrollArea()
        self.params_scroll.setWidgetResizable(True)
        self.params_content = QWidget()
        self.params_content_layout = QVBoxLayout(self.params_content)
        
        self.params_scroll.setWidget(self.params_content)
        self.params_layout.addWidget(self.params_scroll)
        
        # Вкладка "Результаты"
        self.results_tab = QWidget()
        self.results_layout = QVBoxLayout(self.results_tab)
        
        self.metrics_group = QGroupBox("Метрики")
        self.metrics_layout = QVBoxLayout(self.metrics_group)
        
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_layout.addWidget(self.metrics_text)
        
        self.results_layout.addWidget(self.metrics_group)
        
        self.feature_importance_group = QGroupBox("Важность признаков")
        self.feature_importance_layout = QVBoxLayout(self.feature_importance_group)
        
        self.feature_importance_text = QTextEdit()
        self.feature_importance_text.setReadOnly(True)
        self.feature_importance_layout.addWidget(self.feature_importance_text)
        
        self.results_layout.addWidget(self.feature_importance_group)
        
        # Вкладка "Отчеты"
        self.reports_tab = QWidget()
        self.reports_layout = QVBoxLayout(self.reports_tab)
        
        self.generate_report_btn = QPushButton("Создать отчет")
        self.generate_report_btn.clicked.connect(self.generate_report)
        
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        
        self.reports_layout.addWidget(self.generate_report_btn)
        self.reports_layout.addWidget(self.report_text)
        
        # Добавление вкладок
        self.tabs.addTab(self.config_tab, "Конфигурация")
        self.tabs.addTab(self.params_tab, "Параметры модели")
        self.tabs.addTab(self.results_tab, "Результаты")
        self.tabs.addTab(self.reports_tab, "Отчеты")
        
        self.tabs_layout.addWidget(self.tabs)
        
        # Добавление фреймов в сплиттер
        self.splitter.addWidget(self.data_frame)
        self.splitter.addWidget(self.tabs_frame)
        
        # Установка размеров сплиттера
        self.splitter.setSizes([300, 400])
        
        # Добавление сплиттера в главный layout
        self.main_layout.addWidget(self.splitter)
    
    def load_data(self):
        """Загружает данные из CSV файла."""
        # Открываем диалог выбора файла
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл с данными", "", "CSV файлы (*.csv);;Excel файлы (*.xlsx *.xls)"
        )
        
        if file_path:
            try:
                # Загрузка данных
                self.data = self.data_loader.load_data(file_path)
                
                # Отображение данных в таблице
                model = PandasModel(self.data)
                self.data_table.setModel(model)
                
                # Обновление списка столбцов
                self.update_column_list()
                
                # Активация кнопок
                self.model_config_btn.setEnabled(True)
                self.train_model_btn.setEnabled(True)
                
                # Выводим сообщение в строке состояния
                self.statusBar().showMessage(f"Данные загружены: {os.path.basename(file_path)}")
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка загрузки данных", str(e))
    
    def update_column_list(self):
        """Обновляет список столбцов в выпадающих списках."""
        if self.data is not None:
            # Очищаем список
            self.target_combo.clear()
            
            # Добавляем все столбцы
            for column in self.data.columns:
                if column != 'Дата':  # Исключаем столбец с датой
                    self.target_combo.addItem(column)
    
    def configure_model(self):
        """Открывает диалог настройки модели."""
        if self.data is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите данные")
            return
        
        # Определяем тип модели
        model_type = self.get_model_type_key()
        
        # Создаем диалог
        dialog = ModelConfigDialog(model_type, self)
        
        # Если пользователь подтвердил настройки
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Получаем параметры
            self.model_params = dialog.get_params()
            
            # Выводим информацию о параметрах
            params_text = "Выбранные параметры:\n\n"
            for param, value in self.model_params.items():
                params_text += f"{param}: {value}\n"
            
            # Переключаемся на вкладку параметров
            self.tabs.setCurrentIndex(1)
            
            # Очищаем существующие параметры
            for i in reversed(range(self.params_content_layout.count())):
                self.params_content_layout.itemAt(i).widget().deleteLater()
            
            # Добавляем параметры
            for param, value in self.model_params.items():
                param_layout = QHBoxLayout()
                param_label = QLabel(f"{param}:")
                param_value = QLabel(f"{value}")
                
                param_layout.addWidget(param_label)
                param_layout.addWidget(param_value)
                param_layout.addStretch()
                
                self.params_content_layout.addLayout(param_layout)
            
            # Добавляем растягивающийся элемент в конец
            self.params_content_layout.addStretch()
            
            # Обновляем строку состояния
            self.statusBar().showMessage("Параметры модели настроены")
    
    def get_model_type_key(self) -> str:
        """
        Преобразует значение из комбобокса в ключ для модели.
        
        Returns:
            str: Ключ модели ('random_forest' или 'knn')
        """
        model_type = self.model_type_combo.currentText()
        
        if model_type == "Random Forest":
            return "random_forest"
        elif model_type == "KNN Regressor":
            return "knn"
        else:
            return "unknown"
    
    def train_model(self):
        """Обучает модель с заданными параметрами."""
        if self.data is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите данные")
            return
        
        # Получаем целевой столбец
        target_column = self.target_combo.currentText()
        
        if not target_column:
            QMessageBox.warning(self, "Внимание", "Выберите целевой столбец")
            return
        
        try:
            # Подготавливаем данные
            test_size = self.split_spin.value()
            random_state = self.random_state_spin.value()
            scale_data = self.scale_check.isChecked()
            
            data_info = self.data_loader.prepare_data(
                target_column, test_size, random_state, scale_data
            )
            
            # Если параметры модели не заданы через диалог, используем значения по умолчанию
            if not hasattr(self, 'model_params'):
                self.model_params = {}
            
            # Получаем тип модели
            model_type = self.get_model_type_key()
            
            # Обучаем модель
            self.statusBar().showMessage("Обучение модели...")
            
            # Получаем обучающие и тестовые данные
            X_train = data_info['X_train']
            X_test = data_info['X_test']
            y_train = data_info['y_train']
            y_test = data_info['y_test']
            features = data_info['features']
            
            # Обучаем модель
            self.model_results = self.model_trainer.train_model(
                model_type, X_train, y_train, X_test, y_test, features, self.model_params
            )
            
            # Отображаем результаты
            self.display_results()
            
            # Активируем кнопку сохранения
            self.save_model_btn.setEnabled(True)
            
            # Обновляем строку состояния
            self.statusBar().showMessage("Модель успешно обучена")
            
            # Переключаемся на вкладку результатов
            self.tabs.setCurrentIndex(2)
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка обучения модели", str(e))
    
    def display_results(self):
        """Отображает результаты обучения модели."""
        if self.model_results:
            # Метрики
            metrics_text = "Метрики качества модели:\n\n"
            metrics_text += format_metrics(self.model_results['metrics'])
            metrics_text += f"\n\nВремя обучения: {self.model_results['training_time']:.2f} секунд"
            
            self.metrics_text.setText(metrics_text)
            
            # Важность признаков (если доступно)
            if 'feature_importance' in self.model_results and self.model_results['feature_importance']:
                importance_text = "Важность признаков:\n\n"
                
                # Сортируем признаки по важности
                sorted_importance = sorted(
                    self.model_results['feature_importance'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                for feature, importance in sorted_importance:
                    importance_text += f"{feature}: {importance:.4f}\n"
                
                self.feature_importance_text.setText(importance_text)
            else:
                self.feature_importance_text.setText("Информация о важности признаков недоступна для данного типа модели")
    
    def save_model(self):
        """Сохраняет обученную модель."""
        if not self.model_trainer.trained:
            QMessageBox.warning(self, "Внимание", "Сначала обучите модель")
            return
        
        # Открываем диалог сохранения
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить модель", "saved_models/model.joblib", "Joblib файлы (*.joblib)"
        )
        
        if file_path:
            try:
                # Сохраняем модель
                saved_path = self.model_trainer.save(file_path)
                
                # Выводим сообщение
                QMessageBox.information(
                    self, "Модель сохранена", f"Модель успешно сохранена в файл:\n{saved_path}"
                )
                
                # Обновляем строку состояния
                self.statusBar().showMessage(f"Модель сохранена: {os.path.basename(saved_path)}")
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка сохранения модели", str(e))
    
    def load_existing_model(self):
        """Загружает существующую модель."""
        # Открываем диалог выбора файла
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить модель", "saved_models", "Joblib файлы (*.joblib)"
        )
        
        if file_path:
            try:
                from models.model_utils import load_model
                
                # Загружаем модель и информацию о ней
                model, model_info = load_model(file_path)
                
                # Устанавливаем модель в тренер
                self.model_trainer.model = model
                self.model_trainer.model_type = model_info['model_type']
                self.model_trainer.training_time = model_info['training_time']
                self.model_trainer.metrics = model_info['metrics']
                self.model_trainer.best_params = model_info['params']
                self.model_trainer.feature_names = model_info['feature_names']
                self.model_trainer.trained = True
                
                # Создаем результаты для отображения
                self.model_results = {
                    'model_type': model_info['model_type'],
                    'training_time': model_info['training_time'],
                    'metrics': model_info['metrics'],
                    'params': model_info['params']
                }
                
                # Добавляем важность признаков, если доступно
                if hasattr(model, 'feature_importances_'):
                    self.model_results['feature_importance'] = {
                        model_info['feature_names'][i]: model.feature_importances_[i]
                        for i in range(len(model_info['feature_names']))
                    }
                
                # Отображаем результаты
                self.display_results()
                
                # Активируем кнопку сохранения
                self.save_model_btn.setEnabled(True)
                
                # Переключаемся на вкладку результатов
                self.tabs.setCurrentIndex(2)
                
                # Обновляем строку состояния
                self.statusBar().showMessage(f"Модель загружена: {os.path.basename(file_path)}")
                
                # Выводим информацию
                QMessageBox.information(
                    self, "Модель загружена", 
                    f"Модель успешно загружена из файла:\n{file_path}\n\n"
                    f"Тип модели: {model_info['model_type']}\n"
                    f"Дата обучения: {model_info.get('trained_date', 'Неизвестно')}"
                )
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка загрузки модели", str(e))
    
    def generate_report(self):
        """Генерирует отчет о результатах обучения модели."""
        if not self.model_trainer.trained:
            QMessageBox.warning(self, "Внимание", "Сначала обучите модель")
            return
        
        try:
            # Создаем информацию о наборе данных
            dataset_info = {
                'total_rows': len(self.data) if self.data is not None else 0,
                'features': self.data_loader.features,
                'target': self.data_loader.target_column
            }
            
            # Генерируем отчет
            model_name = f"{self.model_results['model_type']}_model"
            report_path = self.report_generator.generate_training_report(
                self.model_results, model_name, dataset_info
            )
            
            # Читаем содержимое отчета
            with open(report_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
            
            # Отображаем отчет
            self.report_text.setText(report_content)
            
            # Выводим сообщение
            QMessageBox.information(
                self, "Отчет создан", f"Отчет успешно создан и сохранен в файл:\n{report_path}"
            )
            
            # Обновляем строку состояния
            self.statusBar().showMessage(f"Отчет создан: {os.path.basename(report_path)}")
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка создания отчета", str(e))