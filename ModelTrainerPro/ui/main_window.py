import os
import sys
from typing import Optional, Dict, Any, List, Tuple

from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QLabel, QPushButton, QComboBox, QFileDialog,
                           QTableView, QTabWidget, QGroupBox, QFormLayout,
                           QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
                           QMessageBox, QTextEdit, QSplitter)
from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PyQt6.QtGui import QColor, QPalette

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_trainer import ModelTrainer
from utils.data_processor import DataProcessor
from utils.file_manager import ModelManager


class DataTableModel(QAbstractTableModel):
    """
    Модель для отображения DataFrame в Qt TableView.
    
    Атрибуты:
        data (pandas.DataFrame): DataFrame для отображения.
    """
    def __init__(self, data):
        """
        Инициализация модели с данными.
        
        Аргументы:
            data (pandas.DataFrame): DataFrame для отображения.
        """
        super().__init__()
        self._data = data

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """
        Возвращает количество строк в модели.
        
        Аргументы:
            parent (QModelIndex): Родительский индекс.
            
        Возвращает:
            int: Количество строк.
        """
        return len(self._data)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """
        Возвращает количество столбцов в модели.
        
        Аргументы:
            parent (QModelIndex): Родительский индекс.
            
        Возвращает:
            int: Количество столбцов.
        """
        return len(self._data.columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        """
        Возвращает данные для указанной роли по заданному индексу.
        
        Аргументы:
            index (QModelIndex): Индекс для получения данных.
            role (int): Роль для получения данных.
            
        Возвращает:
            Any: Запрошенные данные.
        """
        if not index.isValid():
            return None
            
        if role == Qt.ItemDataRole.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)
        
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, 
                  role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        """
        Возвращает данные заголовка для указанной секции и ориентации.
        
        Аргументы:
            section (int): Индекс секции (строки/столбца).
            orientation (Qt.Orientation): Ориентация (горизонтальная/вертикальная).
            role (int): Роль для получения данных.
            
        Возвращает:
            Any: Запрошенные данные заголовка.
        """
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._data.columns[section])
            else:
                return str(section + 1)
        return None


class MainWindow(QMainWindow):
    """
    Главное окно приложения ModelTrainerPro.
    
    Это окно предоставляет пользовательский интерфейс для загрузки данных, 
    выбора и обучения моделей, настройки гиперпараметров, 
    а также сохранения/загрузки обученных моделей.
    """
    def __init__(self):
        """Инициализирует главное окно и настраивает пользовательский интерфейс."""
        super().__init__()
        
        self.data_processor = DataProcessor()
        self.model_trainer = ModelTrainer()
        self.model_manager = ModelManager()

        self.setWindowTitle("ModelTrainerPro")
        self.setMinimumSize(1000, 700)
        self.setup_ui()
        self.apply_styles()
    
    def apply_styles(self) -> None:
        """Применяет пользовательские стили к приложению."""
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(245, 245, 255))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(50, 50, 50))
        palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(240, 235, 255))
        palette.setColor(QPalette.ColorRole.Button, QColor(110, 80, 160))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(130, 100, 180))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        self.setPalette(palette)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F5F5FF;
            }
            QTabWidget::pane {
                border: 1px solid #8A65B3;
                border-top-width: 0px;
                background-color: #FFFFFF;
            }
            QTabBar::tab {
                background-color: #B79FD8;
                color: white;
                padding: 8px 12px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #8A65B3;
            }
            QPushButton {
                background-color: #6E50A0;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #8A65B3;
            }
            QPushButton:pressed {
                background-color: #5A3F87;
            }
            QGroupBox {
                border: 1px solid #B79FD8;
                border-radius: 5px;
                margin-top: 1ex;
                font-weight: bold;
                color: #3A2750;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
            }
            QComboBox {
                border: 1px solid #B79FD8;
                border-radius: 3px;
                padding: 5px;
            }
            QComboBox::drop-down {
                border-left: 1px solid #B79FD8;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                border: 1px solid #B79FD8;
                border-radius: 3px;
                padding: 5px;
            }
            QTextEdit {
                border: 1px solid #B79FD8;
                border-radius: 3px;
            }
        """)
    
    def setup_ui(self) -> None:
        """Настраивает компоненты пользовательского интерфейса."""

        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        data_group = QGroupBox("Загрузка данных")
        data_layout = QHBoxLayout(data_group)
        
        self.load_data_btn = QPushButton("Загрузить CSV")
        self.load_data_btn.clicked.connect(self.load_data)
        
        self.data_path_label = QLabel("Данные не загружены")
        
        data_layout.addWidget(self.load_data_btn)
        data_layout.addWidget(self.data_path_label, 1)

        splitter = QSplitter(Qt.Orientation.Vertical)

        data_preview_group = QGroupBox("Просмотр данных")
        data_preview_layout = QVBoxLayout(data_preview_group)
        
        self.data_table = QTableView()
        self.data_table.setAlternatingRowColors(True)
        data_preview_layout.addWidget(self.data_table)
        
        tabs = QTabWidget()

        training_tab = QWidget()
        training_layout = QVBoxLayout(training_tab)

        model_selection_group = QGroupBox("Выбор модели")
        model_selection_layout = QFormLayout(model_selection_group)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Random Forest", "KNN Regressor"])
        self.model_combo.currentIndexChanged.connect(self.update_hyperparameters)
        
        model_selection_layout.addRow("Модель:", self.model_combo)

        self.hyperparams_group = QGroupBox("Гиперпараметры")
        self.hyperparams_layout = QFormLayout(self.hyperparams_group)
  
        buttons_layout = QHBoxLayout()
        
        self.train_btn = QPushButton("Обучить модель")
        self.train_btn.clicked.connect(self.train_model)
        
        self.save_model_btn = QPushButton("Сохранить модель")
        self.save_model_btn.clicked.connect(self.save_model)
        self.save_model_btn.setEnabled(False)
        
        self.load_model_btn = QPushButton("Загрузить модель")
        self.load_model_btn.clicked.connect(self.load_model)
        
        buttons_layout.addWidget(self.train_btn)
        buttons_layout.addWidget(self.save_model_btn)
        buttons_layout.addWidget(self.load_model_btn)
        
        training_layout.addWidget(model_selection_group)
        training_layout.addWidget(self.hyperparams_group)
        training_layout.addLayout(buttons_layout)
   
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        
        results_layout.addWidget(self.results_text)
  
        tabs.addTab(training_tab, "Обучение")
        tabs.addTab(results_tab, "Результаты")
        
        splitter.addWidget(data_preview_group)
        splitter.addWidget(tabs)
        
        main_layout.addWidget(data_group)
        main_layout.addWidget(splitter, 1)
        
        self.setCentralWidget(central_widget)
        
        self.update_hyperparameters()
        
        self.hyperparameter_widgets = {}

    def update_hyperparameters(self) -> None:
        """Обновляет пользовательский интерфейс гиперпараметров на основе выбранной модели."""
        while self.hyperparams_layout.rowCount() > 0:
            self.hyperparams_layout.removeRow(0)
        
        self.hyperparameter_widgets = {}
        
        model_name = self.model_combo.currentText()
        
        if model_name == "Random Forest":
            n_estimators_spin = QSpinBox()
            n_estimators_spin.setRange(1, 1000)
            n_estimators_spin.setValue(100)
            n_estimators_spin.setSingleStep(10)
            self.hyperparams_layout.addRow("Количество деревьев:", n_estimators_spin)
            self.hyperparameter_widgets["n_estimators"] = n_estimators_spin
            
            max_depth_spin = QSpinBox()
            max_depth_spin.setRange(1, 100)
            max_depth_spin.setValue(None)
            max_depth_spin.setSpecialValueText("None")
            self.hyperparams_layout.addRow("Максимальная глубина:", max_depth_spin)
            self.hyperparameter_widgets["max_depth"] = max_depth_spin
            
            min_samples_split_spin = QSpinBox()
            min_samples_split_spin.setRange(2, 20)
            min_samples_split_spin.setValue(2)
            self.hyperparams_layout.addRow("Минимальное кол-во для разделения:", min_samples_split_spin)
            self.hyperparameter_widgets["min_samples_split"] = min_samples_split_spin
            
            random_state_spin = QSpinBox()
            random_state_spin.setRange(0, 100)
            random_state_spin.setValue(42)
            self.hyperparams_layout.addRow("Random state:", random_state_spin)
            self.hyperparameter_widgets["random_state"] = random_state_spin
        
        elif model_name == "KNN Regressor":

            n_neighbors_spin = QSpinBox()
            n_neighbors_spin.setRange(1, 20)
            n_neighbors_spin.setValue(5)
            self.hyperparams_layout.addRow("Количество соседей:", n_neighbors_spin)
            self.hyperparameter_widgets["n_neighbors"] = n_neighbors_spin
            
            weights_combo = QComboBox()
            weights_combo.addItems(["uniform", "distance"])
            self.hyperparams_layout.addRow("Веса:", weights_combo)
            self.hyperparameter_widgets["weights"] = weights_combo
            
            algorithm_combo = QComboBox()
            algorithm_combo.addItems(["auto", "ball_tree", "kd_tree", "brute"])
            self.hyperparams_layout.addRow("Алгоритм:", algorithm_combo)
            self.hyperparameter_widgets["algorithm"] = algorithm_combo
            
            p_spin = QSpinBox()
            p_spin.setRange(1, 2)
            p_spin.setValue(2)
            self.hyperparams_layout.addRow("Параметр Минковского:", p_spin)
            self.hyperparameter_widgets["p"] = p_spin

    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Получает текущие значения гиперпараметров из пользовательского интерфейса.
        
        Возвращает:
            Dict[str, Any]: Словарь с названиями и значениями гиперпараметров.
        """
        params = {}
        
        for name, widget in self.hyperparameter_widgets.items():
            if isinstance(widget, QComboBox):
                params[name] = widget.currentText()
            elif isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                if name == "max_depth" and widget.value() == widget.minimum():
                    params[name] = None
                else:
                    params[name] = widget.value()
            elif isinstance(widget, QCheckBox):
                params[name] = widget.isChecked()
            elif isinstance(widget, QLineEdit):
                params[name] = widget.text()
                
        return params

    def load_data(self) -> None:
        """Загружает и обрабатывает данные из CSV-файла."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            self.data_processor.load_data(file_path)

            self.data_path_label.setText(file_path)
           
            model = DataTableModel(self.data_processor.data)
            self.data_table.setModel(model)
          
            self.data_table.resizeColumnsToContents()
            
            self.train_btn.setEnabled(True)
            
            QMessageBox.information(self, "Успешно", "Данные успешно загружены")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при загрузке данных: {str(e)}")

    def train_model(self) -> None:
        """Обучает выбранную модель с текущими гиперпараметрами."""
        if self.data_processor.data is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите данные")
            return
            
        try:
         
            model_name = self.model_combo.currentText()
            hyperparams = self.get_hyperparameters()
      
            X_train, X_test, y_train, y_test = self.data_processor.prepare_data_for_training()
      
            metrics, model = self.model_trainer.train(
                model_name, X_train, X_test, y_train, y_test, hyperparams
            )
      
            self.results_text.clear()
            self.results_text.append(f"<h2>Результаты обучения модели {model_name}</h2>")
            self.results_text.append("<h3>Метрики:</h3>")
            for metric_name, value in metrics.items():
                self.results_text.append(f"<p><b>{metric_name}:</b> {value:.4f}</p>")
                
            self.results_text.append("<h3>Гиперпараметры:</h3>")
            for param_name, value in hyperparams.items():
                self.results_text.append(f"<p><b>{param_name}:</b> {value}</p>")
  
            result_file = self.model_trainer.save_results_to_file(
                model_name, hyperparams, metrics
            )
            self.results_text.append(f"<p>Результаты сохранены в файл: {result_file}</p>")
            
            self.save_model_btn.setEnabled(True)
            
            QMessageBox.information(
                self, 
                "Обучение завершено", 
                f"Модель {model_name} успешно обучена.\nMAE: {metrics['MAE']:.4f}\nMSE: {metrics['MSE']:.4f}\nR²: {metrics['R2']:.4f}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при обучении модели: {str(e)}")

    def save_model(self) -> None:
        """Сохраняет обученную модель в файл."""
        if not self.model_trainer.model:
            QMessageBox.warning(self, "Предупреждение", "Сначала обучите модель")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить модель", "", "Model Files (*.pkl);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            self.model_manager.save_model(self.model_trainer.model, file_path)
            QMessageBox.information(self, "Успешно", "Модель успешно сохранена")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при сохранении модели: {str(e)}")

    def load_model(self) -> None:
        """Загружает обученную модель из файла."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить модель", "", "Model Files (*.pkl);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            model = self.model_manager.load_model(file_path)
            self.model_trainer.model = model
            
            self.save_model_btn.setEnabled(True)
            
            model_info = str(model)
            model_name = "Random Forest" if "RandomForest" in model_info else "KNN Regressor"
         
            self.model_combo.setCurrentText(model_name)
            
            QMessageBox.information(
                self, 
                "Успешно", 
                f"Модель {model_name} успешно загружена"
            )
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка при загрузке модели: {str(e)}")