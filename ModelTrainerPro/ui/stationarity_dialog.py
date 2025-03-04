import os
import pandas as pd
from typing import Dict, Any, List, Optional
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QPushButton, QLabel, QComboBox, QTextEdit,
    QGroupBox, QScrollArea, QWidget, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QPixmap, QImage

from utils.stationarity_tests import (
    adf_test, kpss_test, plot_stationarity_tests, 
    format_stationarity_report
)

class StationarityDialog(QDialog):
    """Диалоговое окно для проведения тестов на стационарность."""
    
    def __init__(self, data: pd.DataFrame, parent=None):
        """
        Инициализация диалога тестов стационарности.
        
        Args:
            data (pd.DataFrame): Данные для анализа
            parent: Родительский виджет
        """
        super().__init__(parent)
        
        self.data = data
        
        # Настройка диалога
        self.setWindowTitle("Тесты на стационарность")
        self.setMinimumSize(900, 700)
        
        # Устанавливаем флаг, чтобы окно не блокировало основной интерфейс
        self.setWindowModality(Qt.WindowModality.NonModal)
        
        # Основной layout
        self.main_layout = QVBoxLayout(self)
        
        # Заголовок
        header_label = QLabel("Тесты на стационарность временных рядов")
        header_label.setObjectName("header")
        self.main_layout.addWidget(header_label)
        
        # Группа выбора переменной
        variable_group = QGroupBox("Выбор переменной для анализа")
        variable_layout = QHBoxLayout(variable_group)
        
        self.variable_label = QLabel("Переменная:")
        self.variable_combo = QComboBox()
        
        # Заполняем комбобокс числовыми столбцами данных
        numeric_columns = self.data.select_dtypes(include=['number']).columns.tolist()
        self.variable_combo.addItems(numeric_columns)
        
        self.run_test_btn = QPushButton("Запустить тест")
        self.run_test_btn.clicked.connect(self.run_stationarity_tests)
        
        variable_layout.addWidget(self.variable_label)
        variable_layout.addWidget(self.variable_combo)
        variable_layout.addWidget(self.run_test_btn)
        variable_layout.addStretch()
        
        self.main_layout.addWidget(variable_group)
        
        # Группа результатов
        results_group = QGroupBox("Результаты тестов")
        results_layout = QVBoxLayout(results_group)
        
        # Создаем прокручиваемую область для результатов
        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        
        # Текстовые результаты
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_layout.addWidget(self.results_text)
        
        # Графические результаты
        self.graph_label = QLabel()
        self.graph_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.graph_label.setMinimumHeight(500)
        self.results_layout.addWidget(self.graph_label)
        
        self.results_scroll.setWidget(self.results_widget)
        results_layout.addWidget(self.results_scroll)
        
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
        
        # Сохраняем переменные для результатов
        self.current_results = None
        self.current_image = None
    
    @pyqtSlot()
    def run_stationarity_tests(self):
        """Запускает тесты на стационарность для выбранной переменной."""
        variable = self.variable_combo.currentText()
        
        if not variable:
            QMessageBox.warning(self, "Ошибка", "Выберите переменную для анализа")
            return
        
        try:
            # Получаем данные выбранной переменной
            series = self.data[variable]
            
            # Проверяем, есть ли временной индекс
            if not isinstance(self.data.index, pd.DatetimeIndex):
                # Если в данных есть столбец 'Дата', пробуем использовать его как индекс
                if 'Дата' in self.data.columns:
                    try:
                        series.index = pd.to_datetime(self.data['Дата'])
                    except:
                        # Если не получилось преобразовать, создаем искусственный индекс
                        series.index = pd.date_range(start='2000-01-01', periods=len(series), freq='D')
                else:
                    # Создаем искусственный индекс
                    series.index = pd.date_range(start='2000-01-01', periods=len(series), freq='D')
            
            # Проверяем на наличие пропущенных значений
            if series.isnull().any():
                QMessageBox.warning(
                    self, 
                    "Предупреждение", 
                    "В данных есть пропущенные значения. Они будут исключены из анализа."
                )
            
            # Проводим тесты
            adf_results = adf_test(series)
            kpss_results = kpss_test(series)
            
            # Формируем отчет
            report = format_stationarity_report(variable, adf_results, kpss_results)
            
            # Отображаем текстовые результаты
            self.results_text.setMarkdown(report)
            
            # Генерируем график
            image_base64 = plot_stationarity_tests(series, variable)
            
            # Создаем QImage из данных base64
            import base64
            from PyQt6.QtGui import QImage, QPixmap
            img_data = base64.b64decode(image_base64)
            image = QImage()
            image.loadFromData(img_data)
            pixmap = QPixmap(image)
            
            # Отображаем график
            self.graph_label.setPixmap(pixmap.scaled(
                self.graph_label.width(), 
                self.graph_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            
            # Сохраняем результаты
            self.current_results = {
                'variable': variable,
                'adf_results': adf_results,
                'kpss_results': kpss_results,
                'report': report,
                'image_base64': image_base64
            }
            
            # Активируем кнопку экспорта
            self.export_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка анализа", f"Произошла ошибка при анализе: {str(e)}")
    
    @pyqtSlot()
    def export_results(self):
        """Экспортирует результаты тестов на стационарность в файл."""
        if not self.current_results:
            QMessageBox.warning(self, "Нет результатов", "Сначала проведите тест")
            return
    
        try:
            from PyQt6.QtWidgets import QFileDialog
        
            # Открываем диалог сохранения файла
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить результаты", "", "HTML файлы (*.html);;Markdown файлы (*.md)"
            )
        
            if not file_path:
                return
        
            variable = self.current_results['variable']
            report = self.current_results['report']
            image_base64 = self.current_results['image_base64']
        
            if file_path.endswith('.html'):
                # Формируем HTML-отчет без использования f-строк для проблемной части
                html_header = """
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <title>Результаты теста на стационарность</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; }
                        h1, h2, h3 { color: #483D8B; }
                        .container { max-width: 1000px; margin: 0 auto; }
                        .results { margin-bottom: 30px; }
                        .image { text-align: center; margin: 20px 0; }
                        img { max-width: 100%; }
                        table { border-collapse: collapse; width: 100%; }
                        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                        th { background-color: #f2f2f2; }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>Результаты теста на стационарность</h1>
                        <div class="results">
                """
            
                html_footer = """
                        </div>
                        <div class="image">
                            <img src="data:image/png;base64,{image_base64}" alt="Графики стационарности">
                        </div>
                    </div>
                </body>
                </html>
                """
            
                # Заменяем переносы строк в отчете
                formatted_report = report.replace('\n', '<br>')
            
                # Собираем все части HTML-документа
                html_content = html_header
                html_content += formatted_report
                html_content = html_content.replace('{image_base64}', image_base64)
                html_content += html_footer.replace('{image_base64}', image_base64)
            
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
        
            elif file_path.endswith('.md'):
                # Формируем Markdown-отчет
                md_header = "# Результаты теста на стационарность для " + variable + "\n\n"
                md_content = md_header + report + "\n\n## Графики\n\n"
                md_content += "![Графики стационарности](data:image/png;base64," + image_base64 + ")\n"
            
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(md_content)
        
            QMessageBox.information(
                self, 
                "Экспорт успешен", 
                f"Результаты успешно сохранены в файл:\n{file_path}"
            )
        
        except Exception as e:
            QMessageBox.critical(self, "Ошибка экспорта", f"Произошла ошибка при экспорте: {str(e)}")
