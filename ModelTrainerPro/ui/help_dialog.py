from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTextEdit, QPushButton, 
    QHBoxLayout, QTabWidget, QWidget
)
from PyQt6.QtCore import Qt

class HelpDialog(QDialog):
    """Диалоговое окно справки приложения."""
    
    def __init__(self, parent=None):
        """
        Инициализация диалога справки.
        
        Args:
            parent: Родительский виджет
        """
        super().__init__(parent)
        
        # Настройка диалога
        self.setWindowTitle("Справка ModelTrainerPro")
        self.setMinimumSize(700, 500)
        
        # Основной layout
        self.main_layout = QVBoxLayout(self)
        
        # Создаем вкладки для разных разделов справки
        self.tabs = QTabWidget()
        
        # Вкладка "Общая информация"
        self.general_tab = QWidget()
        self.general_layout = QVBoxLayout(self.general_tab)
        
        self.general_text = QTextEdit()
        self.general_text.setReadOnly(True)
        self.general_text.setHtml(self.get_general_help())
        
        self.general_layout.addWidget(self.general_text)
        
        # Вкладка "Параметры моделей"
        self.params_tab = QWidget()
        self.params_layout = QVBoxLayout(self.params_tab)
        
        self.params_text = QTextEdit()
        self.params_text.setReadOnly(True)
        self.params_text.setHtml(self.get_model_params_help())
        
        self.params_layout.addWidget(self.params_text)
        
        # Вкладка "Метрики и результаты"
        self.metrics_tab = QWidget()
        self.metrics_layout = QVBoxLayout(self.metrics_tab)
        
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setHtml(self.get_metrics_help())
        
        self.metrics_layout.addWidget(self.metrics_text)
        
        # Вкладка "Использование модели"
        self.usage_tab = QWidget()
        self.usage_layout = QVBoxLayout(self.usage_tab)
        
        self.usage_text = QTextEdit()
        self.usage_text.setReadOnly(True)
        self.usage_text.setHtml(self.get_model_usage_help())
        
        self.usage_layout.addWidget(self.usage_text)
        
        # Добавляем вкладки
        self.tabs.addTab(self.general_tab, "Общая информация")
        self.tabs.addTab(self.params_tab, "Параметры моделей")
        self.tabs.addTab(self.metrics_tab, "Метрики и результаты")
        self.tabs.addTab(self.usage_tab, "Использование модели")
        
        # Добавляем вкладки в основной layout
        self.main_layout.addWidget(self.tabs)
        
        # Кнопка закрытия
        self.buttons_layout = QHBoxLayout()
        
        self.close_button = QPushButton("Закрыть")
        self.close_button.clicked.connect(self.accept)
        
        self.buttons_layout.addStretch()
        self.buttons_layout.addWidget(self.close_button)
        
        self.main_layout.addLayout(self.buttons_layout)
    
    def get_general_help(self) -> str:
        """
        Возвращает HTML-текст с общей информацией о приложении.
        
        Returns:
            str: HTML-текст
        """
        return """
        <h2>ModelTrainerPro - Обучение моделей машинного обучения</h2>
        
        <p>Приложение позволяет загружать данные, обучать модели машинного обучения и оценивать их качество.</p>
        
        <h3>Основные шаги работы с приложением:</h3>
        
        <ol>
            <li><b>Загрузка данных:</b>
                <ul>
                    <li>Нажмите кнопку "Загрузить данные"</li>
                    <li>Выберите CSV или Excel файл с данными</li>
                    <li>После загрузки данные отобразятся в таблице</li>
                </ul>
            </li>
            <li><b>Настройка модели:</b>
                <ul>
                    <li>Выберите тип модели из выпадающего списка (Random Forest или KNN Regressor)</li>
                    <li>Нажмите кнопку "Настроить модель"</li>
                    <li>В появившемся диалоге настройте параметры модели</li>
                </ul>
            </li>
            <li><b>Настройка обучения:</b>
                <ul>
                    <li>Перейдите на вкладку "Конфигурация"</li>
                    <li>Выберите целевой столбец (зависимую переменную)</li>
                    <li>При необходимости, настройте параметры разделения данных</li>
                </ul>
            </li>
            <li><b>Обучение модели:</b>
                <ul>
                    <li>Нажмите кнопку "Обучить модель"</li>
                    <li>После обучения результаты появятся на вкладке "Результаты"</li>
                </ul>
            </li>
            <li><b>Анализ результатов:</b>
                <ul>
                    <li>На вкладке "Результаты" вы увидите метрики качества модели</li>
                    <li>Для Random Forest будет доступна информация о важности признаков</li>
                </ul>
            </li>
            <li><b>Создание отчёта:</b>
                <ul>
                    <li>Перейдите на вкладку "Отчёты"</li>
                    <li>Нажмите кнопку "Создать отчёт"</li>
                    <li>Отчёт будет сохранён в папке "reports" и показан в текстовом поле</li>
                </ul>
            </li>
            <li><b>Сохранение/загрузка модели:</b>
                <ul>
                    <li>Нажмите "Сохранить модель" для сохранения обученной модели</li>
                    <li>Нажмите "Загрузить модель" для загрузки ранее обученной модели</li>
                </ul>
            </li>
        </ol>
        """
    
    def get_model_params_help(self) -> str:
        """
        Возвращает HTML-текст с описанием параметров моделей.
        
        Returns:
            str: HTML-текст
        """
        return """
        <h2>Параметры моделей машинного обучения</h2>
        
        <h3>Random Forest (Случайный лес)</h3>
        
        <p>Random Forest - это ансамблевый метод, основанный на построении множества деревьев решений.</p>
        
        <ul>
            <li><b>n_estimators</b> - количество деревьев в лесу.
                <ul>
                    <li>Чем больше деревьев, тем более устойчивы результаты, но дольше время обучения.</li>
                    <li>Рекомендуемое значение: 100-500 для большинства задач.</li>
                </ul>
            </li>
            <li><b>max_depth</b> - максимальная глубина дерева.
                <ul>
                    <li>Помогает бороться с переобучением. Если установлено значение 0 - это означает "без ограничений".</li>
                    <li>Рекомендуемое значение: 5-20 в зависимости от размера данных.</li>
                </ul>
            </li>
            <li><b>min_samples_split</b> - минимальное количество выборок для разделения внутреннего узла.
                <ul>
                    <li>Больше значение - меньше разделений и более простая модель.</li>
                    <li>Рекомендуемое значение: 2-10.</li>
                </ul>
            </li>
            <li><b>min_samples_leaf</b> - минимальное количество выборок для листового узла.
                <ul>
                    <li>Больше значение - более простая модель, меньше переобучение.</li>
                    <li>Рекомендуемое значение: 1-5.</li>
                </ul>
            </li>
            <li><b>random_state</b> - начальное значение для генератора случайных чисел.
                <ul>
                    <li>Обеспечивает воспроизводимость результатов.</li>
                    <li>Может быть любым целым числом.</li>
                </ul>
            </li>
        </ul>
        
        <h3>KNN Regressor (Метод k-ближайших соседей)</h3>
        
        <p>KNN предсказывает значение для новой точки на основе значений k ближайших к ней точек из обучающей выборки.</p>
        
        <ul>
            <li><b>n_neighbors</b> - количество соседей.
                <ul>
                    <li>Определяет, сколько ближайших соседей учитывать при предсказании.</li>
                    <li>Маленькие значения могут приводить к переобучению, большие - к недообучению.</li>
                    <li>Рекомендуемое значение: 3-10, часто используют нечётные числа.</li>
                </ul>
            </li>
            <li><b>weights</b> - весовая функция.
                <ul>
                    <li><i>uniform</i> - все соседи имеют одинаковый вес.</li>
                    <li><i>distance</i> - веса обратно пропорциональны расстоянию до соседа (чем ближе, тем больше вес).</li>
                </ul>
            </li>
            <li><b>algorithm</b> - алгоритм для вычисления ближайших соседей.
                <ul>
                    <li><i>auto</i> - автоматический выбор наиболее подходящего алгоритма.</li>
                    <li><i>ball_tree</i>, <i>kd_tree</i>, <i>brute</i> - конкретные алгоритмы поиска соседей.</li>
                </ul>
            </li>
            <li><b>p</b> - степень метрики Минковского.
                <ul>
                    <li>p=1 - Манхэттенское расстояние (сумма модулей разностей).</li>
                    <li>p=2 - Евклидово расстояние (корень из суммы квадратов разностей).</li>
                </ul>
            </li>
        </ul>
        """
    
    def get_metrics_help(self) -> str:
        """
        Возвращает HTML-текст с описанием метрик оценки моделей.
        
        Returns:
            str: HTML-текст
        """
        return """
        <h2>Метрики оценки качества моделей</h2>
        
        <p>Для оценки качества модели регрессии используются следующие метрики:</p>
        
        <ul>
            <li><b>MAE (Mean Absolute Error)</b> - средняя абсолютная ошибка.
                <ul>
                    <li>Среднее абсолютных разностей между предсказанными и фактическими значениями.</li>
                    <li>Показывает среднюю величину ошибки без учёта направления.</li>
                    <li>Единицы измерения совпадают с единицами измерения целевой переменной.</li>
                    <li>Чем ниже MAE, тем лучше модель.</li>
                </ul>
            </li>
            <li><b>MSE (Mean Squared Error)</b> - средняя квадратичная ошибка.
                <ul>
                    <li>Среднее квадратов разностей между предсказанными и фактическими значениями.</li>
                    <li>Штрафует большие ошибки сильнее, чем маленькие.</li>
                    <li>Единицы измерения - квадрат единиц измерения целевой переменной.</li>
                    <li>Чем ниже MSE, тем лучше модель.</li>
                </ul>
            </li>
            <li><b>RMSE (Root Mean Squared Error)</b> - корень из средней квадратичной ошибки.
                <ul>
                    <li>Квадратный корень из MSE.</li>
                    <li>Приводит ошибку к тем же единицам измерения, что и целевая переменная.</li>
                    <li>Чем ниже RMSE, тем лучше модель.</li>
                </ul>
            </li>
            <li><b>R² (Коэффициент детерминации)</b> - доля дисперсии целевой переменной, объясненная моделью.
                <ul>
                    <li>R² = 1 - (MSE модели / Дисперсия целевой переменной)</li>
                    <li>Показывает, насколько хорошо модель объясняет изменения целевой переменной.</li>
                    <li>Значения от 0 до 1, где 1 означает идеальное предсказание.</li>
                    <li>R² < 0 означает, что модель хуже, чем простое среднее значение.</li>
                    <li>Интерпретация:<br>
                        0.9-1.0 - отличное качество<br>
                        0.8-0.9 - хорошее качество<br>
                        0.7-0.8 - удовлетворительное качество<br>
                        0.6-0.7 - посредственное качество<br>
                        < 0.6 - низкое качество</li>
                </ul>
            </li>
        </ul>
        
        <h3>Важность признаков</h3>
        
        <p>Для модели Random Forest вычисляется важность признаков, которая показывает, насколько каждый признак влияет на предсказания модели.</p>
        
        <ul>
            <li>Значения важности находятся в диапазоне от 0 до 1, где 1 - максимальная важность.</li>
            <li>Сумма всех важностей равна 1.</li>
            <li>Чем выше важность признака, тем больше он влияет на предсказания модели.</li>
            <li>Признаки с нулевой важностью можно исключить из модели без потери качества.</li>
        </ul>
        """
    
    def get_model_usage_help(self) -> str:
        """
        Возвращает HTML-текст с описанием использования обученных моделей.
        
        Returns:
            str: HTML-текст
        """
        return """
        <h2>Использование обученных моделей</h2>
        
        <p>После обучения модели можно использовать её для предсказания на новых данных следующими способами:</p>
        
        <h3>1. Использование в рамках приложения</h3>
        
        <p>Обученную модель можно сохранить в файл и позже загрузить для использования:</p>
        
        <ol>
            <li>Нажмите кнопку "Сохранить модель" и выберите место для сохранения файла .joblib</li>
            <li>Позже вы можете загрузить модель, нажав "Загрузить модель" и выбрав сохранённый файл</li>
            <li>Загруженная модель будет готова к использованию с новыми данными</li>
        </ol>
        
        <h3>2. Использование в Python-скриптах</h3>
        
        <p>Сохранённую модель можно использовать в ваших Python-скриптах:</p>
        
        <pre>
import joblib

# Загрузка модели из файла
loaded_data = joblib.load('путь_к_файлу_модели.joblib')
model = loaded_data['model']
model_info = loaded_data['info']

# Подготовка новых данных (должны иметь те же признаки, что и при обучении)
import pandas as pd
new_data = pd.read_csv('новые_данные.csv')

# Предобработка данных (так же, как при обучении)
features = model_info['feature_names']
X_new = new_data[features]

# Масштабирование данных, если использовалось при обучении
if 'scaler' in model_info and model_info['scaler'] is not None:
    X_new = model_info['scaler'].transform(X_new)

# Получение предсказаний
predictions = model.predict(X_new)
print(predictions)
        </pre>
        
        <h3>3. Использование модели в продакшене</h3>
        
        <p>Для использования модели в продакшене можно:</p>
        
        <ul>
            <li><b>Создать API</b> - разработать API на основе Flask или FastAPI, который будет принимать запросы с данными и возвращать предсказания</li>
            <li><b>Интегрировать в приложение</b> - встроить модель в веб-приложение или десктопное приложение</li>
            <li><b>Автоматизировать процесс</b> - настроить автоматический запуск скрипта с предсказаниями по расписанию</li>
        </ul>
        
        <p>Пример кода для создания простого API на Flask:</p>
        
        <pre>
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Загрузка модели
loaded_data = joblib.load('model.joblib')
model = loaded_data['model']
feature_names = loaded_data['info']['feature_names']
scaler = loaded_data['info'].get('scaler')

@app.route('/predict', methods=['POST'])
def predict():
    # Получение данных из запроса
    data = request.json
    
    # Преобразование в DataFrame
    df = pd.DataFrame(data, index=[0])
    
    # Проверка наличия всех необходимых признаков
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        return jsonify({'error': f'Missing features: {missing_features}'})
    
    # Выбор только нужных признаков
    X = df[feature_names]
    
    # Применение масштабирования, если оно использовалось
    if scaler is not None:
        X = scaler.transform(X)
    
    # Получение предсказания
    prediction = model.predict(X)
    
    # Возврат результата
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
        </pre>
        """


def accept(self):
    """Закрывает диалог с кодом 'Accepted'."""
    self.done(QDialog.DialogCode.Accepted)