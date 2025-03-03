"""
Модуль с вспомогательными функциями для работы с данными и моделями.
"""
import os
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def load_data(file_path: str) -> pd.DataFrame:
    """
    Загрузка данных из CSV-файла.
    
    Args:
        file_path (str): Путь к CSV-файлу
        
    Returns:
        pd.DataFrame: Загруженные данные
    """
    # Определение типа файла по расширению
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.csv':
        # Пробуем разные разделители
        for sep in [',', ';', '\t']:
            try:
                df = pd.read_csv(file_path, sep=sep)
                if len(df.columns) > 1:  # Если данные правильно разделены
                    return df
            except Exception:
                continue
        # Если ни один разделитель не подошел
        raise ValueError("Не удалось правильно прочитать CSV-файл")
    elif file_ext in ['.xls', '.xlsx']:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {file_ext}")


def preprocess_data(
    df: pd.DataFrame, 
    target_column: str, 
    test_size: float = 0.2, 
    random_state: int = 42,
    scale: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Optional[StandardScaler]]:
    """
    Предобработка данных для обучения модели.
    
    Args:
        df (pd.DataFrame): Исходные данные
        target_column (str): Название целевой колонки
        test_size (float, optional): Размер тестовой выборки. По умолчанию 0.2.
        random_state (int, optional): Случайное состояние для воспроизводимости. По умолчанию 42.
        scale (bool, optional): Флаг стандартизации данных. По умолчанию True.
        
    Returns:
        Tuple: (X_train, X_test, y_train, y_test, feature_names, scaler)
            X_train (np.ndarray): Обучающие признаки
            X_test (np.ndarray): Тестовые признаки
            y_train (np.ndarray): Обучающие целевые значения
            y_test (np.ndarray): Тестовые целевые значения
            feature_names (List[str]): Названия признаков
            scaler (Optional[StandardScaler]): Объект для стандартизации данных
    """
    # Проверка наличия целевой колонки
    if target_column not in df.columns:
        raise ValueError(f"Колонка {target_column} не найдена в данных")
    
    # Подготовка данных
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Сохранение названий признаков
    feature_names = X.columns.tolist()
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Стандартизация данных
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        X_train = X_train.values
        X_test = X_test.values
    
    return X_train, X_test, y_train.values, y_test.values, feature_names, scaler


def plot_feature_importance(model: Any, feature_names: List[str], save_path: Optional[str] = None) -> plt.Figure:
    """
    Построение графика важности признаков.
    
    Args:
        model (Any): Обученная модель
        feature_names (List[str]): Список имен признаков
        save_path (Optional[str], optional): Путь для сохранения графика. По умолчанию None.
        
    Returns:
        plt.Figure: Объект графика
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Модель не поддерживает анализ важности признаков")
    
    # Получение важности признаков
    importances = model.feature_importances_
    
    # Сортировка признаков по важности
    indices = np.argsort(importances)[::-1]
    
    # Построение графика
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.title("Важность признаков")
    plt.barh(range(len(indices)), importances[indices], color='violet')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Важность")
    plt.tight_layout()
    
    # Сохранение графика
    if save_path:
        plt.savefig(save_path)
    
    return fig


def generate_hyperparameter_report(
    model_name: str,
    hyperparams_history: List[Dict[str, Any]],
    results_history: List[Dict[str, float]],
    save_path: str
) -> str:
    """
    Генерация отчета о влиянии гиперпараметров на качество модели.
    
    Args:
        model_name (str): Название модели
        hyperparams_history (List[Dict[str, Any]]): История гиперпараметров
        results_history (List[Dict[str, float]]): История результатов
        save_path (str): Путь для сохранения отчета
        
    Returns:
        str: Путь к созданному файлу отчета
    """
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_hyperparams_report_{timestamp}.txt"
    filepath = os.path.join(save_path, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"Отчет о влиянии гиперпараметров на модель {model_name}\n")
        f.write(f"Дата и время: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Итоговая таблица результатов:\n")
        f.write("-" * 80 + "\n")
        
        # Формирование заголовка таблицы на основе первого набора гиперпараметров
        if hyperparams_history:
            hyperparams_keys = list(hyperparams_history[0].keys())
            header = "| # | " + " | ".join(hyperparams_keys) + " | MSE | MAE | R² |"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            
            # Запись данных
            for i, (hyperparams, results) in enumerate(zip(hyperparams_history, results_history), 1):
                values = [str(hyperparams.get(key, "N/A")) for key in hyperparams_keys]
                metrics = [
                    f"{results.get('mse', 'N/A'):.4f}",
                    f"{results.get('mae', 'N/A'):.4f}",
                    f"{results.get('r2', 'N/A'):.4f}"
                ]
                row = f"| {i} | " + " | ".join(values) + " | " + " | ".join(metrics) + " |"
                f.write(row + "\n")
            
            f.write("-" * len(header) + "\n\n")
        
        # Анализ влияния каждого гиперпараметра
        f.write("Анализ влияния гиперпараметров:\n")
        f.write("=" * 80 + "\n\n")
        
        if len(hyperparams_history) > 1:
            all_params = set()
            for params in hyperparams_history:
                all_params.update(params.keys())
            
            for param in all_params:
                f.write(f"Влияние гиперпараметра '{param}':\n")
                f.write("-" * 50 + "\n")
                
                # Сбор уникальных значений параметра и соответствующих результатов
                param_values = {}
                for hp, res in zip(hyperparams_history, results_history):
                    if param in hp:
                        value = hp[param]
                        if value not in param_values:
                            param_values[value] = []
                        param_values[value].append(res)
                
                # Вычисление средних метрик для каждого значения параметра
                for value, results_list in param_values.items():
                    avg_mse = sum(r.get('mse', 0) for r in results_list) / len(results_list)
                    avg_mae = sum(r.get('mae', 0) for r in results_list) / len(results_list)
                    avg_r2 = sum(r.get('r2', 0) for r in results_list) / len(results_list)
                    
                    f.write(f"  Значение: {value}\n")
                    f.write(f"    Среднее MSE: {avg_mse:.4f}\n")
                    f.write(f"    Среднее MAE: {avg_mae:.4f}\n")
                    f.write(f"    Среднее R²: {avg_r2:.4f}\n")
                    f.write(f"    Количество экспериментов: {len(results_list)}\n\n")
                
                # Определение лучшего значения параметра
                best_value = None
                best_r2 = -float('inf')
                for value, results_list in param_values.items():
                    avg_r2 = sum(r.get('r2', 0) for r in results_list) / len(results_list)
                    if avg_r2 > best_r2:
                        best_r2 = avg_r2
                        best_value = value
                
                f.write(f"  Рекомендуемое значение '{param}': {best_value} (среднее R²: {best_r2:.4f})\n\n")
        else:
            f.write("Недостаточно данных для анализа влияния гиперпараметров.\n")
            f.write("Попробуйте провести несколько экспериментов с разными значениями гиперпараметров.\n\n")
        
        f.write("\nПримечание: R² (коэффициент детерминации) ближе к 1.0 означает лучшую модель.\n")
        f.write("MSE и MAE меньше означает лучшую модель.\n")
    
    return filepath
