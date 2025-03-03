import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def save_model(model: Any, model_info: Dict[str, Any], path: str) -> str:
    """
    Сохраняет модель и информацию о ней в файл.
    
    Args:
        model (Any): Обученная модель
        model_info (Dict[str, Any]): Информация о модели (гиперпараметры, метрики и т.д.)
        path (str): Путь для сохранения модели
        
    Returns:
        str: Полный путь к сохраненной модели
        
    Raises:
        IOError: Если не удалось сохранить модель
    """
    try:
        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Сохраняем модель и информацию о ней
        save_data = {
            'model': model,
            'info': model_info
        }
        
        joblib.dump(save_data, path)
        return path
    except Exception as e:
        raise IOError(f"Ошибка при сохранении модели: {str(e)}")

def load_model(path: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Загружает модель и информацию о ней из файла.
    
    Args:
        path (str): Путь к файлу модели
        
    Returns:
        Tuple[Any, Dict[str, Any]]: Кортеж с моделью и информацией о ней
        
    Raises:
        IOError: Если не удалось загрузить модель
    """
    try:
        # Загружаем модель и информацию о ней
        save_data = joblib.load(path)
        
        return save_data['model'], save_data['info']
    except Exception as e:
        raise IOError(f"Ошибка при загрузке модели: {str(e)}")

def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Оценивает качество модели на тестовых данных.
    
    Args:
        model (Any): Обученная модель
        X_test (np.ndarray): Тестовые признаки
        y_test (np.ndarray): Тестовые целевые значения
        
    Returns:
        Dict[str, float]: Словарь с метриками качества
    """
    # Делаем предсказания
    y_pred = model.predict(X_test)
    
    # Рассчитываем метрики
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred)
    }
    
    return metrics

def get_model_params(model_type: str) -> Dict[str, Dict[str, Any]]:
    """
    Возвращает доступные гиперпараметры для указанного типа модели.
    
    Args:
        model_type (str): Тип модели ('random_forest' или 'knn')
        
    Returns:
        Dict[str, Dict[str, Any]]: Словарь с информацией о гиперпараметрах
        
    Raises:
        ValueError: Если указан неподдерживаемый тип модели
    """
    if model_type == 'random_forest':
        return {
            'n_estimators': {
                'type': 'int',
                'default': 100,
                'min': 10,
                'max': 1000,
                'description': 'Количество деревьев в лесу'
            },
            'max_depth': {
                'type': 'int',
                'default': None,
                'min': 1,
                'max': 100,
                'description': 'Максимальная глубина дерева'
            },
            'min_samples_split': {
                'type': 'int',
                'default': 2,
                'min': 2,
                'max': 20,
                'description': 'Минимальное количество выборок для разделения внутреннего узла'
            },
            'min_samples_leaf': {
                'type': 'int',
                'default': 1,
                'min': 1,
                'max': 20,
                'description': 'Минимальное количество выборок для листового узла'
            },
            'random_state': {
                'type': 'int',
                'default': 42,
                'min': 0,
                'max': 100,
                'description': 'Начальное значение для генератора случайных чисел'
            }
        }
    elif model_type == 'knn':
        return {
            'n_neighbors': {
                'type': 'int',
                'default': 5,
                'min': 1,
                'max': 50,
                'description': 'Количество соседей'
            },
            'weights': {
                'type': 'select',
                'default': 'uniform',
                'options': ['uniform', 'distance'],
                'description': 'Весовая функция'
            },
            'algorithm': {
                'type': 'select',
                'default': 'auto',
                'options': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'description': 'Алгоритм для вычисления ближайших соседей'
            },
            'p': {
                'type': 'int',
                'default': 2,
                'min': 1,
                'max': 2,
                'description': 'Степень метрики Минковского (1 - Манхэттен, 2 - Евклидово расстояние)'
            }
        }
    else:
        raise ValueError(f"Неподдерживаемый тип модели: {model_type}")

def format_metrics(metrics: Dict[str, float]) -> str:
    """
    Форматирует метрики для отображения.
    
    Args:
        metrics (Dict[str, float]): Словарь с метриками
        
    Returns:
        str: Отформатированная строка с метриками
    """
    lines = [
        f"MAE (Средняя абсолютная ошибка): {metrics['mae']:.4f}",
        f"MSE (Средняя квадратичная ошибка): {metrics['mse']:.4f}",
        f"RMSE (Корень из средней квадратичной ошибки): {metrics['rmse']:.4f}",
        f"R² (Коэффициент детерминации): {metrics['r2']:.4f}"
    ]
    
    return "\n".join(lines)

def get_feature_importance(model: Any, feature_names: List[str]) -> Dict[str, float]:
    """
    Получает важности признаков из модели (если доступно).
    
    Args:
        model (Any): Обученная модель
        feature_names (List[str]): Список имен признаков
        
    Returns:
        Dict[str, float]: Словарь с важностями признаков
    """
    # Проверяем, есть ли у модели атрибут feature_importances_
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        return {feature_names[i]: importance[i] for i in range(len(feature_names))}
    return {}
