import numpy as np
import os
import time
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from models.model_utils import save_model, evaluate_model, get_feature_importance

class ModelTrainer:
    """Класс для обучения моделей машинного обучения."""
    
    def __init__(self):
        """Инициализация тренера моделей."""
        self.model = None
        self.model_type = None
        self.trained = False
        self.training_time = 0
        self.feature_names = None
        self.metrics = None
        self.best_params = None
    
    def train_model(self, model_type: str, X_train: np.ndarray, y_train: np.ndarray, 
                   X_test: np.ndarray, y_test: np.ndarray, 
                   feature_names: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обучает модель заданного типа на тренировочных данных.
        
        Args:
            model_type (str): Тип модели ('random_forest' или 'knn')
            X_train (np.ndarray): Обучающие признаки
            y_train (np.ndarray): Обучающие целевые значения
            X_test (np.ndarray): Тестовые признаки
            y_test (np.ndarray): Тестовые целевые значения
            feature_names (List[str]): Список имен признаков
            params (Dict[str, Any]): Гиперпараметры модели
            
        Returns:
            Dict[str, Any]: Результаты обучения и оценки модели
            
        Raises:
            ValueError: Если указан неподдерживаемый тип модели
        """
        self.model_type = model_type
        self.feature_names = feature_names
        
        # Засекаем время начала обучения
        start_time = time.time()
        
        # Выбираем и инициализируем модель в зависимости от типа
        if model_type == 'random_forest':
            # Фильтруем параметры
            rf_params = {
                'n_estimators': params.get('n_estimators', 100),
                'random_state': params.get('random_state', 42)
            }
            
            # Добавляем опциональные параметры, если они предоставлены
            if 'max_depth' in params and params['max_depth'] is not None:
                rf_params['max_depth'] = params['max_depth']
            if 'min_samples_split' in params:
                rf_params['min_samples_split'] = params['min_samples_split']
            if 'min_samples_leaf' in params:
                rf_params['min_samples_leaf'] = params['min_samples_leaf']
            
            # Создаем и обучаем модель
            self.model = RandomForestRegressor(**rf_params)
            
        elif model_type == 'knn':
            # Создаем и обучаем KNN модель
            self.model = KNeighborsRegressor(
                n_neighbors=params.get('n_neighbors', 5),
                weights=params.get('weights', 'uniform'),
                algorithm=params.get('algorithm', 'auto'),
                p=params.get('p', 2)
            )
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {model_type}")
        
        # Обучаем модель
        self.model.fit(X_train, y_train)
        
        # Замеряем время обучения
        self.training_time = time.time() - start_time
        
        # Вычисляем метрики на тестовых данных
        self.metrics = evaluate_model(self.model, X_test, y_test)
        
        # Сохраняем параметры модели
        self.best_params = params
        
        # Получаем важности признаков (если доступно)
        feature_importance = get_feature_importance(self.model, self.feature_names)
        
        # Помечаем модель как обученную
        self.trained = True
        
        # Возвращаем результаты
        return {
            'model_type': self.model_type,
            'training_time': self.training_time,
            'metrics': self.metrics,
            'params': self.best_params,
            'feature_importance': feature_importance
        }
    
    def cross_validate(self, model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                      param_grid: Dict[str, List[Any]], cv: int = 5,
                      n_jobs: int = -1) -> Dict[str, Any]:
        """
        Проводит перекрестную проверку для поиска оптимальных гиперпараметров.
        
        Args:
            model_type (str): Тип модели ('random_forest' или 'knn')
            X_train (np.ndarray): Обучающие признаки
            y_train (np.ndarray): Обучающие целевые значения
            param_grid (Dict[str, List[Any]]): Сетка параметров для перебора
            cv (int, optional): Количество фолдов для перекрестной проверки. По умолчанию 5.
            n_jobs (int, optional): Количество параллельных задач. По умолчанию -1 (все доступные ядра).
            
        Returns:
            Dict[str, Any]: Результаты поиска по сетке параметров
            
        Raises:
            ValueError: Если указан неподдерживаемый тип модели
        """
        # Выбираем базовую модель в зависимости от типа
        if model_type == 'random_forest':
            base_model = RandomForestRegressor()
        elif model_type == 'knn':
            base_model = KNeighborsRegressor()
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {model_type}")
        
        # Засекаем время начала поиска
        start_time = time.time()
        
        # Создаем объект GridSearchCV
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            n_jobs=n_jobs,
            scoring='neg_mean_squared_error',
            verbose=1,
            return_train_score=True
        )
        
        # Выполняем поиск по сетке параметров
        grid_search.fit(X_train, y_train)
        
        # Замеряем время выполнения
        search_time = time.time() - start_time
        
        # Возвращаем результаты
        return {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_,  # Преобразуем neg_mean_squared_error обратно в MSE
            'cv_results': grid_search.cv_results_,
            'search_time': search_time
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Делает предсказания с помощью обученной модели.
        
        Args:
            X (np.ndarray): Признаки для предсказания
            
        Returns:
            np.ndarray: Предсказанные значения
            
        Raises:
            ValueError: Если модель не обучена
        """
        if not self.trained or self.model is None:
            raise ValueError("Модель не обучена")
        
        return self.model.predict(X)
    
    def save(self, path: str) -> str:
        """
        Сохраняет обученную модель и информацию о ней.
        
        Args:
            path (str): Путь для сохранения модели
            
        Returns:
            str: Полный путь к сохраненной модели
            
        Raises:
            ValueError: Если модель не обучена
        """
        if not self.trained or self.model is None:
            raise ValueError("Нет обученной модели для сохранения")
        
        # Собираем информацию о модели
        model_info = {
            'model_type': self.model_type,
            'training_time': self.training_time,
            'metrics': self.metrics,
            'params': self.best_params,
            'feature_names': self.feature_names,
            'trained_date': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Сохраняем модель и информацию
        return save_model(self.model, model_info, path)
    
    def generate_hyperparameter_report(self, results: Dict[str, Any], output_path: str) -> str:
        """
        Генерирует отчет о влиянии гиперпараметров на качество модели.
        
        Args:
            results (Dict[str, Any]): Результаты поиска гиперпараметров
            output_path (str): Путь для сохранения отчета
            
        Returns:
            str: Путь к сохраненному отчету
        """
        # Создаем директорию, если она не существует
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Получаем результаты кросс-валидации
        cv_results = results['cv_results']
        
        # Создаем отчет
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Отчет о влиянии гиперпараметров на качество модели\n\n")
            f.write(f"Дата создания: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Лучшие параметры\n\n")
            
            for param, value in results['best_params'].items():
                f.write(f"* {param}: {value}\n")
            
            f.write(f"\n## Лучший результат\n\n")
            f.write(f"* RMSE: {np.sqrt(results['best_score']):.4f}\n")
            f.write(f"* MSE: {results['best_score']:.4f}\n\n")
            
            f.write(f"## Время поиска\n\n")
            f.write(f"* {results['search_time']:.2f} секунд\n\n")
            
            f.write(f"## Результаты по итерациям\n\n")
            
            # Создаем таблицу результатов
            f.write("| # | ")
            
            # Записываем заголовки параметров
            param_names = [name for name in cv_results['params'][0].keys()]
            for param in param_names:
                f.write(f"{param} | ")
            
            f.write("Mean Test RMSE | Mean Train RMSE |\n")
            f.write("|" + "-|" * (len(param_names) + 3) + "\n")
            
            # Записываем строки результатов
            for i in range(len(cv_results['params'])):
                f.write(f"| {i+1} | ")
                
                # Параметры
                for param in param_names:
                    f.write(f"{cv_results['params'][i][param]} | ")
                
                # Тестовые и тренировочные метрики
                test_rmse = np.sqrt(-cv_results['mean_test_score'][i])
                train_rmse = np.sqrt(-cv_results['mean_train_score'][i])
                
                f.write(f"{test_rmse:.4f} | {train_rmse:.4f} |\n")
            
            f.write("\n## Анализ влияния гиперпараметров\n\n")
            
            # Анализируем влияние каждого параметра
            for param in param_names:
                f.write(f"### Влияние параметра '{param}'\n\n")
                
                # Собираем уникальные значения параметра и средние результаты для них
                param_values = {}
                for i, params in enumerate(cv_results['params']):
                    value = params[param]
                    if value not in param_values:
                        param_values[value] = []
                    
                    param_values[value].append(-cv_results['mean_test_score'][i])
                
                # Вычисляем средние значения метрик для каждого значения параметра
                f.write("| Значение | Среднее MSE | Среднее RMSE |\n")
                f.write("|-|-|-|\n")
                
                for value, scores in param_values.items():
                    avg_mse = np.mean(scores)
                    avg_rmse = np.sqrt(avg_mse)
                    f.write(f"| {value} | {avg_mse:.4f} | {avg_rmse:.4f} |\n")
                
                f.write("\n")
            
            f.write("## Выводы\n\n")
            f.write("На основе проведенного исследования гиперпараметров можно сделать следующие выводы:\n\n")
            
            # Простой анализ для определения наиболее влиятельных параметров
            param_influence = {}
            for param in param_names:
                unique_values = set([params[param] for params in cv_results['params']])
                
                if len(unique_values) > 1:
                    # Собираем средние оценки для каждого значения параметра
                    value_scores = {}
                    for i, params in enumerate(cv_results['params']):
                        value = params[param]
                        if value not in value_scores:
                            value_scores[value] = []
                        value_scores[value].append(-cv_results['mean_test_score'][i])
                    
                    # Вычисляем средние для каждого значения
                    avg_scores = [np.mean(scores) for scores in value_scores.values()]
                    
                    # Вычисляем разброс средних оценок
                    score_range = max(avg_scores) - min(avg_scores)
                    param_influence[param] = score_range
            
            # Сортируем параметры по степени влияния
            sorted_influence = sorted(param_influence.items(), key=lambda x: x[1], reverse=True)
            
            for param, influence in sorted_influence:
                if influence > 0.001:  # Порог значимости
                    best_value = None
                    best_score = float('inf')
                    
                    # Находим лучшее значение параметра
                    for i, params in enumerate(cv_results['params']):
                        if -cv_results['mean_test_score'][i] < best_score:
                            best_score = -cv_results['mean_test_score'][i]
                            best_value = params[param]
                    
                    f.write(f"* Параметр '{param}' оказывает значительное влияние на качество модели. "
                            f"Оптимальное значение: {best_value}.\n")
            
            f.write("\nРекомендуется использовать лучшие параметры, указанные в начале отчета, для достижения наилучшего качества модели.\n")
        
        return output_path
       