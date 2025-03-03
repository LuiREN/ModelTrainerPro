import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ReportGenerator:
    """Класс для генерации отчетов о результатах обучения."""
    
    def __init__(self, output_dir: str = 'reports'):
        """
        Инициализация генератора отчетов.
        
        Args:
            output_dir (str, optional): Директория для сохранения отчетов. По умолчанию 'reports'.
        """
        self.output_dir = output_dir
        # Создаем директорию, если она не существует
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_training_report(self, training_results: Dict[str, Any], 
                                model_name: str, dataset_info: Dict[str, Any]) -> str:
        """
        Генерирует отчет о результатах обучения модели.
        
        Args:
            training_results (Dict[str, Any]): Результаты обучения модели
            model_name (str): Название модели
            dataset_info (Dict[str, Any]): Информация о наборе данных
            
        Returns:
            str: Путь к сгенерированному отчету
        """
        # Формируем имя файла отчета
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_filename = f"{model_name}_{timestamp}.txt"
        report_path = os.path.join(self.output_dir, report_filename)
        
        # Создаем отчет
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Отчет об обучении модели {model_name}\n\n")
            f.write(f"Дата создания: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Информация о модели
            f.write("## Информация о модели\n\n")
            f.write(f"* Тип модели: {training_results['model_type']}\n")
            f.write(f"* Время обучения: {training_results['training_time']:.2f} секунд\n\n")
            
            # Параметры модели
            f.write("## Параметры модели\n\n")
            for param, value in training_results['params'].items():
                f.write(f"* {param}: {value}\n")
            f.write("\n")
            
            # Метрики качества
            f.write("## Метрики качества\n\n")
            for metric, value in training_results['metrics'].items():
                f.write(f"* {metric.upper()}: {value:.4f}\n")
            f.write("\n")
            
            # Информация о наборе данных
            f.write("## Информация о наборе данных\n\n")
            f.write(f"* Количество строк: {dataset_info.get('total_rows', 'Н/Д')}\n")
            f.write(f"* Количество признаков: {len(dataset_info.get('features', []))}\n")
            f.write(f"* Целевая переменная: {dataset_info.get('target', 'Н/Д')}\n")
            
            # Список признаков
            if 'features' in dataset_info:
                f.write("\n### Список признаков\n\n")
                for feature in dataset_info['features']:
                    f.write(f"* {feature}\n")
            f.write("\n")
            
            # Важность признаков (если доступно)
            if 'feature_importance' in training_results and training_results['feature_importance']:
                f.write("## Важность признаков\n\n")
                
                # Сортируем признаки по важности
                sorted_importance = sorted(
                    training_results['feature_importance'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                
                for feature, importance in sorted_importance:
                    f.write(f"* {feature}: {importance:.4f}\n")
                f.write("\n")
            
            # Выводы и рекомендации
            f.write("## Выводы и рекомендации\n\n")
            
            # Оценка качества модели
            r2 = training_results['metrics'].get('r2', 0)
            if r2 > 0.9:
                quality = "отличное"
            elif r2 > 0.8:
                quality = "хорошее"
            elif r2 > 0.7:
                quality = "удовлетворительное"
            elif r2 > 0.6:
                quality = "посредственное"
            else:
                quality = "низкое"
            
            f.write(f"* Качество модели: {quality} (R² = {r2:.4f})\n")
            
            # Рекомендации по улучшению
            f.write("* Рекомендации:\n")
            
            if r2 < 0.7:
                f.write("  - Попробуйте другие значения гиперпараметров\n")
                f.write("  - Рассмотрите возможность добавления новых признаков\n")
                f.write("  - Проверьте данные на наличие выбросов и аномалий\n")
            
            # Важные признаки
            if 'feature_importance' in training_results and training_results['feature_importance']:
                top_features = sorted_importance[:min(5, len(sorted_importance))]
                f.write("* Наиболее важные признаки для модели:\n")
                for feature, importance in top_features:
                    f.write(f"  - {feature} (важность: {importance:.4f})\n")
        
        return report_path
    
    def generate_comparison_report(self, models_results: List[Dict[str, Any]], 
                                 dataset_info: Dict[str, Any]) -> str:
        """
        Генерирует отчет сравнения нескольких моделей.
        
        Args:
            models_results (List[Dict[str, Any]]): Список результатов обучения моделей
            dataset_info (Dict[str, Any]): Информация о наборе данных
            
        Returns:
            str: Путь к сгенерированному отчету
        """
        # Формируем имя файла отчета
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_filename = f"models_comparison_{timestamp}.txt"
        report_path = os.path.join(self.output_dir, report_filename)
        
        # Создаем отчет
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Сравнительный анализ моделей\n\n")
            f.write(f"Дата создания: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Информация о наборе данных
            f.write("## Информация о наборе данных\n\n")
            f.write(f"* Количество строк: {dataset_info.get('total_rows', 'Н/Д')}\n")
            f.write(f"* Количество признаков: {len(dataset_info.get('features', []))}\n")
            f.write(f"* Целевая переменная: {dataset_info.get('target', 'Н/Д')}\n\n")
            
            # Таблица сравнения моделей
            f.write("## Сравнение моделей\n\n")
            f.write("| Модель | Параметры | MAE | MSE | RMSE | R² | Время обучения (с) |\n")
            f.write("|--------|-----------|-----|-----|------|----|-----------------|\n")
            
            for result in models_results:
                model_type = result['model_type']
                
                # Форматируем параметры
                params_str = ", ".join([f"{k}={v}" for k, v in result['params'].items()])
                
                # Метрики
                metrics = result['metrics']
                mae = metrics.get('mae', 0)
                mse = metrics.get('mse', 0)
                rmse = metrics.get('rmse', 0)
                r2 = metrics.get('r2', 0)
                
                # Время обучения
                training_time = result.get('training_time', 0)
                
                f.write(f"| {model_type} | {params_str} | {mae:.4f} | {mse:.4f} | {rmse:.4f} | {r2:.4f} | {training_time:.2f} |\n")
            
            f.write("\n")
            
            # Определяем лучшую модель по R²
            best_model = max(models_results, key=lambda x: x['metrics'].get('r2', 0))
            best_model_type = best_model['model_type']
            best_r2 = best_model['metrics'].get('r2', 0)
            
            f.write("## Выводы\n\n")
            f.write(f"* Лучшая модель по метрике R²: {best_model_type} (R² = {best_r2:.4f})\n")
            f.write("* Параметры лучшей модели:\n")
            
            for param, value in best_model['params'].items():
                f.write(f"  - {param}: {value}\n")
            
            # Сравнение по времени обучения
            fastest_model = min(models_results, key=lambda x: x.get('training_time', float('inf')))
            fastest_time = fastest_model.get('training_time', 0)
            
            f.write(f"\n* Самая быстрая модель: {fastest_model['model_type']} ({fastest_time:.2f} с)\n\n")
            
            # Рекомендации
            f.write("## Рекомендации\n\n")
            
            if best_r2 > 0.8:
                f.write("* Модель имеет хорошее качество и готова к использованию в продакшене.\n")
            elif best_r2 > 0.7:
                f.write("* Модель имеет удовлетворительное качество, но перед использованием рекомендуется попробовать улучшить её, настроив гиперпараметры.\n")
            else:
                f.write("* Модель имеет невысокое качество. Рекомендуется:\n")
                f.write("  - Попробовать другие типы моделей\n")
                f.write("  - Провести дополнительную обработку данных\n")
                f.write("  - Добавить новые признаки или преобразовать существующие\n")
        
        return report_path
