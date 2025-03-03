"""
Главный файл приложения ModelTrainerPro.
Запускает основной интерфейс приложения для обучения моделей машинного обучения.
"""

import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QDir

# Добавляем пути к модулям проекта
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Импортируем главное окно приложения
from ui.main_window import MainWindow

def main() -> None:
    """
    Основная функция запуска приложения.
    Создает экземпляр QApplication и запускает главное окно.
    
    Returns:
        None
    """
    # Создаем необходимые директории, если они не существуют
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Создаем экземпляр приложения
    app = QApplication(sys.argv)
    
    # Устанавливаем русскую локализацию
    app.setStyle('Fusion')  # Использование стиля Fusion для лучшей кроссплатформенности
    
    # Создаем и показываем главное окно
    main_window = MainWindow()
    main_window.show()
    
    # Запускаем цикл обработки событий приложения
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
