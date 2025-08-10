import sys
from PyQt6.QtWidgets import QApplication
from gui import MainApp

if __name__ == "__main__":
    """
    The main entry point for the application.
    """
    app = QApplication(sys.argv)
    main_win = MainApp()
    main_win.show()
    sys.exit(app.exec())