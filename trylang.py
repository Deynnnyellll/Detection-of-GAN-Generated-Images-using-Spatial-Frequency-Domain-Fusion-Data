import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QLabel, QPushButton

class DynamicLabelListFlexWrapExample(QMainWindow):
    def __init__(self):
        super().__init__()

        self.labels = []  # List of labels

        self.initUI()

    def initUI(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QGridLayout(self.central_widget)
        self.add_button = QPushButton("Add Labels", self)
        self.add_button.clicked.connect(self.add_labels)
        self.layout.addWidget(self.add_button, 0, 0, 1, 2)

        self.row = 1
        self.column = 0

        self.setWindowTitle("Dynamic Label List Flex Wrap Example")
        self.setGeometry(100, 100, 400, 300)

    def add_labels(self):
        labels_to_add = ["Label A"]
        
        for label_text in labels_to_add:
            label = QLabel(label_text)
            self.layout.addWidget(label, self.row, self.column)
            self.labels.append(label)
            
            # Move to the next column, and if the row is full, wrap to the next row.
            self.column += 1
            if self.column == 2:
                self.column = 0
                self.row += 1

def main():
    app = QApplication(sys.argv)
    ex = DynamicLabelListFlexWrapExample()
    ex.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
