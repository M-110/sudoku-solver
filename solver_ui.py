import sys
from typing import Dict, Tuple

from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtWidgets import QGroupBox

# Create a subclass of QMainWindow to setup the calculator's GUI
class GridUI(QWidget):
    """PyCalc's View (GUI)."""
    def __init__(self):
        """View initializer."""
        super().__init__()
        
        self.setWindowTitle('Sudoku Grid')
        self.setFixedSize(500, 500)
        
        self.values = []
        #self.create_grid()
        
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        
        
        self.cells: Dict[Tuple[int, int], QComboBox] = {}
        
        self.create_group_boxes()
        
        button = QPushButton('Solve')
        button.clicked.connect(self.print_cells)
        self.layout.addWidget(button)
        button2 = QPushButton('Reset')
        self.layout.addWidget(button2)
        
    def create_group_boxes(self):
        for i in range(3):
            for j in range(3):
                group = QGroupBox()
                myGrid = QGridLayout()
                group.setLayout(myGrid)
                self.create_box_cells(myGrid, i, j)
                self.layout.addWidget(group, i, j)
                
    def create_box_cells(self, group, i, j):
        positions = [(x, y) for x in range(3) for y in range(3)]
        for x, y in positions:
            cell = self.create_combo_box()
            self.cells[(x + 3*i, y + 3*j)] = cell
            group.addWidget(cell, x, y)
            
            
    def create_combo_box(self):
        cell = QComboBox()
        cell.addItem('')
        for i in range(1,10):
            cell.addItem(str(i))
        cell.setFixedSize(30,20)
        return cell

    def print_cells(self):
        print('printing cells')
        print(len(self.cells))
        for position, combo_box in self.cells.items():
            if combo_box.currentText():
                print(f'Position: {position} - {combo_box.currentText()}')
    
                
        


app = QApplication(sys.argv)
grid = GridUI()
grid.show()
sys.exit(app.exec_())