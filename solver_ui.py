import sys
from typing import Dict, Tuple, List, Optional
from time import sleep

from PyQt5.QtCore import Qt

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtWidgets import QGroupBox

import sudoku_solver_model

SOLUTION = [
    (1, 1, 1),
    (1, 6, 6),
    (1, 7, 3),
    (2, 2, 3),
    (2, 5, 1),
    (2, 8, 4),
    (3, 3, 9),
    (3, 4, 5),
    (3, 9, 7),
    (4, 3, 6),
    (4, 4, 3),
    (5, 2, 2),
    (5, 5, 8),
    (6, 1, 7),
    (6, 6, 4),
    (7, 3, 5),
    (7, 4, 9),
    (7, 9, 3),
    (8, 1, 9),
    (8, 7, 1),
    (9, 2, 8),
    (9, 5, 2),
    (9, 8, 7),
]


class GridUI(QWidget):
    """PyCalc's View (GUI)."""

    def __init__(self):
        """View initializer."""
        super().__init__()

        self.setWindowIcon(QIcon('icon.png'))
        self.setWindowTitle('Sudoku Solver')
        self.setFixedSize(500, 500)
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.cells: Dict[Tuple[int, int], QComboBox] = {}

        self._create_board_grid()
        self._create_buttons()
        
        self._test()
        
    def _test(self):
        for x, y, value in SOLUTION:
            self.cells[(y-1, x-1)].setCurrentText(str(value))

    def _create_board_grid(self):
        """Create 9 GroupBoxes representing the 9 boxes of the board."""
        for i in range(3):
            for j in range(3):
                box = QGroupBox()
                box_layout = QGridLayout()
                box.setLayout(box_layout)
                self._create_box_cells(box_layout, i, j)
                self.layout.addWidget(box, i, j)

    def _create_box_cells(self, group, i, j):
        """Create 9 cells and add them to their box group."""
        positions = [(x, y) for x in range(3) for y in range(3)]
        for x, y in positions:
            cell = self._create_combo_box()
            self.cells[(x + 3 * i, y + 3 * j)] = cell
            group.addWidget(cell, x, y)

    def _create_combo_box(self) -> QComboBox:
        """Create a combo box with possible values 1-9."""
        cell = QComboBox()
        cell.addItem('')
        for i in range(1, 10):
            cell.addItem(str(i))
        cell.setFixedSize(30, 20)
        return cell

    def _create_buttons(self):
        """Create the Solve and Reset buttons."""
        self.solve_button = QPushButton('Solve')
        self.solve_button.clicked.connect(self.solve)
        self.layout.addWidget(self.solve_button)

        self.reset_button = QPushButton('Reset')
        self.reset_button.clicked.connect(self.reset)
        self.layout.addWidget(self.reset_button)

        self.output = QLabel("")
        self.output.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.output)

    def solve(self):
        """Attempt to solve the puzzle."""
        known_values: List[Tuple[int, int, int]] = [(y + 1, x + 1, int(c.currentText()))
                                                    for (x, y), c in self.cells.items()
                                                    if c.currentText()]
        print(known_values)
        self.output.setText("Solving...")

        solution: Optional[List[Tuple[int, int, int]]] = sudoku_solver_model.solve(known_values)

        if isinstance(solution, str):
            self.output.setText(solution)
            return

        self.output.setText("Solved")

        for x, y, value in solution:
            self.cells[(y - 1, x - 1)].setCurrentText(str(value))

    def reset(self):
        """Reset all cells to their default value."""
        for cell in self.cells.values():
            cell.setCurrentIndex(0)

        self.output.setText("")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = GridUI()
    ui.show()
    sys.exit(app.exec_())
