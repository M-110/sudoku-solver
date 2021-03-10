"""Sudoku solving algorithm."""

from __future__ import annotations
from typing import List, Set, Tuple, Optional
from itertools import combinations


class InvalidGrid(Exception):
    """Exception raised when the Sudoku grid is invalid."""


class Box:
    """Box represents a single grid space on the board.
    
    Neighbors are combinations are lazily computed properties.
    They are reset when possible_values change.
    
    Args:
        col: column index (1 - 9)
        row: row index (1 - 9)
    """

    def __init__(self, col: int, row: int):
        self._col = col
        self._row = row

        self._known_value: Optional[int] = None
        self._possible_values = set(range(1, 10))

        self.row_neighbors = set()
        self.col_neighbors = set()
        self.block_neighbors = set()
        self.neighbors = set()
        self.neighborhood_sets = set()

        self._row_neighbors_possible_values: Optional[Set[int]] = None
        self._col_neighbors_possible_values: set[int] or None = None
        self._block_neighbors_possible_values: set[int] or None = None

        self._combinations_size_2: Optional[Set[int]] = None
        self._combinations_size_3: Optional[Set[int]] = None
        self._combinations_size_4: Optional[Set[int]] = None

        self._row_block_intersection: Optional[Set[int]] = None
        self._col_block_intersection: Optional[Set[int]] = None

        self._row_minus_block: Optional[Set[int]] = None
        self._col_minus_block: Optional[Set[int]] = None

        self._block_minus_row: Optional[Set[int]] = None
        self._block_minus_col: Optional[Set[int]] = None

    def __repr__(self):
        return f'Box({self.col}, {self.row})'

    @property
    def row(self) -> int:
        """Get row index (1 - 9)."""
        return self._row

    @property
    def col(self) -> int:
        """Get column index (1 - 9)."""
        return self._col

    @property
    def known_value(self) -> int or None:
        """Get value of box if it is known otherwise None."""
        return self._known_value

    @property
    def possible_values(self) -> Set[int]:
        """Get or set possible values."""
        for neighbor in self.neighbors:
            if neighbor.known_value:
                self.possible_values = self._possible_values - {neighbor.known_value}
        return self._possible_values

    @possible_values.setter
    def possible_values(self, value: Set[int]):
        if len(value) == 0:
            raise InvalidGrid(f'Box({self.col}, {self.row}) - Empty set!')
        if len(value) == 1 and self._known_value is None:
            self._known_value = list(value)[0]
        self._possible_values = value

        # Reset other values to None so they will be
        # forced to update next time they are called.
        self._combinations_size_2 = None
        self._combinations_size_3 = None
        self._combinations_size_4 = None

        self._row_neighbors_possible_values = None
        self._col_neighbors_possible_values = None
        self._block_neighbors_possible_values = None

    @property
    def row_neighbors_possible_values(self) -> Set[int]:
        """Get possible values of neighbors within the same row."""
        return set(possible_value for neighbor in self.row_neighbors
                   for possible_value in neighbor.possible_values)

    @property
    def col_neighbors_possible_values(self) -> Set[int]:
        """Get possible values of neighbors within the same column."""
        return set(possible_value for neighbor in self.col_neighbors
                   for possible_value in neighbor.possible_values)

    @property
    def block_neighbors_possible_values(self) -> Set[int]:
        """Get possible values of neighbors within the same block."""
        return set(possible_value for neighbor in self.block_neighbors
                   for possible_value in neighbor.possible_values)

    @property
    def combinations_size_2(self) -> List[Set[int]]:
        """Get all 2 length combinations of 2 of box's possible values."""
        if self._combinations_size_2 is None:
            self._combinations_size_2 = [set(combo) for combo in combinations(self.possible_values, 2)]
        return self._combinations_size_2

    @property
    def combinations_size_3(self) -> List[Set[int]]:
        """Get all 3 length combinations of 2 of box's possible values."""
        if self._combinations_size_3 is None:
            self._combinations_size_3 = [set(combo) for combo in combinations(self.possible_values, 3)]
        return self._combinations_size_3

    @property
    def combinations_size_4(self) -> List[Set[int]]:
        """Get all 4 length combinations of 2 of box's possible values."""
        if self._combinations_size_4 is None:
            self._combinations_size_4 = [set(combo) for combo in combinations(self.possible_values, 4)]
        return self._combinations_size_4

    @property
    def row_block_intersection(self) -> Set[Box]:
        """Get all neighbors that have the same row and block."""
        if self._row_block_intersection is None:
            self._row_block_intersection = self.row_neighbors & self.block_neighbors
        return self._row_block_intersection

    @property
    def col_block_intersection(self) -> Set[Box]:
        """Get all neighbors that have the same column and block."""
        if self._col_block_intersection is None:
            self._col_block_intersection = self.col_neighbors & self.block_neighbors
        return self._col_block_intersection

    @property
    def row_minus_block(self) -> Set[Box]:
        """Get all row neighbors outside of the box's block."""
        if self._row_minus_block is None:
            self._row_minus_block = self.row_neighbors - self.block_neighbors
        return self._row_minus_block

    @property
    def col_minus_block(self) -> Set[Box]:
        """Get all column neighbors outside of the box's block."""
        if self._col_minus_block is None:
            self._col_minus_block = self.col_neighbors - self.block_neighbors
        return self._col_minus_block

    @property
    def block_minus_row(self) -> Set[Box]:
        """Get all block neighbors outside of the box's row."""
        if self._block_minus_row is None:
            self._block_minus_row = self.block_neighbors - self.row_neighbors
        return self._block_minus_row

    @property
    def block_minus_col(self) -> Set[Box]:
        """Get all block neighbors outside of the box's column."""
        if self._block_minus_col is None:
            self._block_minus_col = self.block_neighbors - self.col_neighbors
        return self._block_minus_col


class Grid:
    """Grid storing all the values on the sudoku board."""

    def __init__(self):
        self._boxes: List[Box] = self.generate_boxes()
        self._rows = self.get_rows()
        self._cols = self.get_cols()
        self._blocks = self.get_blocks()
        self._neighborhood_types = [self._rows, self._cols, self._blocks]
        self.assign_neighbors()

    @property
    def boxes(self):
        """Get a list of all boxes in the grid."""
        return self._boxes

    @property
    def rows(self):
        """Get a list of rows, each containing a list of boxes in that row."""
        return self._rows

    @property
    def cols(self):
        """Get a list of columns, each containing a list of boxes in that column."""
        return self._cols

    @property
    def blocks(self):
        """Get a list of blocks, each containing a list of boxes in that block."""
        return self._blocks

    @property
    def neighborhood_types(self):
        """Get a list containing lists of the rows, columns and blocks."""
        return self._neighborhood_types

    @staticmethod
    def generate_boxes() -> List[Box]:
        """Returns 81 generated boxes to fill the grid."""
        return [Box(col, row) for col in range(1, 10)
                for row in range(1, 10)]

    def get_rows(self) -> List[List[Box]]:
        """Initialize the list of rows of boxes."""
        return [[box for box in self.boxes if box.row == i]
                for i in range(1, 10)]

    def get_cols(self) -> List[List[Box]]:
        """Initialize the list of columns of boxes."""
        return [[box for box in self.boxes if box.col == i]
                for i in range(1, 10)]

    def get_blocks(self) -> List[List[Box]]:
        """Initialize the list of blocks of boxes."""
        return [[box for box in self.boxes
                 if (box.row - 1) // 3 == i and (box.col - 1) // 3 == j]
                for i in range(3) for j in range(3)]

    def assign_neighbors(self):
        """Assign the row, column, and block neighbors for each box in the grid."""
        for row in self.rows:
            for box in row:
                box.row_neighbors = {*row} - {box}

        for col in self.cols:
            for box in col:
                box.col_neighbors = {*col} - {box}

        for block in self.blocks:
            for box in block:
                box.block_neighbors = {*block} - {box}

        for box in self.boxes:
            box.neighbors = box.row_neighbors | box.col_neighbors | box.block_neighbors
            box.neighborhood_sets = [box.row_neighbors, box.col_neighbors, box.block_neighbors]

    def insert_known_value(self, col: int, row: int, value: int):
        """Directly set the box at specified column/row to value."""
        for box in self.boxes:
            if box.col == col and box.row == row:
                box.possible_values = {value}

    def copy_grid(self) -> Grid:
        """Returns a deep copy of the grid"""
        new_grid = Grid()
        for old_box, new_box in zip(self.boxes, new_grid.boxes):
            new_box.possible_values = set(old_box.possible_values)
        return new_grid


class Solver:
    def __init__(self, grid: Grid):
        self._grid = grid

    @property
    def grid(self):
        """Returns the current grid."""
        return self._grid

    def possible_count(self) -> int:
        """Calculate the sum of the count of possible values among all boxes."""
        return sum(len(box.possible_values) for box in self.grid.boxes)

    def is_solved(self) -> bool:
        """Returns true if puzzle is solved."""
        return self.possible_count() == 81

    def solve(self) -> List[Tuple[int, int, int]] or None:
        """The main solving method.
        
        Repeats the basic solve loop until it it has solved the puzzle, or i no longer
        is reducing the total number of possible values.
        
        If the basic solve loop fails, then it will try the advanced heuristic loop
        which will continue until it has solved the puzzle or it has run out of box pairs
        to create new branches from.
        
        Note: Many of the solving rules are baked into the Box class itself. So each of
              boxes will try to reduce its possible values through its own properties, and
              not be explicitly called through these methods in the Solve class.
        """

        # The initial total count of possible values.
        prev_count = 9 * 81
        while prev_count > self.possible_count():
            # Run the grid through the basic Sudoku logic solving methods.
            # If a solution is found return it.
            while prev_count > self.possible_count():
                prev_count = self.possible_count()
                self.basic_solve_loop()
                if self.is_solved():
                    return self.solution_keys()
            # If the basic Sudoku logic methods weren't sufficient then use a heuristic
            # which guesses a value and checks whether the guessed value invalidates the grid.
            # After the guessing heuristic reduces the possible values, do the basic solve
            # loop above again.
            prev_count = self.possible_count()
            self.guess_between_two_possible_values_heuristic()
            if self.is_solved():
                return self.solution_keys()

        # If the basic solving methods and the guessing heuristic did not work
        # then use the advanced heuristic to try to solve.
        return self.advanced_heuristic()

    def basic_solve_loop(self):
        """Run some basic solving methods on the grid to try to reduce the possible values."""
        self.check_pairs()
        # TODO: Maybe delete this method below?
        self.check_block_col_row_intersections()
        self.test_for_invalid_grid()

    def check_pairs(self):
        """
        Find any matching pairs of possible values in a neighborhood and then remove those
        values from all their other neighbors.

        For example: If neighbor A and neighbor B have {7,2} as their possible values, then the rest of their neighbors
        cannot have 7 and 2
        """
        for box in self.grid.boxes:
            if len(box.possible_values) == 2:
                for neighborhood in box.neighborhood_sets:
                    for neighbor in neighborhood:
                        if neighbor.possible_values == box.possible_values:
                            for non_matching_neighbor in neighborhood - {neighbor}:
                                non_matching_neighbor.possible_values -= box.possible_values

    def check_block_col_row_intersections(self):
        """
        Check if a possible value is unique to a block/row or block/col intersection.
        If a value is unique to that intersection then the value cannot be in the rest of the block or row/col.
        """
        for box in self.grid.boxes:
            # Skip boxes that are already solved.
            if box.known_value:
                continue

            for value in box.possible_values:
                self.check_overlapping_neighbors_for_unique_value(value, box.block_neighbors, box.col_neighbors)
                self.check_overlapping_neighbors_for_unique_value(value, box.block_neighbors, box.row_neighbors)

    @staticmethod
    def check_overlapping_neighbors_for_unique_value(value: int, neighborhood_1: Set[Box], neighborhood_2: Set[Box]):
        """..."""
        # Get all possible values of boxes in neighborhood 1, excluding boxes from neighborhood 2.
        neighborhood_1_minus_2_possible_values = set().union(*[box.possible_values
                                                               for box in (neighborhood_1 - neighborhood_2)])
        # Get all possible values of boxes in neighborhood 2, excluding boxes from neighborhood 1.
        neighborhood_2_minus_1_possible_values = set().union(*[box.possible_values
                                                               for box in (neighborhood_2 - neighborhood_1)])
        # Get boxes in neighborhood 1 that aren't in neighborhood 2.
        neighborhood_1_minus_2_neighbors = neighborhood_1 - neighborhood_2
        # Get boxes in neighborhood 2 that aren't in neighborhood 1.
        neighborhood_2_minus_1_neighbors = neighborhood_2 - neighborhood_1

        #
        if value not in neighborhood_1_minus_2_possible_values:
            for neighbor in neighborhood_1_minus_2_neighbors:
                neighbor.possible_values -= {value}

        if value not in neighborhood_2_minus_1_possible_values:
            for neighbor in neighborhood_2_minus_1_neighbors:
                neighbor.possible_values -= {value}

    def guess_between_two_possible_values_heuristic(self):
        """
        For each box with 2 possible values, create a new grid that guesses between the two values. If it causes an
        invalid board then the box must be the other value.
        """
        for box in self.grid.boxes:
            if len(box.possible_values) == 2:
                a, b = list(box.possible_values)
                try:
                    self.simulate_grid_after_guess(box, a)
                except InvalidGrid:
                    box.possible_values = {b}

                try:
                    self.simulate_grid_after_guess(box, b)
                except InvalidGrid:
                    box.possible_values = {a}

    def simulate_grid_after_guess(self, box: Box, guess: int):
        """
        Copy the current grid and insert the guessed value. Then create a new Solver and run one basic solve loop.
        """
        new_grid = self.grid.copy_grid()
        new_grid.insert_known_value(box.col, box.row, guess)
        guess_solver = Solver(new_grid)
        guess_solver.basic_solve_loop()

    def advanced_heuristic(self) -> List[Tuple[int, int, int]] or None:
        """A guess and check heuristic which chooses a value for cells with two possible values."""
        
        # Create a list of branches based on box's that have exactly two possible values.
        # Each branch will be an instance of Solver with a grid which chooses a value from
        # the two possible values.
        branches: List[Solver] = self.create_branches_from_doubles()
        if not branches:
            # If there are no doubles, end this branch.
            return None
        for branch in branches:
            # Try to solve each new branch.
            # If it creates an invalid grid, move on to the next branch.
            try:
                return branch.solve()
            except InvalidGrid:
                pass

    def create_branches_from_doubles(self) -> List[Solver]:
        """
        Create a solver for each possible value among boxes with two possible values.
        """
        doubles: List[Box] = self.find_doubles()
        branches: List[Solver] = []
        for box in doubles:
            # Create a branch for each possible value.
            a, b = list(box.possible_values)[:2]
            branches.append(self.create_branch(box, a))
            branches.append(self.create_branch(box, b))
        return branches

    def create_branch(self, box: Box, value: int) -> Solver:
        """Create a new Solver instance with a copy of the grid and insert a known value guess."""
        new_grid = self.grid.copy_grid()
        new_grid.insert_known_value(box.col, box.row, value)
        return Solver(new_grid)

    def find_doubles(self) -> List[Box]:
        """Returns all boxes with possible values of length 2"""
        return [box for box in self.grid.boxes if len(box.possible_values) == 2]

    def test_for_invalid_grid(self):
        """Raises an InvalidGrid exception if two neighbors have the same known value.
        
        Raising this exception will be caught and used in the solving heuristics which
         require guessing and checking."""
        for box in self.grid.boxes:
            if box.known_value is None:
                continue
            for neighbor in box.neighbors:
                if box.known_value == neighbor.known_value:
                    raise InvalidGrid(f'Duplicate value: {box} and {neighbor} both are {box.known_value}')

    def solution_keys(self) -> List[Tuple[int, int, int]]:
        """Returns the grid as a list of tuples of the form: (column, row, value)."""
        return [(box.col, box.row, box.known_value) for box in self.grid.boxes]


def input_values_into_grid(grid: Grid, known_values: List[Tuple[int, int, int]]):
    """Adds the known values to the grid."""
    for known_value in known_values:
        grid.insert_known_value(*known_value)


def validate_known_values(known_values) -> str:
    """Returns false if any of the known_values conflict with each other."""
    for box in known_values:
        for other in (set(known_values) - {box}):
            if box[2] == other[2]:
                if box[0] == other[0]:
                    return "Error: Duplicate column values"
                if box[1] == other[1]:
                    return "Error: Duplicate row values"
                if (box[0] - 1) // 3 == (other[0] - 1) // 3 and (box[1] - 1) // 3 == (other[1] - 1) // 3:
                    return "Error: Duplicate block values"
    return ""


def solve(known_values: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]] or str:
    """Attempts to solve a Sudoku puzzle given a list of known_values.
    
    It will return a list of the values and coordinates if it succeeds.
    If it fails it will return an error describing what went wrong.
    
    Args:
        known_values: List of known values in the form of tuples of the
                      form (column, row, value)
    
    Returns:
        If successful:
            List of known values in the form of tuples in the form of (column, row, value)
        If unsuccessful:
            A string that describes the error that occurred.
    """
    if error := validate_known_values(known_values):
        return error

    grid = Grid()
    input_values_into_grid(grid, known_values)
    solver = Solver(grid)
    
    solution = solver.solve()
    
    if solution is None:
        return "Could not find solution"
    
    return solution
