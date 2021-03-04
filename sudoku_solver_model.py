from __future__ import annotations
from typing import List, Set, Tuple
from itertools import combinations
import time


class Box:
    def __init__(self, col, row):
        self._col = col
        self._row = row

        self._known_value = None
        self._possible_values = set(range(1, 10))

        self.row_neighbors = set()
        self.col_neighbors = set()
        self.block_neighbors = set()
        self.neighbors = set()
        self.neighborhood_sets = set()

        self._row_neighbors_possible_values = None
        self._col_neighbors_possible_values = None
        self._block_neighbors_possible_values = None

        self._combinations_size_2 = None
        self._combinations_size_3 = None
        self._combinations_size_4 = None

        self._row_block_intersection = None
        self._col_block_intersection = None

        self._row_minus_block = None
        self._col_minus_block = None

        self._block_minus_row = None
        self._block_minus_col = None

    def __repr__(self):
        return f'Box({self.col}, {self.row})'

    @property
    def row(self) -> int:
        return self._row

    @property
    def col(self) -> int:
        return self._col

    @property
    def known_value(self) -> int or None:
        return self._known_value

    @property
    def possible_values(self) -> Set[int]:
        for neighbor in self.neighbors:
            if neighbor.known_value:
                self.possible_values = self._possible_values - {neighbor.known_value}
        return self._possible_values

    @possible_values.setter
    def possible_values(self, value: Set[int]):
        if len(value) == 0:
            raise AssertionError(f'Box({self.col}, {self.row}) - Empty set!')
        if len(value) == 1 and self._known_value is None:
            self._known_value = list(value)[0]
        self._possible_values = value

        self._combinations_size_2 = None
        self._combinations_size_3 = None
        self._combinations_size_4 = None

        self._row_neighbors_possible_values = None
        self._col_neighbors_possible_values = None
        self._block_neighbors_possible_values = None

    @property
    def row_neighbors_possible_values(self) -> Set[int]:
        return set(possible_value for neighbor in self.row_neighbors
                   for possible_value in neighbor.possible_values)

    @property
    def col_neighbors_possible_values(self) -> Set[int]:
        return set(possible_value for neighbor in self.col_neighbors
                   for possible_value in neighbor.possible_values)

    @property
    def block_neighbors_possible_values(self) -> Set[int]:
        return set(possible_value for neighbor in self.block_neighbors
                   for possible_value in neighbor.possible_values)

    @property
    def combinations_size_2(self) -> List[Set[int]]:
        if self._combinations_size_2 is None:
            self._combinations_size_2 = [set(combo) for combo in combinations(self.possible_values, 2)]
        return self._combinations_size_2

    @property
    def combinations_size_3(self) -> List[Set[int]]:
        if self._combinations_size_3 is None:
            self._combinations_size_3 = [set(combo) for combo in combinations(self.possible_values, 3)]
        return self._combinations_size_3

    @property
    def combinations_size_4(self) -> List[Set[int]]:
        if self._combinations_size_4 is None:
            self._combinations_size_4 = [set(combo) for combo in combinations(self.possible_values, 4)]
        return self._combinations_size_4

    @property
    def row_block_intersection(self) -> Set[Box]:
        if self._row_block_intersection is None:
            self._row_block_intersection = self.row_neighbors & self.block_neighbors
        return self._row_block_intersection

    @property
    def col_block_intersection(self) -> Set[Box]:
        if self._col_block_intersection is None:
            self._col_block_intersection = self.col_neighbors & self.block_neighbors
        return self._col_block_intersection

    @property
    def row_minus_block(self) -> Set[Box]:
        if self._row_minus_block is None:
            self._row_minus_block = self.row_neighbors - self.block_neighbors
        return self._row_minus_block

    @property
    def col_minus_block(self) -> Set[Box]:
        if self._col_minus_block is None:
            self._col_minus_block = self.col_neighbors - self.block_neighbors
        return self._col_minus_block

    @property
    def block_minus_row(self) -> Set[Box]:
        if self._block_minus_row is None:
            self._block_minus_row = self.block_neighbors - self.row_neighbors
        return self._block_minus_row

    @property
    def block_minus_col(self) -> Set[Box]:
        if self._block_minus_col is None:
            self._block_minus_col = self.block_neighbors - self.col_neighbors
        return self._block_minus_col


class Grid:
    def __init__(self):
        self._boxes = self.generate_boxes()
        self._rows = self.get_rows()
        self._cols = self.get_cols()
        self._blocks = self.get_blocks()
        self._neighborhood_types = [self._rows, self._cols, self._blocks]
        self.assign_neighbors()

    @property
    def boxes(self):
        return self._boxes

    @property
    def rows(self):
        return self._rows

    @property
    def cols(self):
        return self._cols

    @property
    def blocks(self):
        return self._blocks

    @property
    def neighborhood_types(self):
        return self._neighborhood_types

    @staticmethod
    def generate_boxes() -> List[Box]:
        return [Box(col, row) for col in range(1, 10)
                for row in range(1, 10)]

    def get_rows(self) -> List[List[Box]]:
        return [[box for box in self.boxes if box.row == i]
                for i in range(1, 10)]

    def get_cols(self) -> List[List[Box]]:
        return [[box for box in self.boxes if box.col == i]
                for i in range(1, 10)]

    def get_blocks(self) -> List[List[Box]]:
        return [[box for box in self.boxes
                 if (box.row - 1) // 3 == i and (box.col - 1) // 3 == j]
                for i in range(3) for j in range(3)]

    def assign_neighbors(self):
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
        """Manually set the value of a certain box"""
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
        return self._grid

    def print_possible(self):
        """Print a grid of possible values, row by row."""
        for row in self.grid.rows:
            print(
                [str(box.possible_values).center(14) if len(box.possible_values) > 1 else ''.center(14) for box in row])

    def print_known(self):
        """Print a grid of known values, row by row."""
        for row in self.grid.rows:
            print([str(box.known_value) if box.known_value else ' ' for box in row])

    def known_count(self) -> int:
        """Calculate the number of Boxes with a known value"""
        return sum(1 for box in self.grid.boxes if box.known_value)

    def possible_count(self) -> int:
        """Calculate the sum of the count of possible values among all boxes"""
        return sum(len(box.possible_values) for box in self.grid.boxes)

    def print_info_stuff(self):
        """Print information on the current grid"""
        print(f'Found: {self.known_count()} among 81')
        print(f'Possible Values {self.possible_count()}')
        print(f'Possibilities to Eliminate {self.possible_count() - 81}')
        self.print_possible()
        self.print_known()

    def check_if_solved(self):
        """Checks if puzzle is solved. Displays the solution and solving time then quits the program."""
        if self.possible_count() == 81:
            self.print_info_stuff()
            print('Quitting')
            end = time.time()
            print(f'Total time: {end - start}')
            quit()

    def solve(self):
        prev_count = 729
        while prev_count > self.possible_count():
            while prev_count > self.possible_count():
                prev_count = self.possible_count()
                self.basic_solve_loop()
                self.check_if_solved()
            prev_count = self.possible_count()
            self.guess_between_two_possible_values_heuristic()
            self.check_if_solved()
        self.advanced_heuristic()

        # If it reaches here, it is unsolvable
        self.print_info_stuff()
        print('Unsolvable Puzzle')

    def basic_solve_loop(self):
        self.check_pairs()
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
            if box.known_value:
                continue

            for value in box.possible_values:
                self.check_overlapping_neighbors_for_unique_value(value, box.block_neighbors, box.col_neighbors)
                self.check_overlapping_neighbors_for_unique_value(value, box.block_neighbors, box.row_neighbors)

    @staticmethod
    def check_overlapping_neighbors_for_unique_value(value: int, neighborhood_1: Set[Box], neighborhood_2: Set[Box]):
        neighborhood_1_minus_2_possible_values = set().union(*[box.possible_values
                                                               for box in (neighborhood_1 - neighborhood_2)])
        neighborhood_2_minus_1_possible_values = set().union(*[box.possible_values
                                                               for box in (neighborhood_2 - neighborhood_1)])
        neighborhood_1_minus_2_neighbors = neighborhood_1 - neighborhood_2
        neighborhood_2_minus_1_neighbors = neighborhood_2 - neighborhood_1

        if value not in neighborhood_1_minus_2_possible_values:
            for neighbor in neighborhood_2_minus_1_neighbors:
                neighbor.possible_values -= {value}

        if value not in neighborhood_2_minus_1_possible_values:
            for neighbor in neighborhood_1_minus_2_neighbors:
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
                except AssertionError:
                    box.possible_values = {b}

                try:
                    self.simulate_grid_after_guess(box, b)
                except AssertionError:
                    box.possible_values = {a}

    def simulate_grid_after_guess(self, box: Box, guess: int):
        """
        Copy the current grid and insert the guessed value. Then create a new Solver and run one basic solve loop.
        """
        new_grid = self.grid.copy_grid()
        new_grid.insert_known_value(box.col, box.row, guess)
        guess_solver = Solver(new_grid)
        guess_solver.basic_solve_loop()

    def advanced_heuristic(self):
        branches = self.create_branches_from_doubles()
        if not branches:
            self.print_info_stuff()
            print('Unable to solve')
            quit()
        for branch in branches:
            try:
                branch.solve()
            except AssertionError:
                pass

    def create_branches_from_doubles(self) -> List[Solver]:
        """
        Create a solver for each possible value among boxes with two possible values
        """
        doubles = self.find_doubles()
        branches = []
        for box in doubles:
            a, b = list(box.possible_values)[:2]
            branches.append(self.create_branch(box, a))
            branches.append(self.create_branch(box, b))
        return branches

    def create_branch(self, box: Box, value: int) -> Solver:
        new_grid = self.grid.copy_grid()
        new_grid.insert_known_value(box.col, box.row, value)
        return Solver(new_grid)

    def find_doubles(self):
        """Returns all boxes with possible values of length 2"""
        return [box for box in self.grid.boxes if len(box.possible_values) == 2]

    def test_for_invalid_grid(self):
        for box in self.grid.boxes:
            if box.known_value is None:
                continue
            for neighbor in box.neighbors:
                if box.known_value == neighbor.known_value:
                    print('Grid before error:')
                    self.print_possible()
                    raise AssertionError(f'Duplicate value: {box} and {neighbor} both are {box.known_value}')


def input_values_into_grid(grid_: Grid, known_values: Tuple[int, int, int]):
    for known_value in known_values:
        grid_.insert_known_value(*known_value)


def solve(known_values):
    grid = Grid()
    input_values_into_grid(my_grid, known_values)
    solver = Solver(grid)
    solver.solve()
