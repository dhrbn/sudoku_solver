import numpy as np
import figure_factory as ff
import matplotlib.pyplot as plt


def _get_area(row, col):
    if row in ['A', 'B', 'C']:
        if col in list(range(3)):
            return 'alpha'
        elif col in list(range(3, 6)):
            return 'bravo'
        else:
            return 'charlie'
    elif row in ['D', 'E', 'F']:
        if col in list(range(3)):
            return 'delta'
        elif col in list(range(3, 6)):
            return 'echo'
        else:
            return 'foxtrot'
    else:
        if col in list(range(3)):
            return 'golf'
        elif col in list(range(3, 6)):
            return 'hotel'
        else:
            return 'india'


class Sudoku:
    def __init__(self, df):
        self.df = df.copy(deep=True)
        self.gold_df = ~df.isnull()
        self.cells = []
        rows = []
        cols = []
        areas = []
        for i, row in df.iterrows():
            for c in df.columns:
                area = _get_area(i, c)
                value = None if np.isnan(df.at[i, c]) else int(df.at[i, c])
                self.cells.append(Cell(i, c, area, value=value))
                rows.append(i)
                cols.append(c)
                areas.append(area)

        self.rows = list(set(rows))
        self.cols = list(set(cols))
        self.areas = list(set(areas))

    def resolve(self, max_iter=50):
        n_possible = 1000
        idx = 0
        while idx < max_iter and n_possible > 81:
            idx += 1

            # Updating possible values
            for cell in self.cells:
                if cell.value is not None:
                    for cc in self.cells:
                        if cc is not cell and cc.value is None:
                            if cc.row == cell.row or cc.col == cell.col or cc.area == cell.area:
                                cc.update_possible_values([cell.value])

            # Checking if a value is the only in its area, row or col
            for cell in self.cells:
                if cell.value is None:
                    for pv in cell.possible_values:
                        is_only_pv = True
                        for cc_area in [cc for cc in self.cells if cc is not cell and cc.area == cell.area]:
                            if pv in cc_area.possible_values:
                                is_only_pv = False
                                break
                        if is_only_pv:
                            cell.update_possible_values([pval for pval in cell.possible_values if pval != pv])

                        is_only_pv = True
                        for cc_row in [cc for cc in self.cells if cc is not cell and cc.row == cell.row]:
                            if pv in cc_row.possible_values:
                                is_only_pv = False
                                break
                        if is_only_pv:
                            cell.update_possible_values([pval for pval in cell.possible_values if pval != pv])

                        is_only_pv = True
                        for cc_col in [cc for cc in self.cells if cc is not cell and cc.col == cell.col]:
                            if pv in cc_col.possible_values:
                                is_only_pv = False
                                break
                        if is_only_pv:
                            cell.update_possible_values([pval for pval in cell.possible_values if pval != pv])

            n_possible = 0
            for cell in self.cells:
                n_possible += len(cell.possible_values)

        for i, row in self.df.iterrows():
            for c in self.df.columns:
                self.df.at[i, c] = [cell.value for cell in self.cells if cell.id == f'{i}{c}'][0]

    def iterate(self):
        pass

    def iplot(self):
        plt.figure()
        for i in range(0, 10):
            width = 3 if i in [0, 3, 6, 9] else 1
            plt.plot([i, i], [0, 9], color='k', linewidth=width)
            plt.plot([0, 9], [i, i], color='k', linewidth=width)

        for idx, i_row in enumerate(self.df.iterrows()):
            i, row = i_row
            for c in self.df.columns:
                if self.df.at[i, c] is not None and not np.isnan(self.df.at[i, c]):
                    weight = 'bold' if self.gold_df.at[i, c] else 'normal'
                    color = 'b' if self.gold_df.at[i, c] else 'k'
                    plt.annotate(f'{self.df.at[i, c]:.0f}', (c + 0.3, 8 - idx + 0.45), color=color, weight=weight)
        plt.show()

    def get_grid_filling(self):
        return ff.get_grid_filling(self)


class Cell:
    def __init__(self, row, col, area, value=None):
        self.row = row
        self.col = col
        self.area = area
        self.value = value
        self.gold_value = value is not None
        self.id = f'{row}{col}'
        self.possible_values = list(range(1, 10)) if value is None else [value]

    def update_possible_values(self, to_remove=None):
        if self.value is None and to_remove is not None:
            self.possible_values = [v for v in self.possible_values if v not in to_remove]
            if len(self.possible_values) == 1:
                self.value = self.possible_values[0]
                # print(f'New value for {self.id}: {self.value}')

    def __repr__(self):
        return f'Cell object. Row: {self.row}. Column: {self.col}. Area: {self.area}. Value: {self.value}. ' \
               f'Possible values: {self.possible_values}'
