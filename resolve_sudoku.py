import os
from sudoku_solver import Sudoku
from sudoku_image import SudokuImage
from plotly_html_page import PlotlyHtmlPage

os.makedirs('results', exist_ok=True)

# im_file = 'data/sudoku1.png'
# TODO: A 6 detected as an 8

# im_file = 'data/sudoku2.png'  # fails to detect the 5 at the top right, detects it as an 8
# WORKS WITH get_transformed_image instead of get_rotated_image

# im_file = 'data/sudoku3.png'  # fails to detect the grid, rotation is not sufficient, there is also a stretching
# WORKS WITH get_transformed_image instead of get_rotated_image

# im_file = 'data/sudoku4.png'  # fails first row of the grid detection
# WORKS WITH get_transformed_image instead of get_rotated_image

# im_file = 'data/sudoku5.png'  # grid and number detection works but filling algo needs to be improved
# WORKS WITH get_transformed_image instead of get_rotated_image

# im_file = 'data/sudoku6.png'  # works
# WORKS WITH get_transformed_image instead of get_rotated_image

# im_file = 'data/sudoku7.png'  # fails, there is also a stretching
# WORKS WITH get_transformed_image instead of get_rotated_image

# im_file = 'data/sudoku8.png'  # fails to detect the 9 at the bottom left, detects it as an 8
# WORKS WITH get_transformed_image instead of get_rotated_image

for i in range(1, 12):
# for i in range(9, 10):
    im_file = f'data/sudoku{i}.png'

    page = PlotlyHtmlPage(f"{im_file.split('/')[-1]}", f"results/{im_file.split('/')[-1]}.html")

    si = SudokuImage(im_file)
    page.add_figure(si.get_rotated_image_figure())
    df = si.extract_sudoku()
    page.add_figure(si.get_grid_lines_figure())
    page.add_figure(si.get_numbers_detection())

    sdk = Sudoku(df)
    sdk.resolve()
    page.add_figure(sdk.get_grid_filling())

    page.save_to(auto_open=True)
