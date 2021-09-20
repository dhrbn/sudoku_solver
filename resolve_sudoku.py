import os
from sudoku_solver import Sudoku
from sudoku_image import SudokuImage
from plotly_html_page import PlotlyHtmlPage

os.makedirs('results', exist_ok=True)

for i in range(1, 12):
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
