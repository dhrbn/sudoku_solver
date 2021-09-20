import os
import plotly as py
import webbrowser

EXE_PATH, _ = os.path.split(os.path.realpath(__file__))
# EXE_PATH, _ = os.path.split(EXE_PATH)
FONT_PATH = os.path.join(EXE_PATH, 'fonts')


class PlotlyHtmlPage:
    def __init__(self, title=None, filepath=None):
        self.plots = []
        self.title = title
        self.filepath = filepath
        self.folder = os.path.split(filepath)[0] if filepath is not None else None
        self.fontpath = os.path.relpath(FONT_PATH, start=self.folder).replace('\\', '/')
        self.has_plotly_runtime = False  # to include static runtime only once!

    def add_text(self, title, text):
        id = len(self.plots)
        plot = dict(
            div=text,
            title=title,
            id='text-{}'.format(id),
            plot_resize=False
        )
        self.plots.append(plot)

    def set_title(self, title):
        self.title = title

    def set_filepath(self, filepath):
        self.filepath = filepath
        self.folder = os.path.split(filepath)[0] if filepath is not None else None

    def add_figure(self, figure, background_color='#ffffff'):
        id = len(self.plots)
        plot = dict(
            div=py.offline.plot(figure,
                                output_type='div',
                                include_plotlyjs=(not self.has_plotly_runtime),
                                image_width='100%',
                                image_height='100%',
                                config=dict(
                                    displaylogo=False,
                                )
                                ),
            title=figure.layout.title.text or 'plot #{}'.format(id),
            id='plot-{}'.format(id),
            plot_resize=True,
            # background_color=background_color,
        )
        self.plots.append(plot)
        self.has_plotly_runtime = True

    def save_to(self, auto_open=False):
        def tab_html(plot):
            return '''
				<button class="tab" id="tab-{id}" onclick="showPlot('{id}', {plot_resize:d})">{title}</button>
			'''.format(**plot)

        def panel_html(plot):
            return '''
			<div class="panel" id="panel-{id}">
				{div}
			</div>
			'''.format(**plot)

        script = '''
			function showPlot(tabId, autoResize) {
				for (const panel of document.getElementsByClassName("panel")) {
					panel.style.display = "none";
				}
				for (const tab of document.getElementsByClassName("tab")) {
					tab.className = tab.className.replace(" active", "");
				}
				const tabElt = document.getElementById(`tab-${tabId}`);
				const panelElt = document.getElementById(`panel-${tabId}`);
				tabElt.className += " active";
				panelElt.style.display = "block";
				if (autoResize) {
					Plotly.Plots.resize(panelElt.querySelector('.plotly-graph-div'));
				}
			} 
		'''

        # automatically activate the first tab only when all content is loaded (for efficient auto-resize)
        first_plot = self.plots[0]
        script += '''
			document.addEventListener('readystatechange', () => showPlot('{id}', {plot_resize:d}));
		'''.format(**first_plot)

        css = '''
            @font-face {
              font-family: 'Montserrat';
              src: '''
        css += f"url('{self.fontpath}/Montserrat-Bold.ttf');"
        css += '''
              font-weight: bold;
            }
            @font-face {
              font-family: 'Montserrat';
              src: '''
        css += f"url('{self.fontpath}/Montserrat-Medium.ttf');"
        css += '''
            }'''
        # h1 -> height: 60px; width: 100%; position: relative; overflow-y: hidden;
        # main -> overflow: hidden;
        css += '''

			* {
				font-family: Montserrat, "Century Gothic", sans-serif;
				box-sizing: border-box;
			}
			.main {
				position: absolute;
				left: 0; top: 0; right: 0; bottom: 0;
				display: flex;
				flex-direction: column;
				background-color: #1193f5;

			}
			h1 {
				height: 60px;
				color: #ffffff;
                overflow-x: hidden; 
                margin: 12px 0px 3px 5px;
                display: inline-block;
			}
			._tabs {
				display: flex;
				flex-direction: row;
				background-color: white;
			}
			.tab {
				height: 60px;
				width: 200px;
				border-radius: 0 0 5px 5px;
				border-bottom: 0;
				border-right: 0;
				border-left: 0;
				border-top: 0;
				cursor: pointer;
				background-color: #f0f1f5;
				font-size: 18px;
				font-weight: bold;
				color: #bac0cc;
				margin: 0 5px 0 0;
				border-color: #bac0cc;
			}
			.tab.active {
				color: white;
				font-weight: bold;
				background-color: #7251f3;
			}
			.tab:hover {
				border-top: 2px solid rgb(251,68,131);
				padding-top: 4px;
			}
			.panels {
				flex-grow: 2;
				background-color: white;
			}
			.panel {
				display: none;
			}
			.main, .panels, .panel, .panel > div, .panel .plotly-graph-div, .panel .plot-container, .panel .svg-container {
				width: 100%;
				height: 100%;
			}
		'''

        html = '''
			<!DOCTYPE html>
			<html>
				<head>
					<title>{title}</title>
					<meta charset="utf-8">
					<script type="text/javascript">{script}</script>
					<style>{css}</style>
				</head>
				<body>
					<div class="main">
						<h1>{title}</h1>
						<div class="_tabs">{tabs}</div>
						<div class="panels">{plots}</div>
					</div>
				</body>
			</html>
			'''.format(
            title=self.title,
            tabs=str("").join(map(tab_html, self.plots)),
            plots=str("").join(map(panel_html, self.plots)),
            script=script,
            css=css
        )
        with open(self.filepath, 'w') as f:
            f.write(html)
        if auto_open:
            print("save to %s" % self.filepath)
            url = "file://" + os.path.abspath(self.filepath)
            webbrowser.open(url)
        else:
            print(f'output-html-file: {os.path.abspath(self.filepath)}')
