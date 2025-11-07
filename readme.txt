TODO: 
	automate venv creation
	automate build and serve
		start litho_sim_venv
		jupyter lite build --output-dir docs --contents content --force
		jupyter lite serve --output-dir=docs
		navigate to 127.0.0.1:8000
	instructions for how to use repo in this readme file
	roadmap sketch


activate python environment
pip install marimo


How to start working?
- open command prompt in the repo
- activate the virtual environment
	litho_sim_venv\Scripts\activate

To create or edit a marimo notebook;
	marimo edit <path_of_notebook>

to make the wheel:
	python -m build --wheel

to test; 
	pip install pytest
	run pytest 
