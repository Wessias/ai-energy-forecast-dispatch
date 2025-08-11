.PHONY: venv install run fast clean

venv:
	python -m venv .venv

install: venv
	. .venv/bin/activate && pip install -r requirements.txt
	# Windows: .venv\Scripts\activate && pip install -r requirements.txt

run:
	. .venv/bin/activate && python -m src.main
	# Windows: .venv\Scripts\activate && python -m src.main

fast:
	. .venv/bin/activate && python -m src.main --fast
	# Windows: .venv\Scripts\activate && python -m src.main --fast

clean:
	rm -rf models reports/figures
