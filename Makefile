.PHONY: install run

install:
	pip install -r requirements.txt

run:
	streamlit run main.py
