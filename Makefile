.PHONY: all debug clean menu help
all:
	python main.py

debug:
	python -m pdb main.py -m

menu:
	python main.py -m

help:
	python main.py -h

clean:
	rm *.pyc
