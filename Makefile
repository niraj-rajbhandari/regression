.PHONY: all clean
all:
	python main.py

debug:
	python -m pdb main.py

clean:
	rm *.pyc
