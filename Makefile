.PHONY: all debug clean menu help
all:
	python main.py -d $(data_set)

debug:
	python -m pdb main.py -d $(data_set)  -m

menu:
	python main.py -d $(data_set) -m

help:
	python main.py -h

clean:
	rm *.pyc
