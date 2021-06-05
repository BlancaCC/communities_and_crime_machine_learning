all: memoria.pdf
%.pdf: %.md
	pandoc --citeproc -o $@ $< --pdf-engine=xelatex 
