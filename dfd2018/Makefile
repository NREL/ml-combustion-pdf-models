#
#  Provides 'make all' to build and 'make clean'
#

FNAME=report

all: $(FNAME).pdf

$(FNAME).pdf: $(FNAME).tex
	rm -f *.aux *.nav *.log *.lof *.lot *.out *.snm *.blg *.bbl *.toc *.vrb *.dvi *.spl *.ps;
	pdflatex $(FNAME);
	pdflatex $(FNAME);
	# bibtex $(FNAME);
	# pdflatex $(FNAME);
	# pdflatex $(FNAME);

clean:
	-rm -f *.aux *.nav *.log *.lof *.lot *.out *.snm *.blg *.bbl *.toc *.vrb *.dvi *.spl *.ps;
