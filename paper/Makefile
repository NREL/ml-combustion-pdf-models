#
#  Provides 'make all' to build and 'make clean'
#

FNAME=report

all: figures $(FNAME).pdf

$(FNAME).pdf: $(FNAME).tex
	rm -f *.aux *.nav *.log *.lof *.lot *.out *.snm *.blg *.bbl *.toc *.vrb *.dvi *.spl *.ps *.xml *.upa *.upb;
	pdflatex $(FNAME);
	pdflatex $(FNAME);
	bibtex $(FNAME);
	pdflatex $(FNAME);
	pdflatex $(FNAME);

figures:
	- convert -density 300 exp_lsb.png ./figure01.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=1 -dLastPage=1 -sOutputFile=./figure02a.pdf src_pv.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=2 -dLastPage=2 -sOutputFile=./figure02b.pdf src_pv.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=3 -dLastPage=3 -sOutputFile=./figure02c.pdf src_pv.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=4 -dLastPage=4 -sOutputFile=./figure02d.pdf src_pv.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=5 -dLastPage=5 -sOutputFile=./figure02e.pdf src_pv.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=6 -dLastPage=6 -sOutputFile=./figure02f.pdf src_pv.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=1 -dLastPage=1 -sOutputFile=./figure03a.pdf dice_0004_slice.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=2 -dLastPage=2 -sOutputFile=./figure03b.pdf dice_0004_slice.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=3 -dLastPage=3 -sOutputFile=./figure03c.pdf dice_0004_slice.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=4 -dLastPage=4 -sOutputFile=./figure03d.pdf dice_0004_slice.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=9 -dLastPage=9 -sOutputFile=./figure04a.pdf pdfs_dice_0004.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=10 -dLastPage=10 -sOutputFile=./figure04b.pdf pdfs_dice_0004.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=11 -dLastPage=11 -sOutputFile=./figure04c.pdf pdfs_dice_0004.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=12 -dLastPage=12 -sOutputFile=./figure04d.pdf pdfs_dice_0004.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=1 -dLastPage=1 -sOutputFile=./figure05a.pdf inputs_dice_0004.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=2 -dLastPage=2 -sOutputFile=./figure05b.pdf inputs_dice_0004.pdf
	- pdfcrop --margins '10 0 20 10' --clip ./figure05a.pdf ./figure05a.pdf
	- pdfcrop --margins '10 100 10 0' --clip ./figure05b.pdf ./figure05b.pdf
	- montage -density 300 -geometry +0+0 ./figure05a.pdf ./figure05b.pdf ./figure05.pdf
	- rm ./figure05a.pdf ./figure05b.pdf
	- cp gen_data.pdf ./figure06.pdf
	- cp rf.pdf ./figure07a.pdf
	- cp cvae.pdf ./figure07b.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=3 -dLastPage=3 -sOutputFile=./figure08al.pdf pdfs_50429.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=4 -dLastPage=4 -sOutputFile=./figure08ar.pdf pdfs_50429.pdf
	- montage -density 300 -geometry +0+0 ./figure08al.pdf ./figure08ar.pdf ./figure08a.pdf
	- rm ./figure08al.pdf ./figure08ar.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=3 -dLastPage=3 -sOutputFile=./figure08bl.pdf pdfs_44209.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=4 -dLastPage=4 -sOutputFile=./figure08br.pdf pdfs_44209.pdf
	- montage -density 300 -geometry +0+0 ./figure08bl.pdf ./figure08br.pdf ./figure08b.pdf
	- rm ./figure08bl.pdf ./figure08br.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=3 -dLastPage=3 -sOutputFile=./figure08cl.pdf pdfs_14523.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=4 -dLastPage=4 -sOutputFile=./figure08cr.pdf pdfs_14523.pdf
	- montage -density 300 -geometry +0+0 ./figure08cl.pdf ./figure08cr.pdf ./figure08c.pdf
	- rm ./figure08cl.pdf ./figure08cr.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=1 -dLastPage=1 -sOutputFile=./figure09a.pdf jsd_dice_0004.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=2 -dLastPage=2 -sOutputFile=./figure09b.pdf jsd_dice_0004.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=1 -dLastPage=1 -sOutputFile=./figure10a.pdf convolution_dice_0004.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=2 -dLastPage=2 -sOutputFile=./figure10b.pdf convolution_dice_0004.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=4 -dLastPage=4 -sOutputFile=./figure11.pdf pdf_distances.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=1 -dLastPage=1 -sOutputFile=./figure12al.pdf dice_predictions_0004.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=2 -dLastPage=2 -sOutputFile=./figure12ar.pdf dice_predictions_0004.pdf
	- montage -density 300 -geometry +0+0 ./figure12al.pdf ./figure12ar.pdf ./figure12a.pdf
	- rm ./figure12al.pdf ./figure12ar.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=1 -dLastPage=1 -sOutputFile=./figure12bl.pdf dice_predictions_0006.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=2 -dLastPage=2 -sOutputFile=./figure12br.pdf dice_predictions_0006.pdf
	- montage -density 300 -geometry +0+0 ./figure12bl.pdf ./figure12br.pdf ./figure12b.pdf
	- rm ./figure12bl.pdf ./figure12br.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=1 -dLastPage=1 -sOutputFile=./figure12cl.pdf dice_predictions_skip.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=2 -dLastPage=2 -sOutputFile=./figure12cr.pdf dice_predictions_skip.pdf
	- montage -density 300 -geometry +0+0 ./figure12cl.pdf ./figure12cr.pdf ./figure12c.pdf
	- rm ./figure12cl.pdf ./figure12cr.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=1 -dLastPage=1 -sOutputFile=./figure13a.pdf convolution_skip.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=3 -dLastPage=3 -sOutputFile=./figure13b.pdf convolution_skip.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=4 -dLastPage=4 -sOutputFile=./figure13c.pdf convolution_skip.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=3 -dLastPage=3 -sOutputFile=./figure14al.pdf pdfs_66089.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=4 -dLastPage=4 -sOutputFile=./figure14ar.pdf pdfs_66089.pdf
	- montage -density 300 -geometry +0+0 ./figure14al.pdf ./figure14ar.pdf ./figure14a.pdf
	- rm ./figure14al.pdf ./figure14ar.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=3 -dLastPage=3 -sOutputFile=./figure14bl.pdf pdfs_159948.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=4 -dLastPage=4 -sOutputFile=./figure14br.pdf pdfs_159948.pdf
	- montage -density 300 -geometry +0+0 ./figure14bl.pdf ./figure14br.pdf ./figure14b.pdf
	- rm ./figure14bl.pdf ./figure14br.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=3 -dLastPage=3 -sOutputFile=./figure14cl.pdf pdfs_182976.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=4 -dLastPage=4 -sOutputFile=./figure14cr.pdf pdfs_182976.pdf
	- montage -density 300 -geometry +0+0 ./figure14cl.pdf ./figure14cr.pdf ./figure14c.pdf
	- rm ./figure14cl.pdf ./figure14cr.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=1 -dLastPage=1 -sOutputFile=./figure15a.pdf dice_predictions_comp_1030.pdf
	- gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dFirstPage=2 -dLastPage=2 -sOutputFile=./figure15b.pdf dice_predictions_comp_1030.pdf

clean:
	-rm -f $(FNAME).pdf *.aux *.acn *.glo *.ist *.nav *.log *.lof *.lot *.out *.snm *.blg *.bbl *.toc *.vrb *.dvi *.spl *.ps *.xml *.upa *.upb figure*.pdf figure*.png;
