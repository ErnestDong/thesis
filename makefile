main:
	latexmk
view:
	open "thesis.pdf"
clean:
	latexmk -c
docx:
	pandoc -s thesis.tex --bibliography=thesis.bib -o thesis.docx
