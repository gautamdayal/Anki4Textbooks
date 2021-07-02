import re
import PyPDF2     
# creating a pdf file object 
pdfFileObj = open('../books/beato_book.pdf', 'rb') 
# creating a pdf reader object 
pdfReader = PyPDF2.PdfFileReader(pdfFileObj) 
print(pdfReader.numPages) 
    
# creating a page object 
pageObj = pdfReader.getPage(4) 
    
# extracting text from page 
txt = pageObj.extractText()
print(txt)

# closing the pdf file object 
pdfFileObj.close() 

# f = False

# for line in txt:
#     if re.match("^[A-Z]+$", line):
#         if f: f.close()
#         f = open(line + '.txt', 'w')

#     else:
#         f.write(line + "\n")