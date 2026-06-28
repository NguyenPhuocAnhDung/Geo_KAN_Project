import docx2txt
import sys
text = docx2txt.process("tieulieuthamkhao/THAM KHẢO PHÁC THẢO BÀI BÁO Q1.docx")
print(text[:2000])
