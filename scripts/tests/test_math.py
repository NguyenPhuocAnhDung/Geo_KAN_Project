import docx
from docx.oxml import parse_xml
doc = docx.Document()
p = doc.add_paragraph()
math_xml = '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"><m:r><m:rPr><m:scr m:val="roman"/></m:rPr><m:t>Shift</m:t></m:r></m:oMath>'
try:
    el = parse_xml(math_xml)
    p._p.append(el)
    doc.save("test_math.docx")
    print("Success")
except Exception as e:
    print(e)
