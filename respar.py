from docling.document_converter import DocumentConverter
from bs4 import BeautifulSoup



source = r"c:\Users\ABHISHRUT\Desktop\HarshNewResume.pdf"  
converter = DocumentConverter()
result = converter.convert(source)
print(result.document.export_to_markdown()) 

html_doc = result.document.export_to_html()
soup = BeautifulSoup(html_doc, 'html.parser')
print(soup.prettify())