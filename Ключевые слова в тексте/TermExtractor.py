from rutermextract import TermExtractor
from pandas import DataFrame

term_extractor = TermExtractor()
l1 = []
l2 = []

filename = input("Введите полный путь к файлу:")
with open(filename, 'r') as file:
    text = file.read().replace('\n', '')
    
for term in term_extractor(text):
    l1.append(term.normalized)
    l2.append(term.count)

    
file.close()

df = DataFrame({'Term': l1, 'Frequency': l2})
df.to_excel("term.xlsx")