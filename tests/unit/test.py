from spacy import displacy
import language
import utils

# utils.print_all_translations()
translation_translator = 'fa#1'
pips = 'dep,pos,root,lemma'

nlp = language.Pipeline(pips, translation_translator)

'''
    2#1
    soure : 2
    ayeh : 1
    split : #
'''
doc = nlp('1#1')
doc = nlp('آل عمران # 4')
doc = nlp('حمد لله رب العالمین')
print(doc)
print(doc._.revelation_order)
print(doc._.ayah)
print(doc._.surah)
print(doc._.sim_ayahs)
print(doc._.translations)

for d in doc:
    print(d)
    print(d.head)
    print(d.dep_)
    print(d._.dep_arc)
    print(d._.root)
    print(d.lemma_)
    print(d.pos_)
    print('s')

dictionary = language.to_json(pips, doc)
print(dictionary)
