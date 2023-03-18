from spacy import displacy
import language
import utils




# utils.print_all_translations()
# translation_translator = 'fa#1'
# pips = 'dep,pos,root,lemma'

# nlp = language.Pipeline(pips, translation_translator)
# doc = nlp('27#30')

# options = {"compact": True, "bg": "#09a3d5",
#            "color": "white", "font": "xb-niloofar"}
# displacy.serve(doc, style="dep", options=options)

'''
    2#1
    soure : 2
    ayeh : 1
    split : #
'''
# doc = nlp('1#1')
# doc = nlp('آل عمران # 4')
# doc = nlp('حمد لله رب العالمین')
import language

translation_translator = 'fa#1'
pips = 'dep,pos,root,lemma'
nlp = language.Pipeline(pips, translation_translator)

doc = nlp('3#1')

print(doc)
print(doc._.text)
print(doc._.surah)
print(doc._.ayah)
print(doc._.revelation_order)
print(doc._.sim_ayahs)
print(doc._.translations)
