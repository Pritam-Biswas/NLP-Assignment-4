from nltk.corpus import wordnet
synonyms = []
antonyms = []
print (wordnet.synsets("active", pos = 'n'))
for syn in wordnet.synsets("active"):
	print (syn)
	print (syn.lemmas())
	for l in syn.lemmas():
		synonyms.append(l.name())
		if l.antonyms():
			# print (l.antonyms())
			antonyms.append(l.antonyms()[0].name())

	print ()

print(set(synonyms))
print(set(antonyms))