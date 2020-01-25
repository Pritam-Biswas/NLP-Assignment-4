def predict_nearest_with_context(self, context):
    stopwords = nltk.stopwords.words('english')
    sent_list = []

    count = 0
    for word in context.left_context:
        if count< 2 and word.lower() not in stopwords:
            sent_list.append(word.lower())
            count+=1
        else:
            break

    sent_list.append(context.lemma)
    count = 0
    for word in context.right_context:
        if count< 2 and word.lower() not in stopwords:
            sent_list.append(word.lower())
            count+=1
        else:
            break

    sentence  = ' '.join(sent_list)
    emb = self.get_embeddings(sentence)

    max_sim, max_sim_word = 0, None
    syn = get_candidates(context.lemma, context.pos)
    for word in syn:
        try:
            word_emb = self.model.wv[word]
        except KeyError:
            word_emb = np.zeros(300,)
        sim = self.get_similarity_by_vec(emb, word_emb)
        if max_sim < sim:
            max_sim = sim
            max_sim_word = word

    return max_sim_word
    # return None # replace for part 5
def get_embeddings(self,word):
    wd = word.split()
    emb_sum= np.zeros(300,)
    for i in wd:
        if i in self.model.vocab:
            emb_sum += np.array(self.model.wv[i])
    return emb_sum
def predict_nearest(self,context):
    syn = get_candidates(context.lemma, context.pos)
    max_sim, max_sim_word = 0, None
    for word in syn:
        try:
            sim = self.model.similarity(word,context.lemma)
        except KeyError:
            sim = 0
        if max_sim < sim:
            max_sim = sim
            max_sim_word = word
    return max_sim_word