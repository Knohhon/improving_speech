from sentence_transformers import SentenceTransformer, util


class SimilarityCos:

    # Text must be str()
    def __init__(self, text_std, text_rep):
        self.text_std = text_std
        self.text_rep = text_rep

    def model(self):
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        return model

    def embed(self):
        model = self.model()
        embd1 = model.encode(self.text_std, convert_to_tensor=True)
        embd2 = model.encode(self.text_rep, convert_to_tensor=True)
        return embd1, embd2

    def similarity_num(self):
        embd1, embd2 = self.embed()
        cos_score = util.cos_sim(embd1, embd2)
        return cos_score
