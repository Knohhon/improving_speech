class UniqueWords:
    def __init__(self, text_rep: str):
        self.text_rep = str(text_rep)

    def unique(self):
        list_rep = self.text_rep.split(' ')
        words_rep = set(list_rep)
        return words_rep


class SimilarityMarker:

    # Категории слов. Цвет фона #E8E8E8
    # Слова-паразиты - #2A46E0
    # Слово находится не в том месте - #253946
    # Слова не было в эталонном тексте - #762CD0
    # Слово было в эталонном тексте но не прозвучало на репетиции - #156050

    def __init__(self, text_std: str, text_rep: str):
        self.words_std = UniqueWords(text_std.lower()).unique()
        self.words_rep = UniqueWords(text_rep.lower()).unique()
        self.list_std = text_std.lower().split(' ')
        self.list_rep = text_rep.lower().split(' ')

    def marker_not_in_std(self):
        in_std_and_rep = self.words_std.intersection(self.words_rep)
        not_in_std = self.words_rep
        for w in in_std_and_rep:
            not_in_std.remove(w)
        return not_in_std

    def marker_not_in_rep(self):
        in_std_and_rep = self.words_rep.intersection(self.words_std)
        not_in_rep = self.words_std
        for w in in_std_and_rep:
            not_in_rep.remove(w)
        return not_in_rep

    def marker_wrong_place(self):
        wrong_place = []
        not_in_rep = self.marker_not_in_rep()
        for i in range(len(self.list_std)):
            if self.list_std[i] != self.list_rep[i]:
                find = 0
                left_border = 0
                right_border = len(self.list_rep)
                if i - 15 > 0:
                    left_border = i - 15
                if i + 15 - len(self.list_rep) < 0:
                    right_border = i + 15
                for j in range(left_border, right_border):
                    if self.list_rep[j] == self.list_std[i]:
                        find = 1
                if find == 0 and self.list_std[i] not in not_in_rep:
                    wrong_place.append(self.list_std[i])
        return wrong_place


class MarkerBadWords:
    def __init__(self, filepath: str, text_rep: str):
        self.filepath = filepath
        self.bad_words_list = None
        self.words_rep = UniqueWords(text_rep).unique()
        self.text_rep = text_rep.lower().split(' ')

    def list_of_bad_words(self):
        with open(self.filepath, 'r', encoding='utf-8') as f:
            self.bad_words_list = f.read().split(', ')

    def bad_words(self):
        bad_words_in_rep = []
        for phrase in self.bad_words_list:
            list_phrase = phrase.split(' ')
            bad_words_in_rep += self.search_bad_words(list_phrase)
        return bad_words_in_rep

    def search_bad_words(self, list_phrase):
        index_of_bad_word = []
        for i in range(len(self.text_rep)):
            if self.text_rep[i] == list_phrase[0]:
                c = 0
                intermediate_index = []
                for j in range(len(list_phrase)):
                    if self.text_rep[i+j] == list_phrase[j]:
                        c += 1
                        intermediate_index.append(i+j)
                    if c == len(list_phrase):
                        index_of_bad_word += intermediate_index
        return index_of_bad_word


def main():
    abc = MarkerBadWords("parasites.txt", "Как бы не так")
    abc.list_of_bad_words()
    b = abc.bad_words()
    print(b)


if __name__ == '__main__':
    main()

