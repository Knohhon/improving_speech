from speech_rec import finaltext

path = "filler_words.txt"


def fwords(pth):
    with open(pth, "r", encoding='utf-8-sig') as filler_words:
        list_filler_words = filler_words.read().split(', ')
        dict_filler_words = dict.fromkeys(list_filler_words, 0)
        return dict_filler_words


def search_in_audiofile(pth):
    bad_filler = []
    filler = fwords(pth)
    words = finaltext.split(' ')
    for w in words:
        if w in filler.keys():
            filler[w] += 1
        else:
            continue
        if 3 <= filler[w] < 5:
            bad_filler.append(w)
            bad_filler[w] = "#FFCF40"
        elif filler[w] >= 5:
            bad_filler[w] = "#B00000"
    return bad_filler


print(search_in_audiofile(path))

