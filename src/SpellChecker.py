from autocorrect import Speller

def correct_sentence(line):
    spell = Speller()
    lines = line.strip().split(' ')
    new_line = ""
    similar_word = {}
    for l in lines:
        new_line += spell(l) + " "
    # similar_word[l]=spell.candidates(l)
    return new_line