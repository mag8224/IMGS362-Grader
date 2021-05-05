
def create_output(textFileName):
    f = open(textFileName, "x")
    f.close()
    return textFileName


def append_score(score, textFileName):
    f = open(textFileName, "a")
    f.write(score)
    f.close()
    return score


def newLine(textFileName):
    f = open(textFileName, "a")
    f.write("\n")
    f.close()
    return
