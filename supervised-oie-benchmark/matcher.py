from __future__ import division
import string
from nltk.translate.bleu_score import sentence_bleu
from nltk.corpus import stopwords

class Matcher:
    @staticmethod
    def bowMatch(ref, ex, ignoreStopwords, ignoreCase):
        """
        A binary function testing for exact lexical match (ignoring ordering) between reference
        and predicted extraction
        """
        s1 = ref.bow()
        s2 = ex.bow()
        if ignoreCase:
            s1 = s1.lower()
            s2 = s2.lower()

        s1Words = s1.split(' ')
        s2Words = s2.split(' ')

        if ignoreStopwords:
            s1Words = Matcher.removeStopwords(s1Words)
            s2Words = Matcher.removeStopwords(s2Words)

        return sorted(s1Words) == sorted(s2Words)

    @staticmethod
    def predMatch(ref, ex, ignoreStopwords, ignoreCase):
        """
        Return whehter gold and predicted extractions agree on the predicate
        """
        s1 = ref.elementToStr(ref.pred)
        s2 = ex.elementToStr(ex.pred)
        if ignoreCase:
            s1 = s1.lower()
            s2 = s2.lower()

        s1Words = s1.split(' ')
        s2Words = s2.split(' ')

        if ignoreStopwords:
            s1Words = Matcher.removeStopwords(s1Words)
            s2Words = Matcher.removeStopwords(s2Words)

        return s1Words  == s2Words


    @staticmethod
    def argMatch(ref, ex, ignoreStopwords, ignoreCase):
        """
        Return whehter gold and predicted extractions agree on the arguments
        """
        sRef = ' '.join([ref.elementToStr(elem) for elem in ref.args])
        sEx = ' '.join([ex.elementToStr(elem) for elem in ex.args])

        count = 0

        for w1 in sRef:
            for w2 in sEx:
                if w1 == w2:
                    count += 1

        # We check how well does the extraction lexically cover the reference
        # Note: this is somewhat lenient as it doesn't penalize the extraction for
        #       being too long
        coverage = float(count) / len(sRef)


        return coverage > Matcher.LEXICAL_THRESHOLD

    @staticmethod
    def bleuMatch(ref, ex, ignoreStopwords, ignoreCase):
        sRef = ref.bow()
        sEx = ex.bow()
        bleu = sentence_bleu(references = [sRef.split(' ')], hypothesis = sEx.split(' '))
        return bleu > Matcher.BLEU_THRESHOLD

    @staticmethod
    def lexicalMatch(ref, ex, ignoreStopwords, ignoreCase):
        sRef = ref.bow().split(' ')
        sEx = ex.bow().split(' ')
        count = 0
        #print(sRef, sEx)
        for w1 in sRef:
        #    for w2 in sEx:
            if w1 in sEx:
                count += 1
                sEx.remove(w1)

        # We check how well does the extraction lexically cover the reference
        # Note: this is somewhat lenient as it doesn't penalize the extraction for
        #       being too long
        coverage = float(count) / len(sRef)
        #print(sRef, sEx, count, coverage)

        return coverage > Matcher.LEXICAL_THRESHOLD

    @staticmethod
    def tuple_match(ref, ex, ignoreStopwords, ignoreCase):
        precision = [0, 0] # 0 out of 0 predicted words match
        recall = [0, 0] # 0 out of 0 reference words match
        # If, for each part, any word is the same as a reference word, then it's a match.

        # print ref
        # print ex
        predicted_words = ex.pred.split()
        gold_words = ref.pred.split()

        matching_words = sum(1 for w in predicted_words if w in gold_words)
        if matching_words == 0:
            return False # t <-> gt is not a match
        precision[0] += matching_words
        precision[1] += len(predicted_words)
        recall[0] += matching_words
        recall[1] += len(gold_words)

        # print ex.args
        # print ref.args
        for i in range(len(ref.args)):
            # print precision, recall
            gold_words = ref.args[i].split()
            recall[1] += len(gold_words)
            if len(ex.args) <= i:
                if i<2:
                    return False
                else:
                    continue
            predicted_words = ex.args[i].split()
            matching_words = sum(1 for w in predicted_words if w in gold_words)
            # print gold_words, predicted_words, matching_words
            if matching_words == 0 and i<2:
                    return False # t <-> gt is not a match
            precision[0] += matching_words
            precision[1] += len(predicted_words)
            # Currently this slightly penalises systems when the reference
            # reformulates the sentence words, because the reformulation doesn't
            # match the predicted word. It's a one-wrong-word penalty to precision,
            # to all systems that correctly extracted the reformulated word.
            recall[0] += matching_words

        # print precision, recall
        prec = 1.0 * precision[0] / precision[1]
        rec = 1.0 * recall[0] / recall[1]
        # print prec, rec
        return [prec, rec]

    @staticmethod
    def removeStopwords(ls):
        return [w for w in ls if w.lower() not in Matcher.stopwords]

    # CONSTANTS
    BLEU_THRESHOLD = 0.4
    LEXICAL_THRESHOLD = 0.5 # Note: changing this value didn't change the ordering of the tested systems
    stopwords = stopwords.words('english') + list(string.punctuation)





