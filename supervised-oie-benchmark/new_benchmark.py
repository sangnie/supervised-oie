'''
Usage:
   benchmark --gold=GOLD_OIE --out=OUTPUT_FILE (--openiefive=OPENIE5 | --stanford=STANFORD_OIE | --ollie=OLLIE_OIE |--reverb=REVERB_OIE | --clausie=CLAUSIE_OIE | --openiefour=OPENIEFOUR_OIE | --props=PROPS_OIE | --tabbed=TABBED_OIE | --benchmarkGold=BENCHMARK_GOLD)) [--exactMatch | --predMatch | --argMatch] [--error-file=ERROR_FILE]

Options:
  --gold=GOLD_OIE              The gold reference Open IE file (by default, it should be under ./oie_corpus/all.oie).
  --benchmarkgold=GOLD_OIE     The benchmark's gold reference. 
  --out-OUTPUT_FILE            The output file, into which the precision recall curve will be written.
  --clausie=CLAUSIE_OIE        Read ClausIE format from file CLAUSIE_OIE.
  --ollie=OLLIE_OIE            Read OLLIE format from file OLLIE_OIE.
  --openiefour=OPENIEFOUR_OIE  Read Open IE 4 format from file OPENIEFOUR_OIE.
  --openiefive=OPENIE5         Read Open IE 5 format from file OPENIE5.
  --props=PROPS_OIE            Read PropS format from file PROPS_OIE
  --reverb=REVERB_OIE          Read ReVerb format from file REVERB_OIE
  --stanford=STANFORD_OIE      Read Stanford format from file STANFORD_OIE
  --tabbed=TABBED_OIE          Read simple tab format file, where each line consists of:
                                sent, prob, pred,arg1, arg2, ...
  --exactmatch                 Use exact match when judging whether an extraction is correct.
'''
from __future__ import division
import docopt
import string
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import re
import logging
import pdb
logging.basicConfig(level = logging.INFO)

from oie_readers.stanfordReader import StanfordReader
from oie_readers.ollieReader import OllieReader
from oie_readers.reVerbReader import ReVerbReader
from oie_readers.clausieReader import ClausieReader
from oie_readers.openieFourReader import OpenieFourReader
from oie_readers.openieFiveReader import OpenieFiveReader
from oie_readers.propsReader import PropSReader
from oie_readers.tabReader import TabReader
from oie_readers.benchmarkGoldReader import BenchmarkGoldReader

from oie_readers.goldReader import GoldReader
from matcher import Matcher
from operator import itemgetter

class Benchmark:
    ''' Compare the gold OIE dataset against a predicted equivalent '''
    def __init__(self, gold_fn):
        ''' Load gold Open IE, this will serve to compare against using the compare function '''
        gr = GoldReader()
        gr.read(gold_fn)
        self.gold = gr.oie

    def compare(self, predicted, matchingFunc, output_fn, error_file = None):
        ''' Compare gold against predicted using a specified matching function.
            Outputs PR curve to output_fn '''

        y_true = []
        y_scores = []
        errors = []
        correct = 0
        incorrect = 0

        correctTotal = 0
        unmatchedCount = 0
        predicted = Benchmark.normalizeDict(predicted)
        gold = Benchmark.normalizeDict(self.gold)

        results = {}
        p = [0 for _ in np.linspace(0,1,500)]
        pl = [0 for _ in np.linspace(0,1,500)]
        r = [0 for _ in np.linspace(0,1,500)]
        rl = [0 for _ in np.linspace(0,1,500)]
        # total_scores = []
        for sent, goldExtractions in gold.items():
            # if sent not in predicted:
            #     # The extractor didn't find any extractions for this sentence
            #     for goldEx in goldExtractions:
            #         unmatchedCount += len(goldExtractions)
            #         correctTotal += len(goldExtractions)
            #     continue
            # print sent

            if sent in predicted:
                # print "yay"
                predictedExtractions = predicted[sent]
            else:
                predictedExtractions = []

            scores = [[None for _ in predictedExtractions] for __ in goldExtractions]
            # print scores

            # print scores
            for i, goldEx in enumerate(goldExtractions):
                for j, predictedEx in enumerate(predictedExtractions):
                    # exact_match_scores[i][j] = Matcher.tuple_exact_match(predictedEx, goldEx)
                    score = Matcher.tuple_match(goldEx, predictedEx,ignoreStopwords = True,ignoreCase = True)
                    # print score
                    scores[i][j] = score
            # total_scores.append(scores)
            # for c, conf in enumerate(range(0,1,0.05)):
            # print scores
            # print len(scores), len(scores[0])
            for c, conf in enumerate(np.linspace(0,1,500)):
                for row in scores:
                    # print row
                    # r[c] += max([score[1] if score and predictedExtractions[col].confidence >= conf else 0 for col, score in enumerate(row)] )
                    r[c] += max([score[1] for col, score in enumerate(row) if score and predictedExtractions[col].confidence >= conf] or [0] )
                    # print([(score[1], predictedExtractions[col].confidence) for col, score in enumerate(row) if score and predictedExtractions[col].confidence >= conf])
                rl[c] += len(scores)

                if len(scores[0]) > 0:
                    for j in range(len(scores[0])):
                        if predictedExtractions[j].confidence >= conf:
                            p[c] += max([scores[i][j][0] if scores[i][j] else 0 for i in range(len(scores))])
                            # print [scores[i][j][0] for i in range(len(scores)) if scores[i][j]]
                            pl[c] += 1


        # --------------------------------------
        # y_true = y_true
        # y_scores = y_scores
        # print(correct, incorrect, unmatchedCount)
        # # recall on y_true, y  (r')_scores computes |covered by extractor| / |True in what's covered by extractor|
        # # to get to true recall we do:
        # # r' * (|True in what's covered by extractor| / |True in gold|) = |true in what's covered| / |true in gold|
        # (p, r), optimal = Benchmark.prCurve(np.array(y_true), np.array(y_scores),
        #                                     recallMultiplier = ((correctTotal - unmatchedCount)/float(correctTotal)))
        # logging.info("AUC: {}\n Optimal (precision, recall, F1, threshold): {}".format(auc(r, p),
        #                                                                                optimal))
        # # Write error log to file
        # if error_file:
        #     logging.info("Writing {} error indices to {}".format(len(errors),
        #                                                          error_file))
        #     with open(error_file, 'w') as fout:
        #         fout.write('\n'.join([str(error)
        #                              for error
        #                               in errors]) + '\n')
        #print [(a,b) for a,b in zip(p,pl)]
        #print [(a,b) for a,b in zip(r,rl)]
        prec_scores = [a/b if b>0 else 1 for a,b in zip(p,pl) ]
        rec_scores = [a/b if b>0 else 0 for a,b in zip(r,rl)]
        #print prec_scores
        #print rec_scores

        # # write PR to file
        with open(output_fn, 'w') as fout:
            fout.write('{0}\t{1}\n'.format("Precision", "Recall"))
            for cur_p, cur_r in sorted(zip(prec_scores, rec_scores), key = lambda (cur_p, cur_r): cur_r):
                fout.write('{0}\t{1}\n'.format(cur_p, cur_r))

    @staticmethod
    def f1(prec, rec):
        try:
            return 2*prec*rec / (prec+rec)
        except ZeroDivisionError:
            return 0

    @staticmethod
    def aggregate_scores_greedily(scores):
        # Greedy match: pick the prediction/gold match with the best f1 and exclude
        # them both, until nothing left matches. Each input square is a [prec, rec]
        # pair. Returns precision and recall as score-and-denominator pairs.
        matches = []
        while True:
            max_s = 0
            gold, pred = None, None
            for i, gold_ss in enumerate(scores):
                if i in [m[0] for m in matches]:
                    # Those are already taken rows
                    print i, matches
                    continue
                for j, pred_s in enumerate(scores[i]):
                    if j in [m[1] for m in matches]:
                        # Those are used columns
                        print j, matches
                        continue
                    if pred_s and Benchmark.f1(*pred_s) > max_s:
                        max_s = Benchmark.f1(*pred_s)
                        gold = i
                        pred = j
            if max_s == 0:
                break
            matches.append([gold, pred])
        # Now that matches are determined, compute final scores.
        prec_scores = [scores[i][j][0] for i,j in matches]
        rec_scores = [scores[i][j][1] for i,j in matches]
        total_prec = sum(prec_scores)
        total_rec = sum(rec_scores)
        scoring_metrics = {"precision" : [total_prec, len(scores[0])],
                           "recall" : [total_rec, len(scores)],
                           "precision_of_matches" : prec_scores,
                           "recall_of_matches" : rec_scores
        }
        # print(scoring_metrics)
        return scoring_metrics

    @staticmethod
    def prCurve(y_true, y_scores, recallMultiplier):
        # Recall multiplier - accounts for the percentage examples unreached
        # Return (precision [list], recall[list]), (Optimal F1, Optimal threshold)
        y_scores = [score \
                    if not (np.isnan(score) or (not np.isfinite(score))) \
                    else 0
                    for score in y_scores]
        
        precision_ls, recall_ls, thresholds = precision_recall_curve(y_true, y_scores)
        recall_ls = recall_ls * recallMultiplier
        optimal = max([(precision, recall, f_beta(precision, recall, beta = 1), threshold)
                       for ((precision, recall), threshold)
                       in zip(zip(precision_ls[:-1], recall_ls[:-1]),
                              thresholds)],
                      key = itemgetter(2))  # Sort by f1 score

        return ((precision_ls, recall_ls),
                optimal)

    # Helper functions:
    @staticmethod
    def normalizeDict(d):
        return dict([(Benchmark.normalizeKey(k), v) for k, v in d.items()])

    @staticmethod
    def normalizeKey(k):
        return Benchmark.removePunct(unicode(Benchmark.PTB_unescape(k.replace(' ','')), errors = 'ignore'))

    @staticmethod
    def PTB_escape(s):
        for u, e in Benchmark.PTB_ESCAPES:
            s = s.replace(u, e)
        return s

    @staticmethod
    def PTB_unescape(s):
        for u, e in Benchmark.PTB_ESCAPES:
            s = s.replace(e, u)
        return s

    @staticmethod
    def removePunct(s):
        return Benchmark.regex.sub('', s)

    # CONSTANTS
    regex = re.compile('[%s]' % re.escape(string.punctuation))

    # Penn treebank bracket escapes
    # Taken from: https://github.com/nlplab/brat/blob/master/server/src/gtbtokenize.py
    PTB_ESCAPES = [('(', '-LRB-'),
                   (')', '-RRB-'),
                   ('[', '-LSB-'),
                   (']', '-RSB-'),
                   ('{', '-LCB-'),
                   ('}', '-RCB-'),]


def f_beta(precision, recall, beta = 1):
    """
    Get F_beta score from precision and recall.
    """
    beta = float(beta) # Make sure that results are in float
    return (1 + pow(beta, 2)) * (precision * recall) / ((pow(beta, 2) * precision) + recall)


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    logging.debug(args)

    if args['--stanford']:
        predicted = StanfordReader()
        predicted.read(args['--stanford'])

    if args['--props']:
        predicted = PropSReader()
        predicted.read(args['--props'])

    if args['--ollie']:
        predicted = OllieReader()
        predicted.read(args['--ollie'])

    if args['--reverb']:
        predicted = ReVerbReader()
        predicted.read(args['--reverb'])

    if args['--clausie']:
        predicted = ClausieReader()
        predicted.read(args['--clausie'])

    if args['--openiefour']:
        predicted = OpenieFourReader()
        predicted.read(args['--openiefour'])

    if args['--openiefive']:
        predicted = OpenieFiveReader()
        predicted.read(args['--openiefive'])

    if args['--benchmarkGold']:
        predicted = BenchmarkGoldReader()
        predicted.read(args['--benchmarkGold'])
 
    if args['--tabbed']:
        predicted = TabReader()
        predicted.read(args['--tabbed'])

    if args['--exactMatch']:
        matchingFunc = Matcher.argMatch

    elif args['--predMatch']:
        matchingFunc = Matcher.predMatch

    elif args['--argMatch']:
        matchingFunc = Matcher.argMatch

    else:
        matchingFunc = Matcher.lexicalMatch

    b = Benchmark(args['--gold'])
    out_filename = args['--out']

    logging.info("Writing PR curve of {} to {}".format(predicted.name, out_filename))
    b.compare(predicted = predicted.oie,
              matchingFunc = matchingFunc,
              output_fn = out_filename,
              error_file = args["--error-file"])
