# -*- coding: utf-8 -*-


def extract():
    return


def expand():
    return


def oov():
    return


def planning(query):
    keywords = query.split(' - ')
    if len(keywords) < 4:
        for i in range(4 - len(keywords)):
            keywords.append(keywords[-1])
    elif len(keywords) > 4:
        keywords = keywords[:4]
    return keywords
