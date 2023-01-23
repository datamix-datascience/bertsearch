"""
Example script to create elasticsearch documents.
"""
import argparse
import json

import pandas as pd
from bert_serving.client import BertClient
bc = BertClient(output_fmt='list')


def create_document(doc, emb, index_name):
    return {
        '_op_type': 'index',
        '_index': index_name,
        'student_name': doc['student_name'],
        'kana': doc['kana'],
        'term': doc['term'],
        'thema': doc['thema'],
        'link': doc['link'],
        'Release': doc['Release'],
        'score': doc['score'],
        'comments': doc['comments'],
        'doc_text': doc['doc_text'],
        'documents_vector': emb
    }


def load_dataset(path):
    docs = []
    df = pd.read_csv(path)
    for row in df.iterrows():
        series = row[1]
        doc = {
            'student_name': series.student_name,
            'kana': series.kana,
            'term': series.term,
            'thema': series.thema,
            'link': series.link,
            'Release': series.Release,
            'score': series.score,
            'comments': series.comments,
            'doc_text': series.thema + series.doc_text
        }
        docs.append(doc)
    return docs


def bulk_predict(docs, batch_size=256):
    """Predict bert embeddings."""
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i: i+batch_size]
        embeddings = bc.encode([doc['doc_text'] for doc in batch_docs])
        for emb in embeddings:
            yield emb


def main(args):
    docs = load_dataset(args.data)
    with open(args.save, 'w') as f:
        for doc, emb in zip(docs, bulk_predict(docs)):
            d = create_document(doc, emb, args.index_name)
            f.write(json.dumps(d) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating elasticsearch documents.')
    parser.add_argument('--data', help='data for creating documents.')
    parser.add_argument('--save', default='documents.jsonl', help='created documents.')
    parser.add_argument('--index_name', default='jobsearch', help='Elasticsearch index name.')
    args = parser.parse_args()
    main(args)
