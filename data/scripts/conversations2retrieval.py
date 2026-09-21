# convert workbench file into BEIR json retrieval format:
# queries.jsonl 
#   {"_id": "0", "text": "What is considered a business expense on a business trip?", "metadata": {}}
# qrels/test.tsv
#   query-id        corpus-id       score
#   8       566392  1

import json
import os
import pandas as pd
import argparse

def read_json(filename: str, encoding: str = "utf-8"):
    with open(filename, mode="r", encoding=encoding) as fp:
        return json.load(fp)


def write_json(filename: str, content: list | dict, encoding: str = "utf-8"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode="w", encoding=encoding) as fp:
        return json.dump(content, fp)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        dest="input",
        help="Path containing dataset to run",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        dest="output_dir",
        help="Path containing dataset to run",
    )
    parser.add_argument(
        "-t",
        "--turns_to_keep",
        type=int,
        default=-1,
        dest="turns_to_keep",
        help="Which turns to keep. It includs user + agent so: -1 is the last turn, -3 is the current question + previous q+a, 0 is full conversation",
    )
    parser.add_argument(
        "-q",
        "--q_only",
        action="store_true",
        dest="q_only",
        help="Only use the questions and not the responses",
    )    
    return parser.parse_args()

if __name__ == "__main__":
    # Step 1: Read command line arguments, environment variables and runtime configuration
    args = parse_args()
    queries = {}
    qrels = {}

    wb_conversations = read_json(args.input)

    wb_index = -1

    # Step 2: iterate through conversations
    for wb_conversation in wb_conversations:
        wb_index += 1

        conversation = []
        collection_name = wb_conversation['retriever']['collection']['name']

        if collection_name not in queries:
            queries[collection_name] = []
            qrels[collection_name] = []

        _id = ""
        rewrite = True
        m_index = -1

        # Step 3: iterate through messages in conversation. Each message is a turn
        for message in wb_conversation['messages']:
            m_index += 1

            # Step 4: track turns
            if not args.q_only or (args.q_only and message['speaker'] == 'user'):
                conversation.append(f"|{message['speaker']}|: {message['text']}")
            if message['speaker'] == 'user':        
                _id = f"{wb_conversation['author']}_{message['timestamp']}"

                # Step 4a: apply turn logic 
                queries[collection_name].append({"_id": f"{_id}", "text": '\n'.join(conversation[args.turns_to_keep:])})
            # Step 5: check responses for unanswerables (these turns are skipped, but the turn will still be part of other tasks) 
            else:
                unanswerable = True
                for context in message['contexts']:
                    if 'feedback' in context and (('editor' in wb_conversation and wb_conversation['editor'] in context['feedback']['relevant'] and context['feedback']['relevant'][wb_conversation['editor']]['value'] == 'yes') \
                        or (('editor' not in wb_conversation or wb_conversation['editor'] not in context['feedback']['relevant']) and context['feedback']['relevant'][wb_conversation['author']]['value'] == 'yes')):
                        qrels[collection_name].append({'query-id': f"{_id}", 'corpus-id': context['document_id'], 'score': 1})
                        unanswerable = False
                if unanswerable:
                   del queries[collection_name][-1]

    # Step 6: save files in BEIR retrieval format 
    for collection_name in queries:
        os.makedirs(f"{args.output_dir}/retrieval/{collection_name}/qrels/", exist_ok=True)
        pd.DataFrame(queries[collection_name]).to_json(f"{args.output_dir}/retrieval/{collection_name}/queries_turns{args.turns_to_keep}_qonly{args.q_only}.jsonl",lines=True, orient='records')
        pd.DataFrame(qrels[collection_name]).to_csv(f"{args.output_dir}/retrieval/{collection_name}/qrels/dev.tsv", index=False, sep='\t')