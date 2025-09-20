import pandas as pd
import ast
import tiktoken


def clean_dataset(df):

    df['cleaned_sources'] = df['sources'].apply(lambda sources: parse_string_to_list(sources, column_name='cleaned_text'))
    df['url'] = df['sources'].apply(lambda sources: parse_string_to_list(sources, column_name='url'))
    df['num_tokens_sources'] = df['sources'].apply(lambda sources: [len(tiktoken.encoding_for_model('gpt-4o').encode(source)) for source in sources])

    return df

def parse_dataset(df):

    list_columns = ['cleaned_sources', 'url', 'num_tokens_sources', 'evaluation_results']

    for col in list_columns:
        df[col] = df[col].apply(lambda x: parse_text_to_list(x))

    return df

def create_batches(df, batch_size=32):
    return [df.iloc[i:i + batch_size] for i in range(0, len(df), batch_size)]

def parse_text_to_list(text:str) -> list:
        """
        Safely evaluates a string containing a Python literal (list/dict)
        into the corresponding Python object.
        """
        return ast.literal_eval(text)


def parse_string_to_list(sources:str, column_name:str):
    
    return [a[column_name] for a in parse_text_to_list(sources.replace("}","},"))]