import pandas as pd
from src.chooser import choose_document
from src.database import parse_dataset,create_batches
from src.editor import edit_document
from src.search import perform_search
from src.utils import save_object
from src.evaluator import evaluate, evaluate_diff
import tqdm


def batch_search_vectorized(df, connector, query_col='query', sources_col='cleaned_sources', response_col='response') -> pd.DataFrame:
    def search_row(row):
        return perform_search(row[query_col], row[sources_col], connector)

    # Apply function to each row
    tqdm.tqdm.pandas(desc="Processing queries")  # Enable progress bar
    df = df.copy()
    df[response_col] = df.progress_apply(search_row, axis=1)
    return df

def batch_evaluate(df, response_col = 'response', sources_col = 'cleaned_sources', evaluation_col='evaluation_results') -> pd.DataFrame:
    def evaluate_row(row):
        return evaluate(row[response_col], row[sources_col])

    # Apply function to each row
    tqdm.tqdm.pandas(desc="Processing evaluations")  # Enable progress bar
    df = df.copy()
    df[evaluation_col] = df.progress_apply(evaluate_row, axis=1)
    return df

def batch_choose_edit(df, method, connector, cumulative, format) -> pd.DataFrame:

    def choose_doc_row(row):
        scores = [(a*0.5+b*0.5) for a,b in row['evaluation_results']]
        return choose_document(row['cleaned_sources'], scores)
    def edit_doc_row(row,cumulative):
        edited_doc = edit_document(method, row['cleaned_sources'][row['choosen_doc_idx']], query=row['query'], connector=connector)
        if cumulative:
            return format.format(source=row['cleaned_sources'][row['choosen_doc_idx']], section=edited_doc)
        return edited_doc

    def replace_doc_row(row):
        sources = row['cleaned_sources'].copy()
        sources[row['choosen_doc_idx']] = row['choosen_doc_edited']
        return sources

    tqdm.tqdm.pandas(desc="Choosing and Editing Document")
    df = df.copy()
    df['choosen_doc_idx'] = df.progress_apply(choose_doc_row, axis=1)
    df['choosen_doc_edited'] = df.progress_apply(edit_doc_row,cumulative=cumulative, axis=1)
    df['cleaned_sources'] = df.progress_apply(replace_doc_row, axis=1)
    return df

def batch_evaluate_diff(df, old_results_col = 'evaluation_results', new_results_col = 'evaluation_results_new', output_col = 'evaluation_diff') -> pd.DataFrame:
    def evaluate_diff_row(row):
        return evaluate_diff(row[old_results_col], row[new_results_col])
    tqdm.tqdm.pandas(desc="Processing evaluation differences")
    df = df.copy()
    df[output_col] = df.progress_apply(evaluate_diff_row, axis=1)
    return df

def run_pipeline(original_df,connector, batch_size, batch_timeout, save_intermediate=True, saving_path = './search_results/'):

    ## Preprocessing
    print("Starting preprocessing...")
    # df = clean_dataset(original_df)
    df = parse_dataset(original_df)
    batches = create_batches(df, batch_size=batch_size)
    if batch_timeout:
        batches = batches[:batch_timeout]  # Limit to batch_timeout batches if specified

    ## Searching
    print("Starting search...")
    for i, batch in enumerate(batches):
        print(f"Processing batch {i+1}/{len(batches)}")
        print(f"Batch size: {len(batch)}")
        if not batch.get('batch_nr'):
            batch = batch_search_vectorized(batch, connector)
            batch['batch_nr'] = i + 1
            batches[i] = batch

    if save_intermediate:
        save_object(batches, saving_path + 'search_results.pkl')

    ## Evaluating
    print("Starting evaluation...")

    for i, batch in enumerate(batches):
        print(f"Evaluating batch {i+1}/{len(batches)}")
        batch = batch_evaluate(batch)
        batches[i] = batch

    if save_intermediate:
        save_object(batches, saving_path + 'search_results_evaluated.pkl')

    return pd.concat(batches, ignore_index=True)


def run_method(df,method,connector, batch_size, batch_timeout,edit_prompt, cumulative,save_intermediate=True, saving_path = './search_results/'):

    print("Starting preprocessing...")
    df = parse_dataset(df) 
    batches = create_batches(df, batch_size)
    if batch_timeout:
        batches = batches[:batch_timeout]

    for i, batch in enumerate(batches):
        print(f"Processing batch {i+1}/{len(batches)}")

        print("Choosing and editing documents...")
        batch = batch_choose_edit(batch, method, connector,cumulative=cumulative,format=edit_prompt)
        if save_intermediate:
            save_object(batch,saving_path +f'method_{method}_batch{i+1}.pkl')

        print("Searching documents...")
        batch = batch_search_vectorized(batch, connector, query_col='query', sources_col='cleaned_sources', response_col='response_new')
        if save_intermediate:
            save_object(batch,saving_path +f'method_{method}_batch{i+1}.pkl')

        print("Evaluating documents...")
        batch = batch_evaluate(batch,response_col='response_new',sources_col='cleaned_sources',evaluation_col='evaluation_results_new')
        if save_intermediate:
            save_object(batch,saving_path +f'method_{method}_batch{i+1}.pkl')


        print("Evaluating the differences")
        batch = batch_evaluate_diff(batch, old_results_col='evaluation_results', new_results_col='evaluation_results_new', output_col='evaluation_diff')
        if save_intermediate:
            save_object(batch,saving_path +f'method_{method}_batch{i+1}.pkl')

        batch['batch_nr'] = i + 1
        batches[i] = batch


    if save_intermediate:
        save_object(batches, saving_path + f'method_{method}.pkl')

    return pd.concat(batches, ignore_index=True)