import pandas as pd
from src.pipeline import run_method
from src.method_eval import summarize_differences, print_diff_summary, batch_evaluate_diff
from connector.chatgpt import ChatGPTConnector




if __name__ == "__main__":

    connector = ChatGPTConnector("chatgpt-4o-latest")
    method = 'addELI5' 
    df = pd.read_csv("./data/processed_test.csv")


    # df = run_pipeline(
    #     original_df = df,
    #     connector = connector,
    #     batch_size = 50,
    #     batch_timeout = 10,
    #     save_intermediate = True,
    #     saving_path = './search_results/'
    # )
    # format = "{section}\n\n\n  {source} "
    # df = run_method(
    #     df,
    #     method=method,
    #     connector=connector,
    #     cumulative=True,
    #     edit_prompt= format,
    #     batch_size=5,
    #     batch_timeout=1,
    #     save_intermediate=True,
    #     saving_path='./search_results/'
    # )
    # print(df)
    # print(df.to_csv(f"./data/method_{method}.csv", index=False))

    methods = ['structure','queryCenteric','summary','Trainingbias','faq','addFreshness','full_qna','speech','author','addELI5']
    for method in methods:
        df = pd.read_pickle(f'search_results/method_{method}.pkl')
        results_list = batch_evaluate_diff(df, concat=False)
        for b, res in enumerate(results_list):
            summary = summarize_differences(res, diff_col="evaluation_diff", index_col="choosen_doc_idx") # type: ignore
        print_diff_summary(summary, method_name=f"{method} (Batch {b})") # type: ignore