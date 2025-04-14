from gather_clues import *
import pandas as pd

def main():
    get_respective_answers('data/qnsdata.csv','data/qwenanswers.csv','data/qns_ans.csv')
    obtain_final_result('data/qns_ans.csv','data/normalise_ans_to_qns.csv', func=answer_to_question)
    obtain_final_result('data/normalise_ans_to_qns.csv','data/o3_answer_using_text.csv',func=finalise_from_answer)
    df=pd.read_csv('data/o3_answer_using_text.csv')
    remove_col(df, 'pred', 'data/o3_answer_using_text.csv')
    change_col_name('data/o3_answer_using_text.csv','answer','pred')
    remove_col(df, 'Unnamed: 0', 'data/o3_answer_using_text.csv')

if __name__ == "__main__":
    main()
