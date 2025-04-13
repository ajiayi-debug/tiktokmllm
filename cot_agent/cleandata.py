from preprocessqns import *
import pandas as pd


df=pd.read_csv("qnsdata.csv")
# Regex pattern to extract the list portion
list_regex = r"\[.*\]"
df["breakdown"]=df["breakdown"].apply(lambda x: extract_list(x, list_regex))


df.to_csv("clean_qnsdata.csv")
