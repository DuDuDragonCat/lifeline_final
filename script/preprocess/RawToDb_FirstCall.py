# -- coding: utf-8 --
# %%
import os
import re
import copy
import sqlite3
import glob
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from itertools import chain

# %%
# ckip模型設置
from ckiptagger import WS, POS, construct_dictionary

# 必免ckiptagger 使用半精度 float32
import tensorflow as tf

tf.config.experimental.enable_tensor_float_32_execution(False)
MODEL_PATH = "/home/dudulon/GitHub/ckipmodel"
ws = WS(MODEL_PATH, disable_cuda=False)
pos = POS(MODEL_PATH)
UserDict = {"生命線": 1, "安心專線": 1, "張老師": 1, "馬上上工": 1}
UserDict = construct_dictionary(UserDict)

# %%
# 20220126暫時排除的文本，這些文本
# RM_FIES = ["20131024185700", "20141129231620", "20150621170339", "20181106073727", "20130419043245",
#            "20130427201752", "20150325180647", "20160817102746", "20161001234829", "20160627140732",
#            "20171201193554"]
PATH_FOLDER_ROOT = "/home/dudulon/GitHub/lifeline_final/data/FirstCall"
FOLDERNAME_TRANS = "transcript"
# /nas中不同資料夾的逐字稿/年份/座位/撥打時間.csv
PATH_FILE_TRANS = glob.glob(
    os.path.join(PATH_FOLDER_ROOT, FOLDERNAME_TRANS) + "/*/*/*.csv"
)


def cleanText(text):
    # 去除電話Tag標籤
    text = re.sub("<電話>.*?</電話>", r"", text)
    text = re.sub("<狀聲詞>.*?</狀聲詞>", r"", text)
    text = re.sub("<對第三者>.*?</對第三者>", r"", text)
    text = re.sub("<注音符號>.*?</注音符號>", r"", text)
    text = re.sub("<軍事術語數字>.*?</軍事術語數字>", r"", text)
    # 去除Tag標籤
    text = re.sub("<.*?>", r"", text)
    text = re.sub("＜.*?＞", r"", text)
    # 早期文本出現的標記 出現用括弧註記內容
    # 207_2014_601116_20140414174719
    # 207_2017_888010_20170128154007
    # 207_2018_888005_20181001132338
    text = re.sub("\(.*?\)", r"", text)
    # 隱私編碼為#及＃
    text = text.replace("#", "")
    text = text.replace("＃", "")
    # 去除...
    # text = text.replace("…", "")
    # 去除空白
    text = text.replace(" ", "")
    return text


# %%
FILERNAME_RATED = "FirstCall.xlsx"
Rate_pd = pd.read_excel(
    os.path.join(PATH_FOLDER_ROOT, FILERNAME_RATED), sheet_name=0, dtype=str
)
# there is a unique case in the excel file
Rate_pd.replace("99（7）", "99", inplace=True)
# rename column
Rate_pd.rename(columns={"basename": "filename"}, inplace=True)
# %%

tmpFilePath = PATH_FILE_TRANS[1]
transFilename = os.path.basename(tmpFilePath)
transFolder = os.path.basename(os.path.dirname(tmpFilePath))
transYear = os.path.basename(os.path.dirname(os.path.dirname(tmpFilePath)))
transFilename = transFilename.replace(".csv", "").replace(" ", "")
transTime = datetime.strptime(transFilename, "%Y%m%d%H%M%S")
# %%
Trans_pd = pd.DataFrame(columns=["case_no", "year", "folder", "filename", "trans_json"])
for tmpFilePath in tqdm(PATH_FILE_TRANS):
    transFilename = os.path.basename(tmpFilePath)
    transFolder = os.path.basename(os.path.dirname(tmpFilePath))
    transYear = os.path.basename(os.path.dirname(os.path.dirname(tmpFilePath)))
    transFilename = transFilename.replace(".csv", "").replace(" ", "")
    transCase_no = Rate_pd["case_no"][Rate_pd["basename"] == transFilename].values[0]
    # 讀取csv檔
    df = pd.read_csv(tmpFilePath)
    if df.columns.size != 2:
        print("逐字稿不是只有兩個column")
        print(tmpFilePath)
        break
    if df.columns[0].lower() != "id" or df.columns[1].lower() != "content":
        print("逐字稿的colname不對")
        print(tmpFilePath)
        break

    # 只選擇ID和Content兩個column
    df.columns = ["id", "content"]
    df = df[["id", "content"]]
    # 去除有na的row
    df = df.dropna(how="all")
    bool_ID = pd.isnull(df["id"])
    bool_Content = pd.isnull(df["content"])
    if len(df[bool_ID].index) != 0:
        print("Miss ID")
        print(tmpFilePath)
    if len(df[bool_Content].index) != 0:
        print("Miss Content")
        print(tmpFilePath)

    df = df.dropna()
    df["seq"] = df.reset_index().index + 1

    # 處理trans
    transProcess = copy.deepcopy(df)
    transProcess["text_clean"] = transProcess.apply(
        lambda x: cleanText(x["content"]), axis=1
    )
    transProcess["tagger_raw"] = transProcess.apply(
        lambda x: list(
            chain.from_iterable(
                ws(
                    [x["text_clean"]],
                    sentence_segmentation=True,  # To consider delimiters
                    coerce_dictionary=UserDict,
                )
            )
        ),
        axis=1,
    )
    transProcess["pos_raw"] = transProcess.apply(
        lambda x: list(chain.from_iterable(pos([x["tagger_raw"]]))), axis=1
    )

    transText = transProcess.to_json(orient="records", force_ascii=False)
    tmpDf = pd.DataFrame(
        {
            "case_no": [transCase_no],
            "year": [transYear],
            "folder": [transFolder],
            "filename": [transFilename],
            "trans_json": [transText],
        }
    )
    Trans_pd = pd.concat([Trans_pd, tmpDf])


# %%
# 合併自殺評定
TransRater_firstCall_df = Trans_pd.merge(
    Rate_pd,
    left_on=["case_no", "year", "folder", "filename"],
    right_on=["case_no", "year", "folder", "filename"],
    suffixes=("", "_rater"),
)
# %%
TABLE_NAME = "trans"
con = sqlite3.connect("../../data/SQLite/firstcall.db")
cur = con.cursor()
cur.execute(
    '''SELECT count(name)
       FROM sqlite_master
       WHERE type="table" AND name="{table_name}"'''.format(
        table_name=TABLE_NAME
    )
)
cur.execute("""DROP TABLE IF EXISTS "{table_name}";""".format(table_name=TABLE_NAME))
cur.execute(
    """CREATE TABLE trans
        (case_no text,
        year text,
        folder text,
        filename text,
        suci_rated text,
        trans_json json)"""
)

for index, row in tqdm(TransRater_firstCall_df.iterrows()):
    con.execute(
        """insert into "{table_name}" values(?, ?, ?, ?, ?, ?)""".format(
            table_name=TABLE_NAME
        ),
        (
            str(row["case_no"]),
            str(row["year"]),
            str(row["folder"]),
            str(row["filename"]),
            str(row["檢核後自殺評分"]),
            row["trans_json"],
        ),
    )
    con.commit()
con.close()

#
# '''CREATE TABLE trans
#                 (folder text,
#                 filename text,
#                 time text,
#                 tag text,
#                 text json)'''

# %%
# Create a SQL connection to our SQLite database
TABLE_NAME = "esos"
con = sqlite3.connect("../../data/SQLite/firstcall.db")
esos_pd = pd.read_excel(
    os.path.join(PATH_FOLDER_ROOT, "esos_FirstCall.xlsx"), sheet_name=0, dtype=str
)
esos_pd.to_sql(name=TABLE_NAME, con=con)
