#%%
import configparser

import pandas as pd
import statsmodels.api as sm
from sqlalchemy import create_engine

conf = configparser.ConfigParser()
conf.read("config.ini", encoding="utf-8")
engine = create_engine("mysql+pymysql://root:Lyj822919@localhost/thesis")

enterprise = pd.read_sql(conf.get("database", "enterprise"), engine)
defaultent = pd.read_sql(conf.get("database", "defaultent"), engine)
macro = pd.read_sql(conf.get("database", "macro"), engine)
defaultent["default"] = 1
defaultent.rename(columns={"发行人": "企业名称"}, inplace=True)
defaultent.drop_duplicates(subset=["企业名称"], keep="last", inplace=True)
enterprise = enterprise.merge(
    defaultent[["企业名称", "default", "违约日期", "违约前一月主体评级"]], on="企业名称", how="left"
)
enterprise["default"] = enterprise["default"].fillna(0)
enterprise = enterprise[enterprise["城投"] != "是"].dropna(subset=["最新评级"])
enterprise = enterprise[enterprise["最新发债日"].dt.year > 2012]
macro["政府支出/GDP"] = macro["公共财政支出:当月值:季"] / macro["GDP:现价:当季值"]
starts = macro["指标名称"].head(1).values[0]


def map_macro(x, param="政府支出/GDP"):
    if x < starts:
        return macro[param].head(1).values[0]
    tmp_macro = macro[macro["指标名称"] < x].tail(1)[param]
    return tmp_macro.values[0]


enterprise["政府支出/GDP"] = enterprise["最新发债日"].apply(map_macro)
enterprise["SHIBOR"] = enterprise["最新发债日"].apply(map_macro, param="SHIBOR:1年:季")
enterprise["波动率"] = enterprise["最新发债日"].apply(map_macro, param="波动率:50ETF期权:季")
#%%
enterprise["房地产政策"] = enterprise["所属行业一级"].apply(
    lambda x: 1 if x == "房地产" else 0
) * enterprise["最新发债日"].apply(lambda x: 1 if x.year > 2020 else 0)
rating = lambda x: x[0] if x != "AAA" else "AA"
# enterprise["最新评级"] = enterprise.apply(
#     lambda x: rating(x["最新评级"])
#     if isinstance(x["违约前一月主体评级"], float)
#     else rating(x["违约前一月主体评级"]),
#     axis=1,
# )
enterprise.set_index(["企业名称"], inplace=True)
dummies = ["最新评级", "企业性质", "是否上市", "最新发债日"]
df = enterprise[
    [
        "default",  # Y
        # macro
        "最新发债日",
        "政府支出/GDP",
        "SHIBOR",
        "波动率",
        # middle
        "流动性",
        "房地产政策",
        # "所属行业一级",
        # micro
        ## company
        "是否上市",
        "企业性质",
        "持有基金占比",
        "大股东持股比例",
        ## operation
        "主营业务收入(万元)",
        "应付账款",
        "标准券",
        ## finance
        "净资产(万元)",
        "现金短债比",
        "Z",
        ## rating
        "最新评级",
    ]
]

df = df.copy()
enterprise_entity = {
    "国有企业": "国有企业",
    "中央国有企业": "国有企业",
    "地方国有企业": "国有企业",
    "集体企业": "集体企业",
    "民营企业": "民营企业",
    "公众企业": "集体企业",
    "外商独资企业": "外资企业",
    "中外合资企业": "外资企业",
    "外资企业": "外资企业",
    "其他企业": "其他企业",
    None: "其他企业",
}
df["企业性质"] = df["企业性质"].apply(lambda x: enterprise_entity[x])
df["最新发债日"] = df["最新发债日"].apply(lambda x: str(x.year) if x.year > 2013 else "2013")
df = df[df["最新发债日"] != "2013"]
df["最新评级"] = df["最新评级"].apply(rating)
#%%
dummy = [i for i in dummies if i in df.columns]
res = pd.concat(
    [pd.get_dummies(df[dummy], drop_first=True), df.drop(columns=dummy)], axis=1
)
res.dropna(inplace=True)
Y = res["default"]
X = res.drop(columns=["default"])
X = X.copy()
X = sm.add_constant(X)
# model = sm.Logit(Y, X.astype(float))
model = sm.Probit(Y, X.astype(float), M=sm.robust.norms.HuberT())
result = model.fit()
print(result.summary())
# tmp = result.summary().as_html()
# with open("logitres.html", "w") as f:
#     f.write(tmp)
res["enterprise"] = res.index
res.reset_index(drop=True, inplace=True)
res.to_sql("test", engine, if_exists="replace")

# %%
