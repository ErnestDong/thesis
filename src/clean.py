#%%
import configparser
import glob

import pandas as pd
from sqlalchemy import create_engine

conf = configparser.ConfigParser()
conf.read("config.ini", encoding="utf-8")
engine = create_engine("mysql+pymysql://root:Lyj822919@localhost/thesis")

#%% 发债企业一览
enterprise = conf.get("database", "enterprise")

engine.execute("DROP TABLE IF EXISTS " + enterprise)
for xlsx in glob.glob("wind/发债企业一览*.xlsx"):
    df = pd.read_excel(xlsx)
    df = df[df["企业名称"] != "数据来源：Wind"].dropna(how="all")
    dates = [i for i in df.columns if df[i].dtype == "datetime64[ns]"]
    for i in dates:
        df[i] = df[i].dt.date
    df.to_sql(enterprise, engine, if_exists="append", index=False)

# %%
defaultent = conf.get("database", "defaultent")

engine.execute("DROP TABLE IF EXISTS " + defaultent)
xlsx = "./wind/债券违约大全(20140101-20220212).xlsx"
df = pd.read_excel(xlsx)
df["违约日期"] = pd.to_datetime(df["违约日期"])

df.to_sql(defaultent, engine, if_exists="replace", index=False)

# %% after wind and create view for altman's Z
enterprise = conf.get("database", "enterprise")

engine.execute("DROP TABLE IF EXISTS " + enterprise)
xlsx = glob.glob("wind/工作簿1.xlsx")[0]
df = pd.read_excel(xlsx)
df.to_sql(enterprise, engine, if_exists="replace", index=False)
#%%
engine.execute(
    """DROP VIEW IF EXISTS altman_z;
    CREATE ALGORITHM = UNDEFINED DEFINER = `root` @`localhost` SQL SECURITY DEFINER VIEW `altman_z` AS
select
  `enterprise`.`企业名称` AS `企业名称`,
  `enterprise`.`最新发行债券代码` AS `最新发行债券代码`,(
    (
      (
        (
          (
            1.2 * (
              (`enterprise`.`流动资产` - `enterprise`.`流动负债`) / `enterprise`.`总资产`
            )
          ) + (
            (
              1.4 * (`enterprise`.`盈余公积` + `enterprise`.`未分配利润`)
            ) / `enterprise`.`总资产`
          )
        ) + ((3.3 * `enterprise`.`EBIT`) / `enterprise`.`总资产`)
      ) + ((0.6 * `enterprise`.`股权价值`) / `enterprise`.`总负债`)
    ) + (`enterprise`.`主营业务收入(万元)` / `enterprise`.`总资产`)
  ) AS `Z_Value`
from
  `enterprise`
ORDER BY `Z_Value` DESC;"""
)
# %%
lowerent = conf.get("database", "lowerratingent")
engine.execute("DROP TABLE IF EXISTS " + lowerent)
xlsx = glob.glob("wind/主体评级调低的企业.xlsx")[0]
df = pd.read_excel(xlsx)
df.to_sql(lowerent, engine, if_exists="replace", index=False)

# %%
macro = conf.get("database", "macro")
engine.execute("DROP TABLE IF EXISTS " + macro)
xlsx = glob.glob("wind/宏观数据.xlsx")[0]
df = pd.read_excel(xlsx, skiprows=1)
df = df.dropna()
df.to_sql(macro, engine, if_exists="replace", index=False)
# %%
