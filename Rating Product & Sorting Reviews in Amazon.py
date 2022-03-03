###################################################
###################################################
# Rating Product & Sorting Reviews in Amazon
###################################################
###################################################


####################################
# Görev 1: Average Rating’i güncel yorumlara göre
# hesaplayınız ve var olan average rating ile kıyaslayınız.
####################################

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_csv("datasets/amazon_review.csv")
df = df_.copy()  # data copy

df.head()

df.info()

# product_code is : B007WTAJTO
# all_data_mean  : 4.58758 but not useful for using in reviews
# there are 4915 variables and 12 columns we have
#


df["overall"].mean()  # mean of rating data for overall(but not useful it on sorting)
#  4.587589 is all data mean result

df["reviewerID"].nunique()
# 4915 reviews are existed

df["reviewTime"].max()  # last review date: '2014-12-07'

# df.sort_values(by="reviewTime", ascending=False)

df.describe().T
#                     count             mean            std              min              25%              50%              75%              max
# overall        4915.00000          4.58759        0.99685          1.00000          5.00000          5.00000          5.00000          5.00000
# unixReviewTime 4915.00000 1379465001.66836 15818574.32275 1339200000.00000 1365897600.00000 1381276800.00000 1392163200.00000 1406073600.00000
# day_diff       4915.00000        437.36704      209.43987          1.00000        281.00000        431.00000        601.00000       1064.00000
# helpful_yes    4915.00000          1.31109       41.61916          0.00000          0.00000          0.00000          0.00000       1952.00000
# total_vote     4915.00000          1.52146       44.12309          0.00000          0.00000          0.00000          0.00000       2020.00000

from helpers.helpers import check_df

check_df(df)

df["day_cut"] = pd.qcut(df["day_diff"], 4, labels=["0-25", "26-50", "51-75", "76-100"])
# I checked the mean results for each time range as below
df.loc[df["day_cut"] == "0-25", "overall"].mean()  # 4.69579
df.loc[df["day_cut"] == "26-50", "overall"].mean() # 4.63614
df.loc[df["day_cut"] == "51-75", "overall"].mean()  # 4.57166
df.loc[df["day_cut"] == "76-100", "overall"].mean() # 4.44625


#  when days are pass out the average ratings are mostly downs


def time_based_weighted_average(dataframe, days, rating, w1=28, w2=26, w3=24, w4=22):
    if w1 + w2 + w3 + w4 == 100:
        return dataframe.loc[dataframe[days] == "0-25", rating].mean() * w1 / 100 + \
               dataframe.loc[dataframe[days] == "26-50", rating].mean() * w2 / 100 + \
               dataframe.loc[dataframe[days] == "51-75", rating].mean() * w3 / 100 + \
               dataframe.loc[dataframe[days] == "76-100", rating].mean() * w4 / 100
    else:
        print("weighted average separations are wrong total must be 100")


df["time_based_weighted_avarage_score"] = time_based_weighted_average(df, days="day_cut", rating="overall", w1=28, w2=26, w3=24, w4=22)
#  time based average rating 4.59559
#  alldata ratings is 4.587589
#  and when we compare the time based is seems more sensible


####################################
# Görev 2: Ürün için ürün detay sayfasında
# görüntülenecek 20 review’i belirleyiniz.
####################################


df.groupby("reviewerID").agg({"total_vote": "sum"}).sort_values(by="total_vote", ascending=False)
# below results are total votes
# 2031    2020
# 4212    1694
# 3449    1505
# 317      495
# 2909     236
#


df[["helpful", "helpful_yes", "total_vote"]].sort_values(by="helpful_yes", ascending=False).head(40)
# when we compare helpful_yes and helpful we can saw the first index is same
#
#            helpful  helpful_yes
# 2031  [1952, 2020]         1952
# 4212  [1568, 1694]         1568
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]


# w,lson lower bound score calculation
def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


# score added to dataframe
df['wilson_lower_bound'] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)
# when we sort by wilson_lower_bound
df.sort_values('wilson_lower_bound', ascending=False).head(20)
# detailes review result colums selected
df.sort_values('wilson_lower_bound', ascending=False)[["reviewerName", "helpful", "overall", "reviewTime", "reviewText"]].head(20)
