
**该notebook以titanic数据为例，梳理了一些ml建模常用的操作，包括如下一些内容：**   

1. 理解数据并可视化分析(pandas,seaborn)  

2. 数据清洗：NaN的不同填充方法、categorical 变量的处理、数据归一化操作(z-score,min-max,normalize,log,boxcox等)  

3. 模型选择(方差-偏差、过拟合-欠拟合)  

4. 优化，包括：  

    4.1 特征增强
        4.1.1 异常值处理：cap_floor（盖帽）
        4.1.2 特征优化：根据背景知识造特征、聚类算法构建新特征、长尾数据的log变换、boxcox、cdf、pdf变换；构建组合特征，自动构建组合特征(gbdt+lr)；构建交互式特征；  

        4.1.3 特征选择：基于统计方法的方差、相关性、gini、mi、chi2选择等，基于模型的迭代消除法等；  

        4.1.4 特征变换：pca、lda、lle(局部线性嵌入)、ae（自编码）、vae（变分自编码）等  

    4.2 数据增强：半监督学习、过采样、根据数据特征造新数据  
    
    4.3 模型增强：
        4.3.1 超参优化：随机搜索、网格搜索、贝叶斯优化
        4.3.2 集成学习：stacking


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import random
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.feature_selection import SelectFromModel,VarianceThreshold
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
%matplotlib inline
plt.style.use('fivethirtyeight')

import warnings
warnings.filterwarnings('ignore')
```

导入数据：

Survived:0代表死亡，1代表存活

Pclass:乘客所持票类

Name:乘客姓名

Sex:乘客性别

Age:乘客年龄

SibSp:乘客兄弟姐妹/配偶的个数

Parch:乘客父母/孩子的个数

Ticket:票号

Fare:乘客所持票的价格

Cabin:乘客所在船舱

Embarked:乘客登船港口:S、C、Q


```python
train_df=pd.read_csv('./titanic/train.csv')
test_df=pd.read_csv('./titanic/test.csv')
```

### 一.理解数据


```python
#查看shape
'训练集：',train_df.shape,'测试集',test_df.shape
```




    ('训练集：', (891, 12), '测试集', (418, 11))




```python
#查看数据前几行
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



测试集相对于训练集少了Survived，即任务目标，为了方便后续的各种处理，这里将数据合并，并切分为features部分以及labels部分


```python
labels=train_df['Survived']
origin_features_df=pd.concat([train_df.drop(columns=['Survived']),test_df]).reset_index(drop=True)
features_df=copy.deepcopy(origin_features_df)
```


```python
features_df.shape
```




    (1309, 11)




```python
features_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
labels.head(5)
```




    0    0
    1    1
    2    1
    3    1
    4    0
    Name: Survived, dtype: int64




```python
#查看数据条数、每个特征的缺失值个数、特征类型
features_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1309 entries, 0 to 1308
    Data columns (total 11 columns):
    PassengerId    1309 non-null int64
    Pclass         1309 non-null int64
    Name           1309 non-null object
    Sex            1309 non-null object
    Age            1046 non-null float64
    SibSp          1309 non-null int64
    Parch          1309 non-null int64
    Ticket         1309 non-null object
    Fare           1308 non-null float64
    Cabin          295 non-null object
    Embarked       1307 non-null object
    dtypes: float64(2), int64(4), object(5)
    memory usage: 112.6+ KB
    

### 1.1特征分类

表现\功能 | 离散型特征 |  数值型特征  
:- | :-: | -:
int,float | PassengerId,**Pclass** | Age,**Pclass**,SibSp,Parch,Fare 
str | Name,Sex,Ticket,Cabin,Embarked |  

从功能上我们可以简单把数据分为两类，一类是离散型特征、一类是数值型特征：  

（1）数值型特征：对**“比较大小”**有意义的特征，比如“高、中、低”，“优秀、良好、一般、差”等这一类特征可以看做数值型特征

（2）离散型特征：PassengerId虽然表现为int，但对其比较大小并无实际意义


```python
#进一步，查看int,float特征的分布
features_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1309.000000</td>
      <td>1309.000000</td>
      <td>1046.000000</td>
      <td>1309.000000</td>
      <td>1309.000000</td>
      <td>1308.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>655.000000</td>
      <td>2.294882</td>
      <td>29.881138</td>
      <td>0.498854</td>
      <td>0.385027</td>
      <td>33.295479</td>
    </tr>
    <tr>
      <th>std</th>
      <td>378.020061</td>
      <td>0.837836</td>
      <td>14.413493</td>
      <td>1.041658</td>
      <td>0.865560</td>
      <td>51.758668</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.170000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>328.000000</td>
      <td>2.000000</td>
      <td>21.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.895800</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>655.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>982.000000</td>
      <td>3.000000</td>
      <td>39.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.275000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1309.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>9.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>




```python
#对于object类型特征，也可以使用describe查看最频繁的那一项,以及总的有多少项
features_df.describe(include=['O'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Sex</th>
      <th>Ticket</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1309</td>
      <td>1309</td>
      <td>1309</td>
      <td>295</td>
      <td>1307</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>1307</td>
      <td>2</td>
      <td>929</td>
      <td>186</td>
      <td>3</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>CA. 2343</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>2</td>
      <td>843</td>
      <td>11</td>
      <td>6</td>
      <td>914</td>
    </tr>
  </tbody>
</table>
</div>




```python
#通过value_counts查看靠前的项目
features_df['Cabin'].value_counts().head(5)
```




    C23 C25 C27        6
    G6                 5
    B57 B59 B63 B66    5
    F2                 4
    D                  4
    Name: Cabin, dtype: int64



### 1.2可视化


```python
#条形图，存活分布
labels.value_counts().plot(kind='bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1624f80b438>




![png](./md_resource/output_19_1.png)



```python
#条形图：查看Cabin的分布
features_df['Cabin'].value_counts().sort_values(ascending=False).head(20).plot(kind='bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x162518aa358>




![png](./md_resource/output_20_1.png)



```python
#饼图，查看性别分布
features_df['Sex'].value_counts().sort_values(ascending=False).plot(kind='pie')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x16251960240>




![png](./md_resource/output_21_1.png)



```python
#直方图，查看年龄分布
features_df['Age'].plot(kind='hist')#注意：有别于条形图，这里会将连续值“分箱”
```




    <matplotlib.axes._subplots.AxesSubplot at 0x162519a4940>




![png](./md_resource/output_22_1.png)



```python
#箱线图，查看年龄分布
features_df['Age'].plot(kind='box')#可见有绝大部分是20-40岁的年轻人
```




    <matplotlib.axes._subplots.AxesSubplot at 0x16251a187f0>




![png](./md_resource/output_23_1.png)



```python
#条形图：查看票价与票类的关系
features_df.groupby('Pclass')['Fare'].mean().plot(kind='bar')#Pclass 1,2,3分别表示头等舱、一等舱、二等舱
```




    <matplotlib.axes._subplots.AxesSubplot at 0x16251a852e8>




![png](./md_resource/output_24_1.png)



```python
#将年龄分成1-10的10个阶段，并统计对应票价
def age_bin(x):
    try:
        return int(x/10)
    except:
        None
features_df['Age_bin']=features_df['Age'].apply(age_bin)
features_df.groupby('Age_bin')['Fare'].mean().plot(kind='bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x16251aef0f0>




![png](./md_resource/output_25_1.png)



```python
#平滑处理
features_df.groupby('Age_bin')['Fare'].mean().rolling(3).mean().plot(kind='line')#注意:前rolling_num-1项会为None
```




    <matplotlib.axes._subplots.AxesSubplot at 0x16251b54b38>




![png](./md_resource/output_26_1.png)



```python
#查看各登船港口与票类的关系
features_df.groupby('Embarked')['Pclass'].mean().plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x16251bc36a0>




![png](./md_resource/output_27_1.png)



```python
#探索不同性别，不同年龄段的存活率
show_df=features_df[:891]
show_df['Survived']=labels
show_df=show_df.groupby(['Sex','Age_bin'])['Survived'].mean().reset_index().pivot(index='Sex',columns='Age_bin',values='Survived').reset_index()
show_df=show_df.where(show_df.notnull(),0)
show_df.set_index('Sex',inplace=True,drop=True)
sns.heatmap(show_df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x16251c70fd0>




![png](./md_resource/output_28_1.png)


**注意float.nan无法通过fillna填充**  

可以发现婴儿和女性的存活率更高


```python
#热图：探索不同特征之间的相关系数
show_df=features_df[:891]
show_df['Survived']=labels
sns.heatmap(show_df.corr())
```




    <matplotlib.axes._subplots.AxesSubplot at 0x16251ea8b38>




![png](./md_resource/output_30_1.png)


可以发现是否存活与Fare/票价的正相关性最强，与Pclass负相关性最强

### 二.清洗特征
大概知道了各特征的类型以及缺失值情况后，我们就可以对数据进行清洗，将其转换成int/float数据类型

#### 2.1 None值填充
（1）删除...dropna...(不建议)    
（2）暴力填充fillna一个确定的值  
（3）均值、中位数、众数项填充  
（4）依赖性填充  
    4.1）基于时间的插值；  
    4.2）建模预测...


```python
#首先查看缺失项
features_df.isnull().sum()
```




    PassengerId       0
    Pclass            0
    Name              0
    Sex               0
    Age             263
    SibSp             0
    Parch             0
    Ticket            0
    Fare              1
    Cabin          1014
    Embarked          2
    Age_bin         263
    dtype: int64




```python
del features_df['Age_bin']
```


```python
#为Cabin直接填充一个定值
features_df['Cabin'].fillna('missing',inplace=True)
```


```python
#为Embarked填充众数项
features_df['Embarked'].fillna(features_df['Embarked'].mode()[0],inplace=True)
```


```python
#Age与Pclass具有比较强的相关性，利用Age所属Pclass组的均值对其缺失值进行填充
features_df.groupby('Pclass')['Age'].mean().reset_index()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>39.159930</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>29.506705</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>24.816367</td>
    </tr>
  </tbody>
</table>
</div>




```python
def fill_age(pclass,age):
    if np.isnan(age):#注意：如果当前列为float/int类型，当前列中的None会被强制转为float的nan类型
        if pclass==1:
            return 39.159930
        elif pclass==2:
            return 29.506705
        else:
            return 24.816367
    else:
        return age
features_df['Age']=features_df.apply(lambda row:fill_age(row['Pclass'],row['Age']),axis=1)
```


```python
features_df['Fare'].fillna(features_df['Fare'].mean(),inplace=True)
```


```python
#检查一下
features_df.isnull().sum()
```




    PassengerId    0
    Pclass         0
    Name           0
    Sex            0
    Age            0
    SibSp          0
    Parch          0
    Ticket         0
    Fare           0
    Cabin          0
    Embarked       0
    dtype: int64



也可以多种填充策略组合，比如先依赖于相关性高的进行填充，然后利用整体的均值/中位数等填充，然后fillna一个确定的值等，另外sklearn.preprocessing.Imputer也可方便进行均值、中位数、众数填充  

#### 注意：
**
（1）这里没有那一种填充方法是“绝对”的好，要与后续建模中具体使用的模型结合起来，均值类型的填充也许对决策树一类的算法比较友好，fillna(0)也许对lr/神经网络一类的算法比较友好；  
（2）切记在填充的时候不要依赖到y标签，假如Age与Survived的相关性比较强，如果Age依赖于Survived的分组均值进行填充...想想会发生什么....真正在预测的时候我们不知道Survived真实取值，所以压根没法填充...  
（3）另外这里使用整体数据的均值/中位数进行填充涉嫌“作弊”，因为利用到了测试集部分，更正确的做法是利用训练集的信息去对训练集合测试集进行填充，但一般在样本量很大的情况下，两者均值/中位数....相差不大，所以...
**


```python
features_df['Age'].mean()
```




    29.348219164247467




```python
features_df[:891]['Age'].mean()
```




    29.269997269360225




```python
#目前的情况如下
features_df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>missing</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>missing</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>missing</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



#### 2.2 类别特征数值化
我们还有5列类别特征，可以将他们分为两类：  
（1）低数量类别特征/low-cardinality categorical attributes：Sex,Embarked  
（2）高数量类别特征/high-cardinality categorical attributes:Name,Ticket,Cabin

我们一个一个看....  

Sex:明显的离散型特征，可以采用ont-hot编码，由于只有两个可取变量，也可以直接编码0/1  
Embarked:明显的离散型特征，可采用one-hot编码  
Name:可以再深入挖掘的一个特征，比如通过名字判断是否结婚  
Ticket:看不出明显的规律，删掉  
Cabin:这里先简单的采用TargetEncoder,一种Smothing的方式，计算y的占比；后面可以扩展特征，比如同一船舱：是否包含中年男性/是否包含小孩....，如果当前乘客为小孩，而他同行的船舱中有中年男性，那存活的可能性更高? 这里，，，先删掉    

这里推荐category_encoders,有多种对离散特征encoding的方式  
pip install category_encoders  
更多：http://contrib.scikit-learn.org/categorical-encoding/  


```python
import category_encoders as ce
del features_df['Name']
del features_df['Ticket']
onehot_encoder = ce.OneHotEncoder(cols=['Embarked']).fit(features_df)
features_df=onehot_encoder.transform(features_df)

ordinay_encoder = ce.OrdinalEncoder(cols=['Sex']).fit(features_df)
features_df=ordinay_encoder.transform(features_df)

#这里我就偷懒了....
target_encoder = ce.TargetEncoder(cols=['Cabin']).fit(features_df[:891],labels)
features_df=target_encoder.transform(features_df)
```


```python
features_df['Sex']=features_df['Sex']-1
```


```python
features_df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked_1</th>
      <th>Embarked_2</th>
      <th>Embarked_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>0.299854</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>0.383838</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>0.299854</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>0.468759</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>3</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>0.299854</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#这里PassengerId直接删掉
del features_df['PassengerId']
```

#### 2.3数据标准化
数据是否需要标准化取决于后面的训练模型，一般来说，数据归一化有如下的一些好处：  
（1）保持量纲的统一；  
（2）梯度下降更稳定；

它对常见算法的影响：  
（1）knn,kmeans：欧氏距离...  
（2）lr,svm,nn：梯度下降...  
（3）pca：偏向于值较大的列  

标准化的方法一般有以下几种：  
（1）z-score：$z=(x-u)/\sigma$  
（2）min-max：$m=(x-x_{min})/(x_{max}-x_{min})$  
（3）行归一化：$\sqrt{(x_1^2+x_2^2+\cdots+x_n^2)}=1,这里x_1,x_2,...,x_n表示每行特征$；  
其他的标准化方法：  
（1）log/boxcox归一化：对长尾分布数据比较有用，可以拉伸头部，压缩尾部

更多：https://www.jianshu.com/p/fa73a07cd750


```python
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer
#z-score归一化
titanic_z_score_df=pd.DataFrame(StandardScaler().fit_transform(features_df),columns=features_df.columns)
#min-max归一化
titanic_min_max_df=pd.DataFrame(MinMaxScaler().fit_transform(features_df),columns=features_df.columns)
#行归一化
titanic_normalize_df=pd.DataFrame(Normalizer().fit_transform(features_df),columns=features_df.columns)
```


```python
titanic_z_score_df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked_1</th>
      <th>Embarked_2</th>
      <th>Embarked_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.841916</td>
      <td>-0.743497</td>
      <td>-0.559957</td>
      <td>0.481288</td>
      <td>-0.445</td>
      <td>-0.503595</td>
      <td>-0.355244</td>
      <td>0.655011</td>
      <td>-0.50977</td>
      <td>-0.32204</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.546098</td>
      <td>1.344995</td>
      <td>0.659292</td>
      <td>0.481288</td>
      <td>-0.445</td>
      <td>0.734503</td>
      <td>0.314031</td>
      <td>-1.526692</td>
      <td>1.96167</td>
      <td>-0.32204</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.841916</td>
      <td>1.344995</td>
      <td>-0.255145</td>
      <td>-0.479087</td>
      <td>-0.445</td>
      <td>-0.490544</td>
      <td>-0.355244</td>
      <td>0.655011</td>
      <td>-0.50977</td>
      <td>-0.32204</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.546098</td>
      <td>1.344995</td>
      <td>0.430683</td>
      <td>0.481288</td>
      <td>-0.445</td>
      <td>0.382925</td>
      <td>0.990773</td>
      <td>0.655011</td>
      <td>-0.50977</td>
      <td>-0.32204</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.841916</td>
      <td>-0.743497</td>
      <td>0.430683</td>
      <td>-0.479087</td>
      <td>-0.445</td>
      <td>-0.488127</td>
      <td>-0.355244</td>
      <td>0.655011</td>
      <td>-0.50977</td>
      <td>-0.32204</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic_min_max_df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked_1</th>
      <th>Embarked_2</th>
      <th>Embarked_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.273456</td>
      <td>0.125</td>
      <td>0.0</td>
      <td>0.014151</td>
      <td>0.226644</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.473882</td>
      <td>0.125</td>
      <td>0.0</td>
      <td>0.139136</td>
      <td>0.323450</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.323563</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.015469</td>
      <td>0.226644</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.436302</td>
      <td>0.125</td>
      <td>0.0</td>
      <td>0.103644</td>
      <td>0.421336</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.436302</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.015713</td>
      <td>0.226644</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic_normalize_df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked_1</th>
      <th>Embarked_2</th>
      <th>Embarked_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.128194</td>
      <td>0.000000</td>
      <td>0.940092</td>
      <td>0.042731</td>
      <td>0.0</td>
      <td>0.309803</td>
      <td>0.012813</td>
      <td>0.042731</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.012375</td>
      <td>0.012375</td>
      <td>0.470268</td>
      <td>0.012375</td>
      <td>0.0</td>
      <td>0.882164</td>
      <td>0.004750</td>
      <td>0.000000</td>
      <td>0.012375</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.109552</td>
      <td>0.036517</td>
      <td>0.949452</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.289400</td>
      <td>0.010950</td>
      <td>0.036517</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.015716</td>
      <td>0.015716</td>
      <td>0.550051</td>
      <td>0.015716</td>
      <td>0.0</td>
      <td>0.834507</td>
      <td>0.007367</td>
      <td>0.015716</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.083208</td>
      <td>0.000000</td>
      <td>0.970766</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.223276</td>
      <td>0.008317</td>
      <td>0.027736</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#注意：z-score,min-max这两种变换都是线性变换，不会改变分布形状
titanic_min_max_df['Age'].plot(kind='hist')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x162530350b8>




![png](./md_resource/output_56_1.png)



```python
titanic_z_score_df['Age'].plot(kind='hist')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x16253099b38>




![png](./md_resource/output_57_1.png)



```python
#行归一化会改变
titanic_normalize_df['Age'].plot(kind='hist')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x162530f6390>




![png](./md_resource/output_58_1.png)



```python
#pdf:标准正态分布的概率密度函数
from scipy.stats import norm
age_mean=features_df['Age'].mean()
age_std=features_df['Age'].std()
features_df['Age'].apply(lambda x:norm.pdf((x-age_mean)/age_std)).plot(kind='hist')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1625317d4e0>




![png](./md_resource/output_59_1.png)



```python
#cdf:分布函数
features_df['Age'].apply(lambda x:norm.cdf((x-age_mean)/age_std)).plot(kind='hist')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x162531f16a0>




![png](./md_resource/output_60_1.png)



```python
#log
features_df['Age'].apply(lambda x:np.log1p(x)).plot(kind='hist')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x162531e1588>




![png](./md_resource/output_61_1.png)



```python
#boxcox
from scipy.stats import boxcox
plt.hist(boxcox(features_df['Age'])[0])
```




    (array([ 51.,  35.,  48., 274., 478., 210., 118.,  55.,  32.,   8.]),
     array([-0.97171828,  2.70253966,  6.37679759, 10.05105553, 13.72531346,
            17.39957139, 21.07382933, 24.74808726, 28.42234519, 32.09660313,
            35.77086106]),
     <a list of 10 Patch objects>)




![png](./md_resource/output_62_1.png)



```python
#最佳lambda
boxcox(features_df['Age'])[1]
```




    0.7627222156380012



### 三.选择基准模型
（1）**目标**：本数据集是预测乘客是否存活，所以可以看做是分类任务；  
（2）**量化目标**：选择合适的评估指标，这里我们可以选择f1；  
（3）从分类模型中选择一个较优的模型作为基准模型，这是一个比较繁琐的工作；  


```python
data_x=StandardScaler().fit_transform(features_df[:891])
#切分训练集测试集
X_train,X_test, y_train, y_test =train_test_split(data_x,labels,test_size=0.2, random_state=42)
#训练模型
classifier=LogisticRegression()
classifier.fit(X_train,y_train)
#预测数据
y_predict=classifier.predict(X_test)
#查看检测指标
f1_score=metrics.f1_score(y_test,y_predict)
f1_score
```




    0.7862068965517242




```python
#为了结果更加客观，可以做k-fold交叉验证，但会更耗时

classifier=LogisticRegression()
scores = cross_val_score(classifier, data_x, labels, scoring='f1', cv = 5)#注意：f1只是看正样本的f1,如果要看整体的用f1_macro,但这一般会使得f1偏高
#查看均值与标准差,均值反映模型的预测能力，标准差可以反映模型的稳定性
np.mean(scores),np.std(scores)
```




    (0.759262777574798, 0.016196297194678546)




```python
#我们再看看另一种分类器
classifier=GradientBoostingClassifier()
scores = cross_val_score(classifier, features_df[:891], labels, scoring='f1', cv = 5)
np.mean(scores),np.std(scores)
```




    (0.7724798337990278, 0.052794300926641495)



#### 定位模型能力：方差与偏差
可以参考下面图为我们的模型做定位：  
![avatar](./source/方差与偏差.png)
来源:https://blog.csdn.net/hertzcat/article/details/80035330

#### 检查过/欠拟合情况
过/欠拟合可以通过模型在训练集/验证集/测试集上的表现来评估，  
（1）训练集/验证集/测试集效果都比较差，可以看作是欠拟合(除非训练数据真的是太差了)，这时可以增加模型的复杂度试一试；  
（2）训练集的表现好，而验证集/测试集的表现差，一般就是过拟合（这也是经常会遇到的问题），可以下面的一些方式常识：  
    2.1）降低模型复杂度：1.换更简单的模型，2.正则化技术  
    2.2）增强训练数据  


```python
#查看lr的训练集，测试集的情况
classifier=LogisticRegression()
classifier.fit(X_train,y_train)
y_test_predict=classifier.predict(X_test)
test_f1_score=metrics.f1_score(y_test,y_test_predict)

y_train_predict=classifier.predict(X_train)
train_f1_score=metrics.f1_score(y_train,y_train_predict)
print('train:',train_f1_score,'\t test:',test_f1_score)
```

    train: 0.7630057803468208 	 test: 0.7862068965517242
    


```python
#查看gbdt的训练集，测试集的情况
classifier=GradientBoostingClassifier()
classifier.fit(X_train,y_train)
y_test_predict=classifier.predict(X_test)
test_f1_score=metrics.f1_score(y_test,y_test_predict)

y_train_predict=classifier.predict(X_train)
train_f1_score=metrics.f1_score(y_train,y_train_predict)
print('train:',train_f1_score,'\t test:',test_f1_score)
```

    train: 0.8641975308641976 	 test: 0.7826086956521738
    

可以发现gbdt有点过拟合了，lr很稳定，然后我们可以看出模型过拟合/欠拟合与模型方差/偏差的一些关系：  
（1）欠拟合模型往往偏差大（这里对应f1指标较小）  
（2）过拟合模型往往方差较大（这里对应f1的标准差较大）  
接下来可以通过降低gbdt中cart树数量的方式来降低模型复杂度：  


```python
classifier=GradientBoostingClassifier(n_estimators=80)#默认是100
scores = cross_val_score(classifier, data_x, labels, scoring='f1', cv = 5)
np.mean(scores),np.std(scores)
```




    (0.7770062297307175, 0.04740066888740136)




```python
classifier=GradientBoostingClassifier(n_estimators=80)
classifier.fit(X_train,y_train)
y_test_predict=classifier.predict(X_test)
test_f1_score=metrics.f1_score(y_test,y_test_predict)

y_train_predict=classifier.predict(X_train)
train_f1_score=metrics.f1_score(y_train,y_train_predict)
print('train:',train_f1_score,'\t test:',test_f1_score)
```

    train: 0.8588957055214724 	 test: 0.7769784172661871
    

### 四.优化
对于建模的优化，可以自然的从三方面来考虑：  
（1）特征优化：异常处理、扩展特征，特征选择，特征转换...  
（2）数据增强：过采样、根据数据特性造新数据、半监督学习...  
（3）模型优化：超参优化、模型集成...

#### 4.1.1 异常处理：盖帽
盖帽操作可以在一定程度上去掉一些异常点，从而提高模型的泛化能力


```python
#将数值低于5%分位数的设置为5%分位数值，高于95%分位数的设置为95%分位数值
low_thresh=5
high_thresh=95
cap_dict={}
for column in features_df.columns:
    low_value=np.percentile(features_df[:891][column],low_thresh)
    high_value=np.percentile(features_df[:891][column],high_thresh)
    if low_value==high_value:#这里相当于不进行盖帽
        low_value=np.min(features_df[:891][column])
        high_value=np.max(features_df[:891][column])
    cap_dict[column]=[low_value,high_value]
#更新
def cap_update(column,x):
    if x>cap_dict[column][1]:
        return cap_dict[column][1]
    elif x<cap_dict[column][0]:
        return cap_dict[column][0]
    else:
        return x
for column in features_df.columns:
    features_df[column]=features_df[column].apply(lambda x:cap_update(column,x))
```


```python
#检查效果
classifier=GradientBoostingClassifier()
scores = cross_val_score(classifier, features_df[:891], labels, scoring='f1', cv = 5)
np.mean(scores),np.std(scores)
```




    (0.7819874859401702, 0.036249676894704055)



可以发现盖帽处理直接将结果提升了差不多1%，而且模型的方差降低了不少，可以起到不错的泛化作用

#### 4.1.2 特征扩展：推理
凭借自己对数据的理解构建有意义的特征，比如：  
通过Cabin关联乘客的同行者的信息，根据前面的性别-存活率的热图，我们扩展这样的特征：是否有其他小孩(Age<=10), 是否有其他女性，是否有其他老人(age>=70),是否有其他青年男性（20<=Age<=50），以及当前cabin的人数


```python
extend_df=origin_features_df[['PassengerId','Age','Sex','Name','Cabin']]
extend_df=extend_df[~extend_df['Cabin'].isnull()]
```


```python
extend_df2=extend_df[['Age','Name','Sex','Cabin']]
extend_df2.columns=['Age2','Name2','Sex2','Cabin']
```


```python
merge_df=pd.merge(extend_df,extend_df2,on='Cabin',how='left')
```


```python
def check_has_other_child(name1,name2,age2):
    if name1==name2:
        return 0
    else:
        if age2<=10:
            return 1
        else:
            return 0
merge_df['Has_other_child']=merge_df.apply(lambda row:check_has_other_child(row['Name'],row['Name2'],row['Age2']),axis=1)
```


```python
def check_has_other_female(name1,name2,sex2):
    if name1==name2:
        return 0
    else:
        if sex2=='female':
            return 1
        else:
            return 0
merge_df['Has_other_female']=merge_df.apply(lambda row:check_has_other_female(row['Name'],row['Name2'],row['Sex2']),axis=1)
```


```python
def check_has_other_old(name1,name2,age2):
    if name1==name2:
        return 0
    else:
        if age2>=70:
            return 1
        else:
            return 0
merge_df['Has_other_old']=merge_df.apply(lambda row:check_has_other_old(row['Name'],row['Name2'],row['Age2']),axis=1)
```


```python
def check_has_young_male(name1,name2,sex2,age2):
    if name1==name2:
        return 0
    else:
        if sex2=='male' and age2>=20 and age2<=50:
            return 1
        else:
            return 0
merge_df['Has_other_young_male']=merge_df.apply(lambda row:check_has_young_male(row['Name'],row['Name2'],row['Sex2'],row['Age2']),axis=1)
```


```python
merge_df=merge_df[['PassengerId','Has_other_child','Has_other_female','Has_other_old','Has_other_young_male']]
```


```python
#去重
gp_df=merge_df.groupby(by=['PassengerId']).agg({'Has_other_child':'max','Has_other_female':'max','Has_other_old':'max','Has_other_young_male':'max'}).reset_index()
```


```python
gp_df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Has_other_child</th>
      <th>Has_other_female</th>
      <th>Has_other_old</th>
      <th>Has_other_young_male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
extend_df=pd.merge(origin_features_df[['PassengerId']],gp_df,on='PassengerId',how='left')
extend_df.fillna(0,inplace=True)
```


```python
extend_df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Has_other_child</th>
      <th>Has_other_female</th>
      <th>Has_other_old</th>
      <th>Has_other_young_male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
del extend_df['PassengerId']
features_df=pd.concat([features_df,extend_df],axis=1)
```


```python
features_df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked_1</th>
      <th>Embarked_2</th>
      <th>Embarked_3</th>
      <th>Has_other_child</th>
      <th>Has_other_female</th>
      <th>Has_other_old</th>
      <th>Has_other_young_male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>0</td>
      <td>22.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7.2500</td>
      <td>0.299854</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>71.2833</td>
      <td>0.383838</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.9250</td>
      <td>0.299854</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>53.1000</td>
      <td>0.468759</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>0</td>
      <td>35.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0500</td>
      <td>0.299854</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#检查效果
classifier=GradientBoostingClassifier()
scores = cross_val_score(classifier, features_df[:891], labels, scoring='f1', cv = 5)
np.mean(scores),np.std(scores)
```




    (0.7793257119630568, 0.04107771702611043)



#### 4.1.2 特征扩展：推理
之前我们删掉了Name，其实Name中的姓可以反应一些特征，比如Mrs可以放映出该乘客已经结婚，Miss表示未婚小姐姐，我们将其提取出来，另外SibSp表示乘客兄弟姐妹/配偶的个数，而Parch表示乘客父母/孩子的个数，可以简单相加表示他们的家庭成员多少，越多的存活率可能越高...  

更多:https://www.kaggle.com/gunesevitan/advanced-feature-engineering-tutorial-with-titanic


```python
features_df['family_size']=features_df['SibSp']+features_df['Parch']+1
```


```python
#统计存活率分布
show_df=features_df[:891]
show_df['Survived']=labels
show_df.groupby('family_size')['Survived'].mean().plot(kind='bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x16253375160>




![png](./md_resource/output_98_1.png)



```python
show_df.groupby('family_size')['Survived'].count().plot(kind='bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x162533e6390>




![png](./md_resource/output_99_1.png)



```python
#提取姓名中的title
origin_features_df['Name'].apply(lambda name:name.split(',')[1].split('.')[0]).value_counts()
```




     Mr              757
     Miss            260
     Mrs             197
     Master           61
     Rev               8
     Dr                8
     Col               4
     Major             2
     Ms                2
     Mlle              2
     the Countess      1
     Dona              1
     Don               1
     Jonkheer          1
     Sir               1
     Capt              1
     Mme               1
     Lady              1
    Name: Name, dtype: int64




```python
titles=['Mr','Miss','Mrs','Master']
def extract_title(name):
    for title in titles:
        if title in name:
            return title
    return 'Other'
features_df['name_title']=origin_features_df['Name'].apply(extract_title)
```


```python
features_df=pd.get_dummies(features_df,columns=['name_title'])
```


```python
features_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked_1</th>
      <th>Embarked_2</th>
      <th>Embarked_3</th>
      <th>Has_other_child</th>
      <th>Has_other_female</th>
      <th>Has_other_old</th>
      <th>Has_other_young_male</th>
      <th>family_size</th>
      <th>name_title_Master</th>
      <th>name_title_Miss</th>
      <th>name_title_Mr</th>
      <th>name_title_Other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>0</td>
      <td>22.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7.2500</td>
      <td>0.299854</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>71.2833</td>
      <td>0.383838</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.9250</td>
      <td>0.299854</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>53.1000</td>
      <td>0.468759</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>0</td>
      <td>35.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0500</td>
      <td>0.299854</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#检查效果
classifier=GradientBoostingClassifier()
scores = cross_val_score(classifier, features_df[:891], labels, scoring='f1', cv = 5)
np.mean(scores),np.std(scores)
```




    (0.7850367518401871, 0.04384734351577558)



#### 4.1.2 扩展特征：添加聚类标签
聚类标签可以看作是对目前样本的做的特征映射，将高维空间相似的特征映射到相同的标签；  
这里演示用kmean生成聚类标签，利用calinski_harabaz选择较优的k，  
更多：https://blog.csdn.net/u010159842/article/details/78624135


```python
cluster_data_np=StandardScaler().fit_transform(features_df)
```


```python
K=range(2,20)
calinski_harabaz_scores=[]
for k in K:
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(cluster_data_np)
    calinski_harabaz_scores.append(metrics.calinski_harabaz_score(cluster_data_np, kmeans.predict(cluster_data_np)))
plt.plot(K,calinski_harabaz_scores,'bx-')
plt.xlabel('k')
plt.ylabel(u'distortion degree')
```




    Text(0,0.5,'distortion degree')




![png](./md_resource/output_107_1.png)



```python
kmeans=KMeans(n_clusters=11)
kmeans.fit(cluster_data_np)
ext_cluster_fea_df=copy.deepcopy(features_df)
ext_cluster_fea_df['cluster_factor']=kmeans.predict(cluster_data_np)
```


```python
ext_cluster_fea_dummy_df = pd.get_dummies(ext_cluster_fea_df,columns=['cluster_factor'])
```


```python
#检查效果
classifier=GradientBoostingClassifier()
scores = cross_val_score(classifier, ext_cluster_fea_dummy_df[:891], labels, scoring='f1', cv = 5)
np.mean(scores),np.std(scores)
```




    (0.7805970033404762, 0.045626939881205225)



不过貌似这样选择的k做的聚类因子未必是最好的....可以多尝试几种...

#### 4.1.2  扩展特征：数值特征
对数值特征的扩展，可以考虑：  
（1）连续值分箱：某些特征分箱可能可以体现不一样的意义...  
（2）log变换等...改变原始数据的分布特性...  
（3）无脑构造多项式/交互特征：$[a,b]->[1,a,b,a^2,b^2,ab]$  
接下来试一试...


```python
ext_fea_df=copy.deepcopy(ext_cluster_fea_dummy_df)
ext_fea_df['Age_bins']=pd.cut(ext_fea_df['Age'],bins=10,labels=False)#对age分箱
ext_fea_df['Fare_bins']=pd.cut(ext_fea_df['Fare'],bins=10,labels=False)#对Fare分箱
ext_fea_df['Age_log']=ext_fea_df['Age'].apply(lambda x:np.log1p(x))#log变换
ext_fea_df['Age_log_cdf']=ext_fea_df['Age_log'].apply(lambda x:norm.cdf((x-ext_fea_df['Age_log'].mean())/ext_fea_df['Age_log'].std()))#cdf变换
```


```python
#lr
ext_fea_np=StandardScaler().fit_transform(ext_fea_df[:891])
classifier=LogisticRegression()
scores = cross_val_score(classifier, ext_fea_np, labels, scoring='f1', cv = 5)
np.mean(scores),np.std(scores)
```




    (0.7686322685829483, 0.03609935046387324)




```python
#gbdt
classifier=GradientBoostingClassifier()
scores = cross_val_score(classifier, ext_fea_np, labels, scoring='f1', cv = 5)
np.mean(scores),np.std(scores)
```




    (0.7823895879512534, 0.04632164288132967)




```python
#构造交互特征：一般来说选择0/1类型的特征来构造更make sense
poly=PolynomialFeatures(degree=2,include_bias=False,interaction_only=False)#无脑多项式转换
poly_fea_np=poly.fit_transform(ext_fea_df)#这里是numpy类型
poly_fea_df=pd.DataFrame(poly_fea_np,columns=poly.get_feature_names())
```

**注意：构造多项式特征慎用，它以$O(n^2)$增涨特征量，如果原始有1000个特征，变换后会有100W个特征...**


```python
poly_fea_df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x0</th>
      <th>x1</th>
      <th>x2</th>
      <th>x3</th>
      <th>x4</th>
      <th>x5</th>
      <th>x6</th>
      <th>x7</th>
      <th>x8</th>
      <th>x9</th>
      <th>...</th>
      <th>x30^2</th>
      <th>x30 x31</th>
      <th>x30 x32</th>
      <th>x30 x33</th>
      <th>x31^2</th>
      <th>x31 x32</th>
      <th>x31 x33</th>
      <th>x32^2</th>
      <th>x32 x33</th>
      <th>x33^2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.0</td>
      <td>0.0</td>
      <td>22.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7.2500</td>
      <td>0.299854</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>9.406483</td>
      <td>1.055688</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>9.831324</td>
      <td>1.103368</td>
      <td>0.123831</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>38.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>71.2833</td>
      <td>0.383838</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>36.0</td>
      <td>36.0</td>
      <td>21.981370</td>
      <td>4.646550</td>
      <td>36.0</td>
      <td>21.981370</td>
      <td>4.646550</td>
      <td>13.421684</td>
      <td>2.837154</td>
      <td>0.599734</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.9250</td>
      <td>0.299854</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>13.183347</td>
      <td>1.942617</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>10.862541</td>
      <td>1.600637</td>
      <td>0.235860</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>35.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>53.1000</td>
      <td>0.468759</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>36.0</td>
      <td>24.0</td>
      <td>21.501114</td>
      <td>4.317604</td>
      <td>16.0</td>
      <td>14.334076</td>
      <td>2.878403</td>
      <td>12.841608</td>
      <td>2.578703</td>
      <td>0.517825</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.0</td>
      <td>0.0</td>
      <td>35.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0500</td>
      <td>0.299854</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>36.0</td>
      <td>0.0</td>
      <td>21.501114</td>
      <td>4.317604</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>12.841608</td>
      <td>2.578703</td>
      <td>0.517825</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 629 columns</p>
</div>




```python
poly_fea_df.shape
```




    (1309, 629)




```python
#看看在lr上的表现
classifier=LogisticRegression()
scores = cross_val_score(classifier, StandardScaler().fit_transform(poly_fea_df[:891]), labels, scoring='f1', cv = 5)
np.mean(scores),np.std(scores)
```




    (0.7529680759275237, 0.047662184324391406)




```python
#gbdt
classifier=GradientBoostingClassifier()
scores = cross_val_score(classifier, poly_fea_df[:891], labels, scoring='f1', cv = 5)
np.mean(scores),np.std(scores)
```




    (0.7877111481159954, 0.043269962360609864)



###### 特征数增加是否会影响模型稳定性？
这里发现特征量的快速增加（10->629），lr的std增加了很多，gbdt有所减少，这是因为特征数量的增加被动的增加了lr模型的复杂度($\sigma(w^Tx+b)$,模型的复杂度与$x$的维度正比)，而gbdt在生成树的时候对于用处不大的特征，选择的少或者压根不会选。

#### 4.1.2 特征扩展：离散特征
离散特征的扩展可以考虑特征组合，比如：  
（1）从make sense的情况下组合特征；  
（2）自动特征组合...

#### make sense的特征
构造乘客性别和票类型的组合特征：

Pclass |  Sex  
-|-
1 | male |
2 | female |  
转换为：  

Pclass_1_female |  Pclass_2_female | Pclass_3_female |  Pclass_1_male |Pclass_2_male |  Pclass_3_male  
-|-|-|-|-|-
0|0|0|1|0|0
0|1|0|0|0|0 



```python
def combine_pclass_sex(pclass,sex):
    if sex=='male':
        return pclass-1
    else:
        return pclass+2
ext_cat_fea_df=copy.deepcopy(poly_fea_df)
ext_cat_fea_df['Pclass_Sex']=origin_features_df.apply(lambda row:combine_pclass_sex(row['Pclass'],row['Sex']),axis=1)
ext_cat_fea_dummy_df = pd.get_dummies(ext_cat_fea_df,columns=['Pclass_Sex'])
```


```python
ext_cat_fea_dummy_df.columns
```




    Index(['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9',
           ...
           'x31 x33', 'x32^2', 'x32 x33', 'x33^2', 'Pclass_Sex_0', 'Pclass_Sex_1',
           'Pclass_Sex_2', 'Pclass_Sex_3', 'Pclass_Sex_4', 'Pclass_Sex_5'],
          dtype='object', length=635)




```python
#gbdt
classifier=GradientBoostingClassifier()
scores = cross_val_score(classifier, ext_cat_fea_dummy_df[:891], labels, scoring='f1', cv = 5)
np.mean(scores),np.std(scores)
```




    (0.7875957671794318, 0.03422694316681235)



#### 4.1.2 特征扩展：自动构建组合特征
比较流行的一种方式是gbdt+lr,即利用gbdt探索不错的特征空间，然后用lr对这些特征空间张成one-hot特征进行拟合；  
![avatar](./source/gbdt_lr.png)
参考：https://www.cnblogs.com/wkang/p/9657032.html


```python
from sklearn.preprocessing import OneHotEncoder
n_trees=100
tree_depth=2#树的深度不必太深
kfold= KFold(n_splits=5,shuffle=True)
scores=[]
for train_index,test_index in kfold.split(ext_cat_fea_dummy_df[:891],labels):
    X_train=ext_cat_fea_dummy_df.loc[train_index]
    y_train=labels[train_index]
    X_test=ext_cat_fea_dummy_df.loc[test_index]
    y_test=labels[test_index]
    
    gbm1 = GradientBoostingClassifier(n_estimators=n_trees,max_depth=tree_depth)
    gbm1.fit(X_train, y_train)
    train_new_feature = gbm1.apply(X_train)
    train_new_feature = train_new_feature.reshape(-1, n_trees)

    enc = OneHotEncoder()

    enc.fit(train_new_feature)

    # # # 每一个属性的最大取值数目
    # # print('每一个特征的最大取值数目:', enc.n_values_)
    # # print('所有特征的取值数目总和:', enc.n_values_.sum())

    train_new_feature2 = np.array(enc.transform(train_new_feature).toarray())

    #训练lr
    lr=LogisticRegression()
    lr.fit(train_new_feature2,y_train)
    #测试
    test_new_feature = gbm1.apply(X_test)
    test_new_feature = test_new_feature.reshape(-1, n_trees)
    test_new_feature2 = np.array(enc.transform(test_new_feature).toarray())

    y_predict=lr.predict(test_new_feature2)
    f1_score=metrics.f1_score(y_test,y_predict)
    scores.append(f1_score)
np.mean(scores),np.std(scores)
```




    (0.7713015276197386, 0.034522977378099305)



**通过一系列的特征扩展，我们将baseline gbdt从f1=0.776,std=0.045提升到f1=0.790,std=0.037；此时的最优fetures为ext_cat_fea_dummy_df，接下来我们的目标是去掉那些噪声特征，利用尽可能少的特征去建模达到和之前模型一样的效果；**  
#### 4.1.3 特征选择
（1）基于统计：方差、相关性、gini、info gain、chi2    
（2）基于模型：RFE递归删减特征、训练基模型，选择权值系数较高的特征  

更多:https://www.jianshu.com/p/1c4ec02dd33f

##### 4.1.3 特征选择-方差
将方差较低的特征过滤掉


```python
var_standard_df=StandardScaler().fit_transform(ext_cat_fea_dummy_df[:891])
VarianceThreshold(threshold=0.01).fit_transform(var_standard_df).shape
```




    (891, 492)




```python
#gbdt
classifier=GradientBoostingClassifier()
scores = cross_val_score(classifier, VarianceThreshold(threshold=0.01).fit_transform(ext_cat_fea_dummy_df[:891]), labels, scoring='f1', cv = 5)
np.mean(scores),np.std(scores)
```




    (0.7870500513055333, 0.041771813326123064)



##### 4.1.3 特征选择-相关性
选择与y标签相关性top的因子建模


```python
ext_cat_fea_add_y_df=copy.deepcopy(ext_cat_fea_dummy_df[:891])
ext_cat_fea_add_y_df['Survived']=labels
ext_cat_fea_add_y_df.corr()['Survived'].abs().sort_values(ascending=False).head(5)
```




    Survived    1.000000
    x1 x32      0.545021
    x1 x6       0.543980
    x1          0.543351
    x1^2        0.543351
    Name: Survived, dtype: float64




```python
#选择相关性>0.2的因子建模，注意要去掉Survived
highly_correlated_features = ext_cat_fea_add_y_df.columns[ext_cat_fea_add_y_df.corr()['Survived'].abs() > 0.1]
highly_correlated_features = highly_correlated_features.drop('Survived')
high_corr_features_df=ext_cat_fea_add_y_df[highly_correlated_features]
```


```python
high_corr_features_df.shape
```




    (891, 214)




```python
high_corr_features_df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x0</th>
      <th>x1</th>
      <th>x4</th>
      <th>x5</th>
      <th>x6</th>
      <th>x7</th>
      <th>x8</th>
      <th>x11</th>
      <th>x13</th>
      <th>x16</th>
      <th>...</th>
      <th>x27 x33</th>
      <th>x30 x31</th>
      <th>x31^2</th>
      <th>x31 x32</th>
      <th>x31 x33</th>
      <th>Pclass_Sex_1</th>
      <th>Pclass_Sex_2</th>
      <th>Pclass_Sex_3</th>
      <th>Pclass_Sex_4</th>
      <th>Pclass_Sex_5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.2500</td>
      <td>0.299854</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>71.2833</td>
      <td>0.383838</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.774425</td>
      <td>36.0</td>
      <td>36.0</td>
      <td>21.981370</td>
      <td>4.646550</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7.9250</td>
      <td>0.299854</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>53.1000</td>
      <td>0.468759</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.719601</td>
      <td>24.0</td>
      <td>16.0</td>
      <td>14.334076</td>
      <td>2.878403</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0500</td>
      <td>0.299854</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 214 columns</p>
</div>




```python
#gbdt
classifier=GradientBoostingClassifier()
scores = cross_val_score(classifier, high_corr_features_df, labels, scoring='f1', cv = 5)
np.mean(scores),np.std(scores)
```




    (0.789829696982462, 0.0483977829936617)



**注意：如果出现与y标签相关性很高的因子要引起重视，它可能是由y=>的因子，这种情况应该删掉，因为在test集中这部分因子可能是NaN**
##### 4.1.3 特征选择-Gini指数
gini指数的计算很简单，训练一个决策树就好了


```python
tree = DecisionTreeClassifier()#如果要用信息增益，设置criterion='entropy'
tree.fit(ext_cat_fea_dummy_df[:891],labels)
importances = pd.DataFrame({ 'feature':ext_cat_fea_dummy_df.columns,'importance': tree.feature_importances_}).sort_values('importance', ascending=False)
```


```python
importances.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>81</th>
      <td>x1 x14</td>
      <td>0.308042</td>
    </tr>
    <tr>
      <th>41</th>
      <td>x0 x7</td>
      <td>0.081556</td>
    </tr>
    <tr>
      <th>6</th>
      <td>x6</td>
      <td>0.060521</td>
    </tr>
    <tr>
      <th>438</th>
      <td>x14 x33</td>
      <td>0.053993</td>
    </tr>
    <tr>
      <th>222</th>
      <td>x5 x33</td>
      <td>0.030702</td>
    </tr>
  </tbody>
</table>
</div>




```python
#选择top因子建模
select_features=importances['feature'].tolist()[:50]
#gbdt
classifier=GradientBoostingClassifier()
scores = cross_val_score(classifier, ext_cat_fea_dummy_df[:891][select_features], labels, scoring='f1', cv = 5)
np.mean(scores),np.std(scores)
```




    (0.7993443854210763, 0.05081475219128071)



##### 4.1.3-chi2选择


```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
```


```python
min_max_standard_df=MinMaxScaler().fit_transform(ext_cat_fea_dummy_df[:891])#chi2要求每一项都>0
```


```python
#选择前50个特征
top_50_features=SelectKBest(chi2, k=50).fit_transform(min_max_standard_df, labels)
```


```python
#gbdt
classifier=GradientBoostingClassifier()
scores = cross_val_score(classifier, top_50_features, labels, scoring='f1', cv = 5)
np.mean(scores),np.std(scores)
```




    (0.7547838029844384, 0.0382334783823522)



##### 4.1.3-RFE递归消除


```python
# from sklearn.feature_selection import RFE
# rfe_df=RFE(estimator=GradientBoostingClassifier(), n_features_to_select=50).fit_transform(ext_cat_fea_dummy_df[:891], labels)
```

**这里相当的慢**


```python
# #gbdt
# classifier=GradientBoostingClassifier()
# scores = cross_val_score(classifier, rfe_df, labels, scoring='f1', cv = 5)
# np.mean(scores),np.std(scores)
```

#### 4.1.3-基于模型选特征
其实这里和gini系数的选择一样，通过训练一个模型来选择特征最优特征，然后再去训练一个模型，只是这里选择特征用的模型与训练用的模型一样


```python
#也可以直接用我们gbdt筛选后的特征
tree = GradientBoostingClassifier()
tree.fit(ext_cat_fea_dummy_df[:891],labels)
importances = pd.DataFrame({ 'feature':ext_cat_fea_dummy_df.columns,'importance': tree.feature_importances_}).sort_values('importance', ascending=False)
importances.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>73</th>
      <td>x1 x6</td>
      <td>0.143721</td>
    </tr>
    <tr>
      <th>35</th>
      <td>x0 x1</td>
      <td>0.119379</td>
    </tr>
    <tr>
      <th>72</th>
      <td>x1 x5</td>
      <td>0.069650</td>
    </tr>
    <tr>
      <th>41</th>
      <td>x0 x7</td>
      <td>0.061407</td>
    </tr>
    <tr>
      <th>6</th>
      <td>x6</td>
      <td>0.047584</td>
    </tr>
  </tbody>
</table>
</div>




```python
#选择top因子建模
select_features=importances['feature'].tolist()[:50]
features_select_top_50_df=ext_cat_fea_dummy_df[select_features]
#gbdt
classifier=GradientBoostingClassifier()
scores = cross_val_score(classifier,features_select_top_50_df[:891],labels, scoring='f1', cv = 5)
np.mean(scores),np.std(scores)
```




    (0.8000794313188381, 0.06017298725378909)



也可以使用SelectFromModel自动选择最优的top n因子


```python
kfold= KFold(n_splits=5,random_state=42,shuffle=True)
scores=[]
top_nums=[]
for train_index,test_index in kfold.split(ext_cat_fea_dummy_df[:891],labels):
    X_train=ext_cat_fea_dummy_df.loc[train_index]
    y_train=labels[train_index]
    X_test=ext_cat_fea_dummy_df.loc[test_index]
    y_test=labels[test_index]
    
    select_feature_model = SelectFromModel(GradientBoostingClassifier())
    X_new_train=select_feature_model.fit_transform(X_train,y_train)
    
    X_new_test=select_feature_model.transform(X_test)
    
    _,top_num=X_new_test.shape
    top_nums.append(top_num)
    
    gbdt=GradientBoostingClassifier()
    gbdt.fit(X_new_train,y_train)
    y_predict=gbdt.predict(X_new_test)
    f1_score=metrics.f1_score(y_test,y_predict)
    scores.append(f1_score)
np.mean(scores),np.std(scores)
```




    (0.7907252589751879, 0.05549700964161168)




```python
np.mean(top_nums)
```




    63.8



##### **注意：之前的特征选择操作都不太严谨，因为是将训练集和验证集合并在一起做的特征选择，再用cv的方式看一下**


```python
kfold= KFold(n_splits=5,random_state=42,shuffle=True)
scores=[]
for train_index,test_index in kfold.split(ext_cat_fea_dummy_df[:891],labels):
    X_train=ext_cat_fea_dummy_df.loc[train_index]
    y_train=labels[train_index]
    X_test=ext_cat_fea_dummy_df.loc[test_index]
    y_test=labels[test_index]
    
    tree = GradientBoostingClassifier()
    tree.fit(X_train,y_train)
    importances = pd.DataFrame({ 'feature':X_train.columns,'importance': tree.feature_importances_}).sort_values('importance', ascending=False)
    
    select_features=importances['feature'].tolist()[:50]
    
    
    gbdt=GradientBoostingClassifier()
    gbdt.fit(X_train[select_features],y_train)
    y_predict=gbdt.predict(X_test[select_features])
    f1_score=metrics.f1_score(y_test,y_predict)
    scores.append(f1_score)
np.mean(scores),np.std(scores)
```




    (0.7853628345234904, 0.054688358631340604)



发现其实只需要1/10的因子就能达到和之前一样的效果，甚至更好....  

#### 4.1.4 特征转换
特征选择是从当前的特征集中选择一个子集，而特征转换是对feature/feature和label做某些数学操作，转换后的特征不在是之前特征的子集，比如：  
（1）pca:主成分分析；  
（2）lda:线性判别分析；  
（3）lle:局部线性嵌入；  
（4）ae:自编码；
（5）vae:变分自编码；

##### 4.1.4-pca
pca是一种无监督的线性降维方式，它构建了一个新的正交坐标系，相应的坐标轴分别叫“第一主成分”，“第二主成分”...，且数据在“第一主成分”坐标轴上**方差**最大，“第二主成分”其次，...通常可以只取前n个主成分，将方差较小的主成分理解为**噪声**；  
![avatar](./source/pca示例.png)
更多：https://blog.csdn.net/program_developer/article/details/80632779 


```python
from sklearn.decomposition import PCA
standard_df=StandardScaler().fit_transform(features_select_top_50_df)
X_pca=PCA(n_components=20).fit_transform(standard_df)
```


```python
#gbdt
classifier=GradientBoostingClassifier()
scores = cross_val_score(classifier,X_pca[:891], labels, scoring='f1', cv = 5)
np.mean(scores),np.std(scores)
```




    (0.7900949253855687, 0.035405470298260994)




```python
plt.scatter(X_pca[:891][:, 0], X_pca[:891][:, 1],marker='o',c=labels)
plt.show()
```


![png](./md_resource/output_165_0.png)


#### 4.1.4-lda
lda是一种线性的有监督降维方式，与pca的最大化方差的目标不同，它的目标是找到这样的新坐标轴：**同类样例的投影尽可能近，异类样例的投影点尽可能远**；  
更多：https://blog.csdn.net/weixin_40604987/article/details/79615968


```python
from sklearn.discriminant_analysis  import LinearDiscriminantAnalysis
kfold= KFold(n_splits=5,random_state=42,shuffle=True)
scores=[]
standard_np=StandardScaler().fit_transform(features_select_top_50_df)
for train_index,test_index in kfold.split(standard_np[:891],labels):
    X_train=standard_np[train_index]
    y_train=labels[train_index]
    X_test=standard_np[test_index]
    y_test=labels[test_index]
    
    lda=LinearDiscriminantAnalysis(n_components=20)
    lda.fit(X_train, y_train)
    
    X_new_train=lda.transform(X_train)
    X_new_test=lda.transform(X_test)
    
    gbdt=GradientBoostingClassifier()
    gbdt.fit(X_new_train,y_train)
    y_predict=gbdt.predict(X_new_test)
    f1_score=metrics.f1_score(y_test,y_predict)
    scores.append(f1_score)
np.mean(scores),np.std(scores)
```




    (0.7569760200754199, 0.0505068477051737)




```python
plt.scatter(X_new_train[:891][:, 0], X_new_train[:891][:, 0],marker='o',c=y_train)
plt.show()
```


![png](./md_resource/output_168_0.png)


#### 4.1.4 lle-局部线性嵌入（LocallyLinearEmbedding）
降维时保持样本局部的线性特征
![avatar](./source/lle示例.jpg)
更多：https://www.cnblogs.com/pinard/p/6266408.html?utm_source=itdadao&utm_medium=referral


```python
from sklearn.manifold import LocallyLinearEmbedding
standard_df=StandardScaler().fit_transform(features_select_top_50_df)
X_lle=LocallyLinearEmbedding(n_components=20).fit_transform(standard_df)
```


```python
#gbdt
classifier=GradientBoostingClassifier()
scores = cross_val_score(classifier,X_lle[:891], labels, scoring='f1', cv = 5)
np.mean(scores),np.std(scores)
```




    (0.7460614383554046, 0.048413373247755194)




```python
plt.scatter(X_lle[:891][:, 0], X_lle[:891][:, 1],marker='o',c=labels)
plt.show()
```


![png](./md_resource/output_172_0.png)


#### 4.1.4 ae-自编码
预测目标就是输入目标，可以把它看做一个压缩和解压的过程，如下，通过encoder把一个高维的数据压缩为低维的数据，通过decoder将低维数据还原为高维的数据，这样这个低维的数据可以看做高维数据的一种“不失真”表示；    
![avatar](./source/ae示例.jpg)
ae在图像和nlp方面都有很多深入的应用，比如cv中的vae，nlp中的bert等...  
更多：https://blog.csdn.net/leida_wt/article/details/85052299


```python
from keras.models import Model
from keras.layers import *
# 指定显卡
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
# 动态申请显存
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)
```

    Using TensorFlow backend.
    


```python
#定义网络结构
epochs=200
batch_size=128
input_dim=50

input_layer=Input(shape=(input_dim,))
encode_layer=Dense(2,activation='relu',name='encoder')(input_layer)
decode_layer=Dense(input_dim,activation='tanh')(encode_layer)

model=Model(inputs=input_layer,outputs=decode_layer)
#获取encode_layer层的输出
encode_layer_model = Model(inputs=model.input,outputs=model.get_layer('encoder').output)
model.compile('adam',loss='mse')
```


```python
#预处理输入数据
ae_standard_np=StandardScaler().fit_transform(features_select_top_50_df)
```


```python
X_train=ae_standard_np[:1200]
X_eval=ae_standard_np[1200:]
```


```python
X_train.shape,X_eval.shape
```




    ((1200, 50), (109, 50))




```python
#训练模型
model.fit(X_train,X_train,batch_size=batch_size,epochs=epochs,validation_data=[X_eval,X_eval])
```

    Train on 1200 samples, validate on 109 samples
    Epoch 1/200
    1200/1200 [==============================] - 1s 431us/step - loss: 1.0197 - val_loss: 1.0672
    Epoch 2/200
    1200/1200 [==============================] - 0s 26us/step - loss: 1.0076 - val_loss: 1.0549
    Epoch 3/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.9979 - val_loss: 1.0463
    Epoch 4/200
    1200/1200 [==============================] - 0s 31us/step - loss: 0.9906 - val_loss: 1.0395
    Epoch 5/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.9843 - val_loss: 1.0336
    Epoch 6/200
    1200/1200 [==============================] - ETA: 0s - loss: 1.072 - 0s 25us/step - loss: 0.9777 - val_loss: 1.0273
    Epoch 7/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.9709 - val_loss: 1.0201
    Epoch 8/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.9629 - val_loss: 1.0119
    Epoch 9/200
    1200/1200 [==============================] - 0s 32us/step - loss: 0.9538 - val_loss: 1.0024
    Epoch 10/200
    1200/1200 [==============================] - 0s 35us/step - loss: 0.9436 - val_loss: 0.9919
    Epoch 11/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.9327 - val_loss: 0.9803
    Epoch 12/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.9207 - val_loss: 0.9675
    Epoch 13/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.9089 - val_loss: 0.9531
    Epoch 14/200
    1200/1200 [==============================] - 0s 25us/step - loss: 0.8954 - val_loss: 0.9385
    Epoch 15/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.8825 - val_loss: 0.9230
    Epoch 16/200
    1200/1200 [==============================] - 0s 33us/step - loss: 0.8690 - val_loss: 0.9072
    Epoch 17/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.8553 - val_loss: 0.8916
    Epoch 18/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.8423 - val_loss: 0.8761
    Epoch 19/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.8295 - val_loss: 0.8612
    Epoch 20/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.8172 - val_loss: 0.8468
    Epoch 21/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.8053 - val_loss: 0.8331
    Epoch 22/200
    1200/1200 [==============================] - 0s 24us/step - loss: 0.7940 - val_loss: 0.8199
    Epoch 23/200
    1200/1200 [==============================] - 0s 24us/step - loss: 0.7832 - val_loss: 0.8081
    Epoch 24/200
    1200/1200 [==============================] - 0s 26us/step - loss: 0.7733 - val_loss: 0.7970
    Epoch 25/200
    1200/1200 [==============================] - 0s 26us/step - loss: 0.7639 - val_loss: 0.7866
    Epoch 26/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.7555 - val_loss: 0.7776
    Epoch 27/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.7478 - val_loss: 0.7694
    Epoch 28/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.7408 - val_loss: 0.7620
    Epoch 29/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.7344 - val_loss: 0.7553
    Epoch 30/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.7284 - val_loss: 0.7491
    Epoch 31/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.7230 - val_loss: 0.7435
    Epoch 32/200
    1200/1200 [==============================] - 0s 26us/step - loss: 0.7179 - val_loss: 0.7384
    Epoch 33/200
    1200/1200 [==============================] - 0s 25us/step - loss: 0.7132 - val_loss: 0.7341
    Epoch 34/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.7088 - val_loss: 0.7302
    Epoch 35/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.7046 - val_loss: 0.7265
    Epoch 36/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.7006 - val_loss: 0.7230
    Epoch 37/200
    1200/1200 [==============================] - 0s 25us/step - loss: 0.6968 - val_loss: 0.7197
    Epoch 38/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.6931 - val_loss: 0.7165
    Epoch 39/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.6894 - val_loss: 0.7136
    Epoch 40/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.6859 - val_loss: 0.7109
    Epoch 41/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.6826 - val_loss: 0.7083
    Epoch 42/200
    1200/1200 [==============================] - 0s 25us/step - loss: 0.6794 - val_loss: 0.7061
    Epoch 43/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.6764 - val_loss: 0.7038
    Epoch 44/200
    1200/1200 [==============================] - 0s 26us/step - loss: 0.6734 - val_loss: 0.7015
    Epoch 45/200
    1200/1200 [==============================] - 0s 26us/step - loss: 0.6705 - val_loss: 0.6995
    Epoch 46/200
    1200/1200 [==============================] - 0s 26us/step - loss: 0.6676 - val_loss: 0.6974
    Epoch 47/200
    1200/1200 [==============================] - 0s 31us/step - loss: 0.6648 - val_loss: 0.6955
    Epoch 48/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.6621 - val_loss: 0.6935
    Epoch 49/200
    1200/1200 [==============================] - 0s 25us/step - loss: 0.6595 - val_loss: 0.6914
    Epoch 50/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.6570 - val_loss: 0.6894
    Epoch 51/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.6547 - val_loss: 0.6874
    Epoch 52/200
    1200/1200 [==============================] - 0s 25us/step - loss: 0.6525 - val_loss: 0.6855
    Epoch 53/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.6504 - val_loss: 0.6838
    Epoch 54/200
    1200/1200 [==============================] - 0s 26us/step - loss: 0.6484 - val_loss: 0.6823
    Epoch 55/200
    1200/1200 [==============================] - 0s 26us/step - loss: 0.6466 - val_loss: 0.6809
    Epoch 56/200
    1200/1200 [==============================] - 0s 24us/step - loss: 0.6447 - val_loss: 0.6794
    Epoch 57/200
    1200/1200 [==============================] - 0s 25us/step - loss: 0.6430 - val_loss: 0.6781
    Epoch 58/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.6413 - val_loss: 0.6769
    Epoch 59/200
    1200/1200 [==============================] - 0s 32us/step - loss: 0.6398 - val_loss: 0.6757
    Epoch 60/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.6382 - val_loss: 0.6744
    Epoch 61/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.6367 - val_loss: 0.6733
    Epoch 62/200
    1200/1200 [==============================] - 0s 32us/step - loss: 0.6353 - val_loss: 0.6718
    Epoch 63/200
    1200/1200 [==============================] - 0s 47us/step - loss: 0.6339 - val_loss: 0.6708
    Epoch 64/200
    1200/1200 [==============================] - 0s 42us/step - loss: 0.6326 - val_loss: 0.6694
    Epoch 65/200
    1200/1200 [==============================] - ETA: 0s - loss: 0.649 - 0s 33us/step - loss: 0.6313 - val_loss: 0.6685
    Epoch 66/200
    1200/1200 [==============================] - 0s 31us/step - loss: 0.6301 - val_loss: 0.6675
    Epoch 67/200
    1200/1200 [==============================] - 0s 33us/step - loss: 0.6289 - val_loss: 0.6665
    Epoch 68/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.6278 - val_loss: 0.6657
    Epoch 69/200
    1200/1200 [==============================] - 0s 33us/step - loss: 0.6267 - val_loss: 0.6647
    Epoch 70/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.6256 - val_loss: 0.6633
    Epoch 71/200
    1200/1200 [==============================] - 0s 32us/step - loss: 0.6246 - val_loss: 0.6620
    Epoch 72/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.6236 - val_loss: 0.6612
    Epoch 73/200
    1200/1200 [==============================] - 0s 26us/step - loss: 0.6226 - val_loss: 0.6606
    Epoch 74/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.6217 - val_loss: 0.6595
    Epoch 75/200
    1200/1200 [==============================] - ETA: 0s - loss: 0.707 - 0s 28us/step - loss: 0.6208 - val_loss: 0.6584
    Epoch 76/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.6199 - val_loss: 0.6576
    Epoch 77/200
    1200/1200 [==============================] - 0s 25us/step - loss: 0.6191 - val_loss: 0.6567
    Epoch 78/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.6182 - val_loss: 0.6565
    Epoch 79/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.6174 - val_loss: 0.6556
    Epoch 80/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.6166 - val_loss: 0.6546
    Epoch 81/200
    1200/1200 [==============================] - 0s 35us/step - loss: 0.6159 - val_loss: 0.6539
    Epoch 82/200
    1200/1200 [==============================] - 0s 34us/step - loss: 0.6152 - val_loss: 0.6530
    Epoch 83/200
    1200/1200 [==============================] - 0s 37us/step - loss: 0.6145 - val_loss: 0.6522
    Epoch 84/200
    1200/1200 [==============================] - 0s 31us/step - loss: 0.6138 - val_loss: 0.6518
    Epoch 85/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.6131 - val_loss: 0.6513
    Epoch 86/200
    1200/1200 [==============================] - 0s 32us/step - loss: 0.6125 - val_loss: 0.6506
    Epoch 87/200
    1200/1200 [==============================] - 0s 33us/step - loss: 0.6119 - val_loss: 0.6500
    Epoch 88/200
    1200/1200 [==============================] - 0s 35us/step - loss: 0.6113 - val_loss: 0.6497
    Epoch 89/200
    1200/1200 [==============================] - 0s 37us/step - loss: 0.6107 - val_loss: 0.6494
    Epoch 90/200
    1200/1200 [==============================] - 0s 42us/step - loss: 0.6102 - val_loss: 0.6489
    Epoch 91/200
    1200/1200 [==============================] - 0s 47us/step - loss: 0.6096 - val_loss: 0.6480
    Epoch 92/200
    1200/1200 [==============================] - 0s 45us/step - loss: 0.6091 - val_loss: 0.6474
    Epoch 93/200
    1200/1200 [==============================] - 0s 42us/step - loss: 0.6086 - val_loss: 0.6467
    Epoch 94/200
    1200/1200 [==============================] - 0s 35us/step - loss: 0.6081 - val_loss: 0.6468
    Epoch 95/200
    1200/1200 [==============================] - 0s 49us/step - loss: 0.6076 - val_loss: 0.6464
    Epoch 96/200
    1200/1200 [==============================] - 0s 40us/step - loss: 0.6071 - val_loss: 0.6460
    Epoch 97/200
    1200/1200 [==============================] - 0s 39us/step - loss: 0.6066 - val_loss: 0.6454
    Epoch 98/200
    1200/1200 [==============================] - 0s 36us/step - loss: 0.6062 - val_loss: 0.6451
    Epoch 99/200
    1200/1200 [==============================] - 0s 48us/step - loss: 0.6058 - val_loss: 0.6444
    Epoch 100/200
    1200/1200 [==============================] - 0s 50us/step - loss: 0.6053 - val_loss: 0.6441
    Epoch 101/200
    1200/1200 [==============================] - 0s 39us/step - loss: 0.6050 - val_loss: 0.6437
    Epoch 102/200
    1200/1200 [==============================] - 0s 35us/step - loss: 0.6046 - val_loss: 0.6430
    Epoch 103/200
    1200/1200 [==============================] - 0s 32us/step - loss: 0.6041 - val_loss: 0.6426
    Epoch 104/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.6037 - val_loss: 0.6424
    Epoch 105/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.6034 - val_loss: 0.6419
    Epoch 106/200
    1200/1200 [==============================] - 0s 32us/step - loss: 0.6031 - val_loss: 0.6416
    Epoch 107/200
    1200/1200 [==============================] - 0s 36us/step - loss: 0.6028 - val_loss: 0.6415
    Epoch 108/200
    1200/1200 [==============================] - 0s 39us/step - loss: 0.6023 - val_loss: 0.6411
    Epoch 109/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.6019 - val_loss: 0.6408
    Epoch 110/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.6015 - val_loss: 0.6403
    Epoch 111/200
    1200/1200 [==============================] - 0s 31us/step - loss: 0.6013 - val_loss: 0.6399
    Epoch 112/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.6008 - val_loss: 0.6397
    Epoch 113/200
    1200/1200 [==============================] - 0s 31us/step - loss: 0.6005 - val_loss: 0.6396
    Epoch 114/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.6001 - val_loss: 0.6391
    Epoch 115/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.5998 - val_loss: 0.6383
    Epoch 116/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.5995 - val_loss: 0.6380
    Epoch 117/200
    1200/1200 [==============================] - 0s 32us/step - loss: 0.5992 - val_loss: 0.6379
    Epoch 118/200
    1200/1200 [==============================] - 0s 36us/step - loss: 0.5989 - val_loss: 0.6377
    Epoch 119/200
    1200/1200 [==============================] - 0s 32us/step - loss: 0.5986 - val_loss: 0.6375
    Epoch 120/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.5983 - val_loss: 0.6374
    Epoch 121/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.5981 - val_loss: 0.6370
    Epoch 122/200
    1200/1200 [==============================] - 0s 42us/step - loss: 0.5978 - val_loss: 0.6367
    Epoch 123/200
    1200/1200 [==============================] - 0s 32us/step - loss: 0.5976 - val_loss: 0.6364
    Epoch 124/200
    1200/1200 [==============================] - 0s 34us/step - loss: 0.5973 - val_loss: 0.6363
    Epoch 125/200
    1200/1200 [==============================] - 0s 34us/step - loss: 0.5970 - val_loss: 0.6363
    Epoch 126/200
    1200/1200 [==============================] - 0s 35us/step - loss: 0.5968 - val_loss: 0.6360
    Epoch 127/200
    1200/1200 [==============================] - 0s 37us/step - loss: 0.5966 - val_loss: 0.6358
    Epoch 128/200
    1200/1200 [==============================] - 0s 33us/step - loss: 0.5964 - val_loss: 0.6359
    Epoch 129/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.5961 - val_loss: 0.6355
    Epoch 130/200
    1200/1200 [==============================] - 0s 33us/step - loss: 0.5960 - val_loss: 0.6355
    Epoch 131/200
    1200/1200 [==============================] - 0s 32us/step - loss: 0.5957 - val_loss: 0.6352
    Epoch 132/200
    1200/1200 [==============================] - 0s 32us/step - loss: 0.5956 - val_loss: 0.6351
    Epoch 133/200
    1200/1200 [==============================] - 0s 33us/step - loss: 0.5953 - val_loss: 0.6348
    Epoch 134/200
    1200/1200 [==============================] - 0s 31us/step - loss: 0.5952 - val_loss: 0.6349
    Epoch 135/200
    1200/1200 [==============================] - 0s 34us/step - loss: 0.5950 - val_loss: 0.6348
    Epoch 136/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.5948 - val_loss: 0.6347
    Epoch 137/200
    1200/1200 [==============================] - 0s 24us/step - loss: 0.5946 - val_loss: 0.6346
    Epoch 138/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.5945 - val_loss: 0.6343
    Epoch 139/200
    1200/1200 [==============================] - 0s 37us/step - loss: 0.5943 - val_loss: 0.6342
    Epoch 140/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.5941 - val_loss: 0.6339
    Epoch 141/200
    1200/1200 [==============================] - 0s 32us/step - loss: 0.5939 - val_loss: 0.6338
    Epoch 142/200
    1200/1200 [==============================] - 0s 40us/step - loss: 0.5938 - val_loss: 0.6338
    Epoch 143/200
    1200/1200 [==============================] - 0s 33us/step - loss: 0.5936 - val_loss: 0.6336
    Epoch 144/200
    1200/1200 [==============================] - 0s 31us/step - loss: 0.5934 - val_loss: 0.6336
    Epoch 145/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.5933 - val_loss: 0.6336
    Epoch 146/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.5931 - val_loss: 0.6332
    Epoch 147/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.5930 - val_loss: 0.6332
    Epoch 148/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.5929 - val_loss: 0.6330
    Epoch 149/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.5927 - val_loss: 0.6328
    Epoch 150/200
    1200/1200 [==============================] - 0s 31us/step - loss: 0.5926 - val_loss: 0.6327
    Epoch 151/200
    1200/1200 [==============================] - 0s 32us/step - loss: 0.5924 - val_loss: 0.6323
    Epoch 152/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.5923 - val_loss: 0.6320
    Epoch 153/200
    1200/1200 [==============================] - 0s 26us/step - loss: 0.5921 - val_loss: 0.6319
    Epoch 154/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.5920 - val_loss: 0.6319
    Epoch 155/200
    1200/1200 [==============================] - 0s 24us/step - loss: 0.5919 - val_loss: 0.6319
    Epoch 156/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.5918 - val_loss: 0.6318
    Epoch 157/200
    1200/1200 [==============================] - 0s 26us/step - loss: 0.5916 - val_loss: 0.6316
    Epoch 158/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.5915 - val_loss: 0.6314
    Epoch 159/200
    1200/1200 [==============================] - 0s 32us/step - loss: 0.5914 - val_loss: 0.6314
    Epoch 160/200
    1200/1200 [==============================] - 0s 31us/step - loss: 0.5913 - val_loss: 0.6316
    Epoch 161/200
    1200/1200 [==============================] - 0s 31us/step - loss: 0.5912 - val_loss: 0.6311
    Epoch 162/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.5911 - val_loss: 0.6309
    Epoch 163/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.5909 - val_loss: 0.6306
    Epoch 164/200
    1200/1200 [==============================] - 0s 26us/step - loss: 0.5908 - val_loss: 0.6306
    Epoch 165/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.5907 - val_loss: 0.6303
    Epoch 166/200
    1200/1200 [==============================] - 0s 25us/step - loss: 0.5906 - val_loss: 0.6306
    Epoch 167/200
    1200/1200 [==============================] - 0s 25us/step - loss: 0.5905 - val_loss: 0.6304
    Epoch 168/200
    1200/1200 [==============================] - 0s 25us/step - loss: 0.5904 - val_loss: 0.6301
    Epoch 169/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.5902 - val_loss: 0.6305
    Epoch 170/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.5902 - val_loss: 0.6305
    Epoch 171/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.5901 - val_loss: 0.6305
    Epoch 172/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.5900 - val_loss: 0.6297
    Epoch 173/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.5899 - val_loss: 0.6297
    Epoch 174/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.5898 - val_loss: 0.6299
    Epoch 175/200
    1200/1200 [==============================] - 0s 26us/step - loss: 0.5896 - val_loss: 0.6299
    Epoch 176/200
    1200/1200 [==============================] - 0s 26us/step - loss: 0.5896 - val_loss: 0.6297
    Epoch 177/200
    1200/1200 [==============================] - 0s 26us/step - loss: 0.5895 - val_loss: 0.6295
    Epoch 178/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.5894 - val_loss: 0.6296
    Epoch 179/200
    1200/1200 [==============================] - ETA: 0s - loss: 0.452 - 0s 28us/step - loss: 0.5892 - val_loss: 0.6293
    Epoch 180/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.5892 - val_loss: 0.6293
    Epoch 181/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.5891 - val_loss: 0.6294
    Epoch 182/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.5890 - val_loss: 0.6292
    Epoch 183/200
    1200/1200 [==============================] - 0s 32us/step - loss: 0.5889 - val_loss: 0.6289
    Epoch 184/200
    1200/1200 [==============================] - 0s 36us/step - loss: 0.5888 - val_loss: 0.6289
    Epoch 185/200
    1200/1200 [==============================] - 0s 41us/step - loss: 0.5888 - val_loss: 0.6288
    Epoch 186/200
    1200/1200 [==============================] - 0s 36us/step - loss: 0.5887 - val_loss: 0.6288
    Epoch 187/200
    1200/1200 [==============================] - 0s 33us/step - loss: 0.5886 - val_loss: 0.6285
    Epoch 188/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.5885 - val_loss: 0.6287
    Epoch 189/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.5884 - val_loss: 0.6286
    Epoch 190/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.5883 - val_loss: 0.6285
    Epoch 191/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.5882 - val_loss: 0.6282
    Epoch 192/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.5882 - val_loss: 0.6280
    Epoch 193/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.5881 - val_loss: 0.6280
    Epoch 194/200
    1200/1200 [==============================] - 0s 32us/step - loss: 0.5881 - val_loss: 0.6280
    Epoch 195/200
    1200/1200 [==============================] - 0s 31us/step - loss: 0.5880 - val_loss: 0.6281
    Epoch 196/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.5879 - val_loss: 0.6282
    Epoch 197/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.5878 - val_loss: 0.6279
    Epoch 198/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.5877 - val_loss: 0.6279
    Epoch 199/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.5877 - val_loss: 0.6276
    Epoch 200/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.5876 - val_loss: 0.6273
    




    <keras.callbacks.History at 0x16263a29278>




```python
X_eval[0]
```




    array([ 0.82766931,  1.93572254,  0.0336646 ,  1.07885292, -0.37019258,
            1.49602053,  0.96276563,  1.25937023, -0.22643032,  1.34499549,
            2.35787478, -0.51502135,  2.56645764,  2.24948554,  1.17060062,
           -0.3250884 , -0.36321285, -0.39213719,  0.6212742 ,  1.06873154,
           -0.18490606,  1.10470414, -0.08773955, -0.35177719,  0.87936559,
            0.55780901, -0.14069042, -0.14198762,  2.42674631, -0.12079052,
           -0.77146693, -0.12027696, -0.17571017,  1.34499549, -0.44771505,
           -0.10208177, -0.19266846,  1.67113775,  0.54501064, -0.47507471,
           -0.12027696, -0.32342025,  2.25917848, -0.29429097, -0.31815144,
           -0.24310938, -0.25669649, -0.34643049, -0.05251453, -0.44281517])




```python
model.predict(X_eval)[0]
```




    array([ 0.84259963,  0.89565194, -0.1657296 ,  0.3699348 , -0.46292907,
            0.74069   ,  0.52614015, -0.3312198 , -0.6547402 ,  0.8768692 ,
            0.8939867 , -0.9338216 ,  0.55263174,  0.7648835 , -0.33672792,
           -0.41581   , -0.5297965 , -0.6790321 ,  0.6871953 ,  0.1279342 ,
           -0.6482408 , -0.19741514,  0.01378946,  0.31400046,  0.7704295 ,
           -0.21264473, -0.5385604 , -0.50612307,  0.46878704, -0.632584  ,
           -0.9613648 , -0.1219368 , -0.03166279,  0.8712636 , -0.60364246,
            0.07714609, -0.6353725 ,  0.9529996 ,  0.25829062, -0.6812642 ,
           -0.12176671,  0.2922581 ,  0.8147812 ,  0.16231813, -0.10317522,
           -0.35576746,  0.08484281, -0.15319254,  0.03710562, -0.6286148 ],
          dtype=float32)




```python
encode_layer_model.predict(X_eval)[0]
```




    array([1.7635584, 0.       ], dtype=float32)




```python
np.mean(X_eval-model.predict(X_eval)),np.std(X_eval-model.predict(X_eval))
```




    (0.08181091146516974, 0.7877961965753176)




```python
ae_new_features=encode_layer_model.predict(ae_standard_np)
```


```python
ae_new_features.shape
```




    (1309, 2)




```python
#gbdt
classifier=GradientBoostingClassifier()
scores = cross_val_score(classifier,ae_new_features[:891], labels, scoring='f1', cv = 5)
np.mean(scores),np.std(scores)
```




    (0.6780143813765033, 0.017743977525973863)




```python
plt.scatter(ae_new_features[:891][:, 0], ae_new_features[:891][:, 1],marker='o',c=labels)
plt.show()
```


![png](./md_resource/output_187_0.png)


#### 4.1.4 vae-变分自编码
变分自编码即在中间添加白噪声，增强自编码的泛化能力  
更多：https://blog.csdn.net/weixin_41923961/article/details/81586082


```python
#定义网络结构
from keras import backend as K
from keras import objectives
latent_dim = 2
intermediate_dim = 32
epsilon_std = 1.0

x = Input(shape=(input_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

#my tips:Gauss sampling,sample Z
def sampling(args): 
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(latent_dim,), mean=0.,stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon
 
# note that "output_shape" isn't necessary with the TensorFlow backend
# my tips:get sample z(encoded)
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
 
# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(input_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)
 
#my tips:loss(restruct X)+KL
def vae_loss(x, x_decoded_mean):
    #my tips:logloss
    xent_loss = input_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    #my tips:see paper's appendix B
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss
 
vae = Model(x, x_decoded_mean)
vae.compile(optimizer='rmsprop', loss=vae_loss)
```


```python
vae.fit(X_train,X_train,batch_size=batch_size,epochs=epochs,validation_data=[X_eval,X_eval])
```

    Train on 1200 samples, validate on 109 samples
    Epoch 1/200
    1200/1200 [==============================] - 0s 385us/step - loss: 36.0322 - val_loss: 34.7021
    Epoch 2/200
    1200/1200 [==============================] - 0s 46us/step - loss: 33.8301 - val_loss: 31.6163
    Epoch 3/200
    1200/1200 [==============================] - 0s 41us/step - loss: 32.4203 - val_loss: 32.2834
    Epoch 4/200
    1200/1200 [==============================] - 0s 40us/step - loss: 29.9816 - val_loss: 30.0823
    Epoch 5/200
    1200/1200 [==============================] - 0s 44us/step - loss: 26.8610 - val_loss: 26.0733
    Epoch 6/200
    1200/1200 [==============================] - 0s 46us/step - loss: 23.5636 - val_loss: 25.6523
    Epoch 7/200
    1200/1200 [==============================] - 0s 44us/step - loss: 18.2856 - val_loss: 9.0534
    Epoch 8/200
    1200/1200 [==============================] - 0s 44us/step - loss: 11.1824 - val_loss: 6.3278
    Epoch 9/200
    1200/1200 [==============================] - 0s 42us/step - loss: 6.7168 - val_loss: 4.9056
    Epoch 10/200
    1200/1200 [==============================] - 0s 49us/step - loss: -2.0989 - val_loss: -6.4802
    Epoch 11/200
    1200/1200 [==============================] - ETA: 0s - loss: -4.34 - 0s 47us/step - loss: -8.4981 - val_loss: -16.3702
    Epoch 12/200
    1200/1200 [==============================] - 0s 44us/step - loss: -20.0082 - val_loss: -29.4273
    Epoch 13/200
    1200/1200 [==============================] - 0s 49us/step - loss: -25.1697 - val_loss: -32.0603
    Epoch 14/200
    1200/1200 [==============================] - 0s 48us/step - loss: -35.0976 - val_loss: -42.6215
    Epoch 15/200
    1200/1200 [==============================] - 0s 46us/step - loss: -44.6793 - val_loss: -54.9334
    Epoch 16/200
    1200/1200 [==============================] - 0s 49us/step - loss: -54.1252 - val_loss: -66.2804
    Epoch 17/200
    1200/1200 [==============================] - 0s 64us/step - loss: -62.6568 - val_loss: -69.8099
    Epoch 18/200
    1200/1200 [==============================] - 0s 71us/step - loss: -79.3135 - val_loss: -92.0042
    Epoch 19/200
    1200/1200 [==============================] - 0s 53us/step - loss: -85.6046 - val_loss: -98.2631
    Epoch 20/200
    1200/1200 [==============================] - 0s 45us/step - loss: -97.7223 - val_loss: -110.7350
    Epoch 21/200
    1200/1200 [==============================] - 0s 43us/step - loss: -108.8734 - val_loss: -121.7681
    Epoch 22/200
    1200/1200 [==============================] - 0s 39us/step - loss: -117.2519 - val_loss: -122.9283
    Epoch 23/200
    1200/1200 [==============================] - 0s 43us/step - loss: -124.0625 - val_loss: -130.6821
    Epoch 24/200
    1200/1200 [==============================] - 0s 39us/step - loss: -127.9313 - val_loss: -134.6431
    Epoch 25/200
    1200/1200 [==============================] - 0s 41us/step - loss: -136.6242 - val_loss: -144.6512
    Epoch 26/200
    1200/1200 [==============================] - 0s 39us/step - loss: -141.2549 - val_loss: -149.7274
    Epoch 27/200
    1200/1200 [==============================] - 0s 41us/step - loss: -147.2086 - val_loss: -150.6741
    Epoch 28/200
    1200/1200 [==============================] - 0s 39us/step - loss: -150.1748 - val_loss: -156.0731
    Epoch 29/200
    1200/1200 [==============================] - 0s 42us/step - loss: -153.8265 - val_loss: -155.1350
    Epoch 30/200
    1200/1200 [==============================] - 0s 43us/step - loss: -159.5195 - val_loss: -164.9775
    Epoch 31/200
    1200/1200 [==============================] - 0s 40us/step - loss: -162.5176 - val_loss: -167.6422
    Epoch 32/200
    1200/1200 [==============================] - 0s 42us/step - loss: -164.5841 - val_loss: -162.5660
    Epoch 33/200
    1200/1200 [==============================] - 0s 38us/step - loss: -169.3451 - val_loss: -172.1781
    Epoch 34/200
    1200/1200 [==============================] - 0s 42us/step - loss: -172.9661 - val_loss: -170.5206
    Epoch 35/200
    1200/1200 [==============================] - 0s 41us/step - loss: -175.2212 - val_loss: -178.1695
    Epoch 36/200
    1200/1200 [==============================] - 0s 39us/step - loss: -176.2896 - val_loss: -174.4894
    Epoch 37/200
    1200/1200 [==============================] - 0s 43us/step - loss: -180.8014 - val_loss: -180.8597
    Epoch 38/200
    1200/1200 [==============================] - 0s 39us/step - loss: -182.6409 - val_loss: -183.8219
    Epoch 39/200
    1200/1200 [==============================] - 0s 42us/step - loss: -184.8439 - val_loss: -186.8923
    Epoch 40/200
    1200/1200 [==============================] - 0s 45us/step - loss: -187.4653 - val_loss: -187.9583
    Epoch 41/200
    1200/1200 [==============================] - 0s 41us/step - loss: -188.7138 - val_loss: -189.6729
    Epoch 42/200
    1200/1200 [==============================] - 0s 42us/step - loss: -191.0132 - val_loss: -192.4526
    Epoch 43/200
    1200/1200 [==============================] - 0s 39us/step - loss: -193.3996 - val_loss: -191.4754
    Epoch 44/200
    1200/1200 [==============================] - 0s 40us/step - loss: -195.4018 - val_loss: -192.4886
    Epoch 45/200
    1200/1200 [==============================] - 0s 43us/step - loss: -197.8354 - val_loss: -198.9803
    Epoch 46/200
    1200/1200 [==============================] - 0s 41us/step - loss: -199.3286 - val_loss: -199.6373
    Epoch 47/200
    1200/1200 [==============================] - 0s 49us/step - loss: -199.9366 - val_loss: -195.8194
    Epoch 48/200
    1200/1200 [==============================] - 0s 43us/step - loss: -202.2465 - val_loss: -201.8374
    Epoch 49/200
    1200/1200 [==============================] - 0s 40us/step - loss: -204.1095 - val_loss: -203.7368
    Epoch 50/200
    1200/1200 [==============================] - 0s 41us/step - loss: -206.1324 - val_loss: -205.3748
    Epoch 51/200
    1200/1200 [==============================] - 0s 43us/step - loss: -207.9977 - val_loss: -208.2765
    Epoch 52/200
    1200/1200 [==============================] - 0s 46us/step - loss: -206.9545 - val_loss: -207.8138
    Epoch 53/200
    1200/1200 [==============================] - 0s 62us/step - loss: -210.4672 - val_loss: -210.5954
    Epoch 54/200
    1200/1200 [==============================] - 0s 52us/step - loss: -211.8306 - val_loss: -211.2946
    Epoch 55/200
    1200/1200 [==============================] - 0s 52us/step - loss: -213.8054 - val_loss: -211.6980
    Epoch 56/200
    1200/1200 [==============================] - 0s 43us/step - loss: -214.7484 - val_loss: -213.9301
    Epoch 57/200
    1200/1200 [==============================] - 0s 42us/step - loss: -215.6093 - val_loss: -215.3631
    Epoch 58/200
    1200/1200 [==============================] - 0s 37us/step - loss: -217.7883 - val_loss: -215.9731
    Epoch 59/200
    1200/1200 [==============================] - 0s 38us/step - loss: -218.1276 - val_loss: -216.4344
    Epoch 60/200
    1200/1200 [==============================] - 0s 39us/step - loss: -220.2750 - val_loss: -217.1644
    Epoch 61/200
    1200/1200 [==============================] - 0s 39us/step - loss: -218.7617 - val_loss: -219.0812
    Epoch 62/200
    1200/1200 [==============================] - 0s 37us/step - loss: -221.8987 - val_loss: -221.1280
    Epoch 63/200
    1200/1200 [==============================] - 0s 40us/step - loss: -221.3936 - val_loss: -221.3479
    Epoch 64/200
    1200/1200 [==============================] - 0s 37us/step - loss: -223.6522 - val_loss: -220.7797
    Epoch 65/200
    1200/1200 [==============================] - 0s 43us/step - loss: -225.2709 - val_loss: -223.5696
    Epoch 66/200
    1200/1200 [==============================] - 0s 40us/step - loss: -225.4791 - val_loss: -224.2963
    Epoch 67/200
    1200/1200 [==============================] - 0s 40us/step - loss: -226.3259 - val_loss: -225.5609
    Epoch 68/200
    1200/1200 [==============================] - 0s 43us/step - loss: -226.7861 - val_loss: -225.8561
    Epoch 69/200
    1200/1200 [==============================] - 0s 37us/step - loss: -228.7305 - val_loss: -222.9846
    Epoch 70/200
    1200/1200 [==============================] - 0s 37us/step - loss: -228.3335 - val_loss: -226.6503
    Epoch 71/200
    1200/1200 [==============================] - 0s 37us/step - loss: -229.9302 - val_loss: -228.6987
    Epoch 72/200
    1200/1200 [==============================] - 0s 40us/step - loss: -230.0094 - val_loss: -229.6452
    Epoch 73/200
    1200/1200 [==============================] - 0s 38us/step - loss: -230.9542 - val_loss: -229.7367
    Epoch 74/200
    1200/1200 [==============================] - 0s 42us/step - loss: -231.8372 - val_loss: -230.7162
    Epoch 75/200
    1200/1200 [==============================] - 0s 42us/step - loss: -232.7775 - val_loss: -231.3263
    Epoch 76/200
    1200/1200 [==============================] - 0s 43us/step - loss: -233.9794 - val_loss: -231.1739
    Epoch 77/200
    1200/1200 [==============================] - 0s 37us/step - loss: -233.5089 - val_loss: -233.4069
    Epoch 78/200
    1200/1200 [==============================] - 0s 41us/step - loss: -233.5800 - val_loss: -230.3409
    Epoch 79/200
    1200/1200 [==============================] - 0s 42us/step - loss: -234.5334 - val_loss: -234.8030
    Epoch 80/200
    1200/1200 [==============================] - 0s 43us/step - loss: -235.5616 - val_loss: -235.6455
    Epoch 81/200
    1200/1200 [==============================] - 0s 39us/step - loss: -236.4999 - val_loss: -235.4594
    Epoch 82/200
    1200/1200 [==============================] - 0s 37us/step - loss: -237.3346 - val_loss: -235.1954
    Epoch 83/200
    1200/1200 [==============================] - 0s 38us/step - loss: -237.3713 - val_loss: -237.6295
    Epoch 84/200
    1200/1200 [==============================] - 0s 45us/step - loss: -237.2432 - val_loss: -237.3528
    Epoch 85/200
    1200/1200 [==============================] - 0s 40us/step - loss: -239.1301 - val_loss: -237.9530
    Epoch 86/200
    1200/1200 [==============================] - 0s 42us/step - loss: -239.6052 - val_loss: -237.2639
    Epoch 87/200
    1200/1200 [==============================] - 0s 42us/step - loss: -240.2287 - val_loss: -240.5369
    Epoch 88/200
    1200/1200 [==============================] - 0s 50us/step - loss: -240.4089 - val_loss: -237.0349
    Epoch 89/200
    1200/1200 [==============================] - 0s 45us/step - loss: -240.1467 - val_loss: -239.0190
    Epoch 90/200
    1200/1200 [==============================] - 0s 52us/step - loss: -241.0824 - val_loss: -241.3419
    Epoch 91/200
    1200/1200 [==============================] - 0s 74us/step - loss: -241.7599 - val_loss: -241.4667
    Epoch 92/200
    1200/1200 [==============================] - 0s 67us/step - loss: -242.0052 - val_loss: -241.6755
    Epoch 93/200
    1200/1200 [==============================] - 0s 50us/step - loss: -241.1170 - val_loss: -240.2891
    Epoch 94/200
    1200/1200 [==============================] - 0s 60us/step - loss: -242.9253 - val_loss: -241.1100
    Epoch 95/200
    1200/1200 [==============================] - 0s 43us/step - loss: -242.8805 - val_loss: -243.4879
    Epoch 96/200
    1200/1200 [==============================] - 0s 51us/step - loss: -243.2285 - val_loss: -243.0897
    Epoch 97/200
    1200/1200 [==============================] - 0s 45us/step - loss: -243.7404 - val_loss: -238.6712
    Epoch 98/200
    1200/1200 [==============================] - 0s 47us/step - loss: -243.7135 - val_loss: -244.2330
    Epoch 99/200
    1200/1200 [==============================] - 0s 43us/step - loss: -243.9751 - val_loss: -244.6698
    Epoch 100/200
    1200/1200 [==============================] - 0s 50us/step - loss: -244.3595 - val_loss: -244.6985
    Epoch 101/200
    1200/1200 [==============================] - 0s 62us/step - loss: -245.7793 - val_loss: -240.5719
    Epoch 102/200
    1200/1200 [==============================] - 0s 47us/step - loss: -245.5833 - val_loss: -244.5527
    Epoch 103/200
    1200/1200 [==============================] - 0s 50us/step - loss: -245.1174 - val_loss: -245.6055
    Epoch 104/200
    1200/1200 [==============================] - 0s 63us/step - loss: -246.6947 - val_loss: -244.5897
    Epoch 105/200
    1200/1200 [==============================] - 0s 66us/step - loss: -246.0648 - val_loss: -247.2596
    Epoch 106/200
    1200/1200 [==============================] - 0s 49us/step - loss: -246.3280 - val_loss: -245.1700
    Epoch 107/200
    1200/1200 [==============================] - 0s 51us/step - loss: -246.9919 - val_loss: -244.3329
    Epoch 108/200
    1200/1200 [==============================] - 0s 47us/step - loss: -247.8342 - val_loss: -246.8969
    Epoch 109/200
    1200/1200 [==============================] - 0s 41us/step - loss: -247.9137 - val_loss: -248.1945
    Epoch 110/200
    1200/1200 [==============================] - 0s 41us/step - loss: -248.4584 - val_loss: -247.3998
    Epoch 111/200
    1200/1200 [==============================] - 0s 41us/step - loss: -248.6404 - val_loss: -249.1832
    Epoch 112/200
    1200/1200 [==============================] - 0s 45us/step - loss: -247.1656 - val_loss: -249.1751
    Epoch 113/200
    1200/1200 [==============================] - 0s 43us/step - loss: -248.9547 - val_loss: -248.4621
    Epoch 114/200
    1200/1200 [==============================] - 0s 59us/step - loss: -249.1383 - val_loss: -248.1487
    Epoch 115/200
    1200/1200 [==============================] - 0s 48us/step - loss: -249.1573 - val_loss: -249.6732
    Epoch 116/200
    1200/1200 [==============================] - 0s 42us/step - loss: -249.4962 - val_loss: -249.2317
    Epoch 117/200
    1200/1200 [==============================] - 0s 42us/step - loss: -250.1192 - val_loss: -249.3401
    Epoch 118/200
    1200/1200 [==============================] - 0s 43us/step - loss: -250.7836 - val_loss: -250.8305
    Epoch 119/200
    1200/1200 [==============================] - 0s 48us/step - loss: -248.8736 - val_loss: -251.2466
    Epoch 120/200
    1200/1200 [==============================] - 0s 44us/step - loss: -250.1215 - val_loss: -250.8976
    Epoch 121/200
    1200/1200 [==============================] - 0s 53us/step - loss: -250.8677 - val_loss: -250.9131
    Epoch 122/200
    1200/1200 [==============================] - 0s 52us/step - loss: -251.9341 - val_loss: -250.6540
    Epoch 123/200
    1200/1200 [==============================] - 0s 52us/step - loss: -251.9594 - val_loss: -250.8118
    Epoch 124/200
    1200/1200 [==============================] - 0s 40us/step - loss: -250.3285 - val_loss: -252.8027
    Epoch 125/200
    1200/1200 [==============================] - 0s 41us/step - loss: -251.6773 - val_loss: -250.9750
    Epoch 126/200
    1200/1200 [==============================] - 0s 40us/step - loss: -251.9734 - val_loss: -252.5638
    Epoch 127/200
    1200/1200 [==============================] - 0s 42us/step - loss: -250.7873 - val_loss: -251.7976
    Epoch 128/200
    1200/1200 [==============================] - 0s 39us/step - loss: -252.1939 - val_loss: -253.2203
    Epoch 129/200
    1200/1200 [==============================] - 0s 39us/step - loss: -251.3227 - val_loss: -252.0128
    Epoch 130/200
    1200/1200 [==============================] - 0s 37us/step - loss: -252.7467 - val_loss: -251.7596
    Epoch 131/200
    1200/1200 [==============================] - 0s 38us/step - loss: -252.6191 - val_loss: -247.9310
    Epoch 132/200
    1200/1200 [==============================] - 0s 39us/step - loss: -251.9406 - val_loss: -254.3064
    Epoch 133/200
    1200/1200 [==============================] - 0s 41us/step - loss: -253.0807 - val_loss: -251.9808
    Epoch 134/200
    1200/1200 [==============================] - 0s 43us/step - loss: -253.0153 - val_loss: -252.5076
    Epoch 135/200
    1200/1200 [==============================] - 0s 40us/step - loss: -253.9682 - val_loss: -253.9988
    Epoch 136/200
    1200/1200 [==============================] - 0s 46us/step - loss: -253.2651 - val_loss: -252.8692
    Epoch 137/200
    1200/1200 [==============================] - 0s 42us/step - loss: -253.3367 - val_loss: -250.8972
    Epoch 138/200
    1200/1200 [==============================] - 0s 42us/step - loss: -254.4583 - val_loss: -255.2847
    Epoch 139/200
    1200/1200 [==============================] - 0s 38us/step - loss: -254.5133 - val_loss: -255.7842
    Epoch 140/200
    1200/1200 [==============================] - 0s 41us/step - loss: -253.1222 - val_loss: -255.5163
    Epoch 141/200
    1200/1200 [==============================] - 0s 39us/step - loss: -254.2175 - val_loss: -256.0805
    Epoch 142/200
    1200/1200 [==============================] - 0s 45us/step - loss: -254.9140 - val_loss: -253.1065
    Epoch 143/200
    1200/1200 [==============================] - 0s 41us/step - loss: -254.6043 - val_loss: -255.1811
    Epoch 144/200
    1200/1200 [==============================] - 0s 39us/step - loss: -255.2690 - val_loss: -255.4678
    Epoch 145/200
    1200/1200 [==============================] - 0s 40us/step - loss: -255.6712 - val_loss: -253.1640
    Epoch 146/200
    1200/1200 [==============================] - 0s 47us/step - loss: -255.0432 - val_loss: -256.3587
    Epoch 147/200
    1200/1200 [==============================] - 0s 42us/step - loss: -255.9580 - val_loss: -257.6453
    Epoch 148/200
    1200/1200 [==============================] - 0s 48us/step - loss: -256.6824 - val_loss: -256.7156
    Epoch 149/200
    1200/1200 [==============================] - 0s 40us/step - loss: -255.9437 - val_loss: -256.4921
    Epoch 150/200
    1200/1200 [==============================] - 0s 40us/step - loss: -256.3107 - val_loss: -257.5755
    Epoch 151/200
    1200/1200 [==============================] - 0s 37us/step - loss: -255.9291 - val_loss: -257.0658
    Epoch 152/200
    1200/1200 [==============================] - 0s 40us/step - loss: -257.1780 - val_loss: -257.6051
    Epoch 153/200
    1200/1200 [==============================] - 0s 39us/step - loss: -257.3651 - val_loss: -257.7213
    Epoch 154/200
    1200/1200 [==============================] - 0s 43us/step - loss: -258.1064 - val_loss: -256.3364
    Epoch 155/200
    1200/1200 [==============================] - 0s 41us/step - loss: -256.5212 - val_loss: -257.9618
    Epoch 156/200
    1200/1200 [==============================] - 0s 41us/step - loss: -257.5103 - val_loss: -259.6528
    Epoch 157/200
    1200/1200 [==============================] - 0s 53us/step - loss: -257.4006 - val_loss: -258.5351
    Epoch 158/200
    1200/1200 [==============================] - 0s 65us/step - loss: -257.8289 - val_loss: -258.3871
    Epoch 159/200
    1200/1200 [==============================] - 0s 48us/step - loss: -258.5210 - val_loss: -259.1702
    Epoch 160/200
    1200/1200 [==============================] - 0s 43us/step - loss: -257.1770 - val_loss: -257.1066
    Epoch 161/200
    1200/1200 [==============================] - 0s 42us/step - loss: -256.9588 - val_loss: -260.0514
    Epoch 162/200
    1200/1200 [==============================] - 0s 40us/step - loss: -258.5932 - val_loss: -253.6860
    Epoch 163/200
    1200/1200 [==============================] - 0s 42us/step - loss: -258.2598 - val_loss: -258.7945
    Epoch 164/200
    1200/1200 [==============================] - 0s 51us/step - loss: -258.0663 - val_loss: -258.8854
    Epoch 165/200
    1200/1200 [==============================] - 0s 42us/step - loss: -258.4618 - val_loss: -259.0408
    Epoch 166/200
    1200/1200 [==============================] - 0s 44us/step - loss: -258.7331 - val_loss: -258.9028
    Epoch 167/200
    1200/1200 [==============================] - 0s 41us/step - loss: -258.2827 - val_loss: -260.8663
    Epoch 168/200
    1200/1200 [==============================] - 0s 41us/step - loss: -258.9435 - val_loss: -259.7676
    Epoch 169/200
    1200/1200 [==============================] - 0s 41us/step - loss: -259.5698 - val_loss: -258.3349
    Epoch 170/200
    1200/1200 [==============================] - 0s 43us/step - loss: -258.4024 - val_loss: -260.0891
    Epoch 171/200
    1200/1200 [==============================] - 0s 42us/step - loss: -259.4961 - val_loss: -257.5442
    Epoch 172/200
    1200/1200 [==============================] - 0s 50us/step - loss: -259.4584 - val_loss: -260.9311
    Epoch 173/200
    1200/1200 [==============================] - 0s 65us/step - loss: -260.0174 - val_loss: -260.1952
    Epoch 174/200
    1200/1200 [==============================] - 0s 66us/step - loss: -259.9890 - val_loss: -260.7972
    Epoch 175/200
    1200/1200 [==============================] - 0s 42us/step - loss: -258.8522 - val_loss: -260.9875
    Epoch 176/200
    1200/1200 [==============================] - 0s 45us/step - loss: -260.6150 - val_loss: -260.8354
    Epoch 177/200
    1200/1200 [==============================] - 0s 43us/step - loss: -260.0136 - val_loss: -261.3633
    Epoch 178/200
    1200/1200 [==============================] - 0s 45us/step - loss: -259.9702 - val_loss: -262.4725
    Epoch 179/200
    1200/1200 [==============================] - 0s 42us/step - loss: -260.5567 - val_loss: -261.3696
    Epoch 180/200
    1200/1200 [==============================] - 0s 42us/step - loss: -260.7674 - val_loss: -262.6675
    Epoch 181/200
    1200/1200 [==============================] - 0s 37us/step - loss: -260.4077 - val_loss: -262.8889
    Epoch 182/200
    1200/1200 [==============================] - 0s 42us/step - loss: -260.9360 - val_loss: -261.5552
    Epoch 183/200
    1200/1200 [==============================] - 0s 40us/step - loss: -262.0768 - val_loss: -263.3439
    Epoch 184/200
    1200/1200 [==============================] - 0s 41us/step - loss: -260.9784 - val_loss: -261.0277
    Epoch 185/200
    1200/1200 [==============================] - 0s 42us/step - loss: -261.4523 - val_loss: -259.2585
    Epoch 186/200
    1200/1200 [==============================] - 0s 40us/step - loss: -261.6441 - val_loss: -262.5110
    Epoch 187/200
    1200/1200 [==============================] - 0s 41us/step - loss: -261.3417 - val_loss: -262.2525
    Epoch 188/200
    1200/1200 [==============================] - 0s 42us/step - loss: -261.8420 - val_loss: -263.2095
    Epoch 189/200
    1200/1200 [==============================] - 0s 42us/step - loss: -261.9902 - val_loss: -263.1647
    Epoch 190/200
    1200/1200 [==============================] - 0s 45us/step - loss: -261.7678 - val_loss: -260.9618
    Epoch 191/200
    1200/1200 [==============================] - 0s 42us/step - loss: -261.9328 - val_loss: -262.9184
    Epoch 192/200
    1200/1200 [==============================] - 0s 44us/step - loss: -262.0058 - val_loss: -264.4958
    Epoch 193/200
    1200/1200 [==============================] - 0s 58us/step - loss: -262.0998 - val_loss: -259.6314
    Epoch 194/200
    1200/1200 [==============================] - 0s 70us/step - loss: -262.2454 - val_loss: -262.4057
    Epoch 195/200
    1200/1200 [==============================] - 0s 47us/step - loss: -262.0322 - val_loss: -263.2751
    Epoch 196/200
    1200/1200 [==============================] - 0s 41us/step - loss: -262.6861 - val_loss: -265.0775
    Epoch 197/200
    1200/1200 [==============================] - 0s 38us/step - loss: -263.2413 - val_loss: -264.7730
    Epoch 198/200
    1200/1200 [==============================] - 0s 42us/step - loss: -262.4230 - val_loss: -260.6247
    Epoch 199/200
    1200/1200 [==============================] - 0s 41us/step - loss: -261.8484 - val_loss: -265.1191
    Epoch 200/200
    1200/1200 [==============================] - 0s 42us/step - loss: -263.5959 - val_loss: -263.4030
    




    <keras.callbacks.History at 0x1620b37d4e0>




```python
encoder = Model(x, z_mean)
```


```python
ae_new_features=encoder.predict(ae_standard_np)
```


```python
#gbdt
classifier=GradientBoostingClassifier()
scores = cross_val_score(classifier,ae_new_features[:891], labels, scoring='f1', cv = 5)
np.mean(scores),np.std(scores)
```




    (0.6989266662463456, 0.03092964026276171)




```python
plt.scatter(ae_new_features[:891][:, 0], ae_new_features[:891][:, 1],marker='o',c=labels)
plt.show()
```


![png](./md_resource/output_194_0.png)


添加白噪声数据进行正则化后，更容易将数据分散开

##### 特征增强总结
特征增强很关键，是后续操作的基础，其实最有用的在于造出make sense的特征，从之前的操作来看我们造了几个make sense的特征就立即把模型从0.77+提升到了0.78+，而通过后续一系列的复杂特征变换（聚类、交互特征、特征选择...）才从0.78+提升到0.79+，接下来我们在features_select_top_50_df基础上尝试一些数据增强的方式；  

#### 4.2 数据增强
提供更多数据给模型训练，可从两方面来考虑：  

（1）利用其余的未标记数据进行无监督学习，在我们的标记数据进行监督学习（半监督学习），比如nlp任务中收集海量的文本数据训练embedding，然后再在其他nlp任务上做fine tuning；  
（2）在当前数据的基础上造出相似的数据，比如nlp任务中删除某一个词、替换同义词...，cv任务中缩放、旋转、翻转图片、gan...

##### 4.2.1 数据增强-半监督学习
这里没有多余的feature数据，我们假设test部分就是多出来的部分；  

在pca上做对比...


```python
#增强前
standard_df=StandardScaler().fit_transform(features_select_top_50_df[:891])
X_pca=PCA(n_components=20).fit_transform(standard_df)
#gbdt
classifier=GradientBoostingClassifier()
scores = cross_val_score(classifier,X_pca, labels, scoring='f1', cv = 5)
np.mean(scores),np.std(scores)
```




    (0.7791925787728027, 0.03804481662137799)




```python
#增强后
standard_df=StandardScaler().fit_transform(features_select_top_50_df)
X_pca=PCA(n_components=20).fit_transform(standard_df)
#gbdt
classifier=GradientBoostingClassifier()
scores = cross_val_score(classifier,X_pca[:891], labels, scoring='f1', cv = 5)
np.mean(scores),np.std(scores)
```




    (0.7915154050620196, 0.03498963750537092)



在ae上对比...


```python
#增强前
#定义网络结构
epochs=200
batch_size=128
input_dim=50

input_layer=Input(shape=(input_dim,))
encode_layer=Dense(20,activation='relu',name='encoder')(input_layer)
decode_layer=Dense(input_dim,activation='tanh')(encode_layer)

model=Model(inputs=input_layer,outputs=decode_layer)
#获取encode_layer层的输出
encode_layer_model = Model(inputs=model.input,outputs=model.get_layer('encoder').output)
model.compile('adam',loss='mse')

ae_standard_np=StandardScaler().fit_transform(features_select_top_50_df[:891])
X_train=ae_standard_np[:750]
X_eval=ae_standard_np[750:]

#训练模型
model.fit(X_train,X_train,batch_size=batch_size,epochs=epochs,validation_data=[X_eval,X_eval])
ae_new_features=encode_layer_model.predict(ae_standard_np)
#gbdt
classifier=GradientBoostingClassifier()
scores = cross_val_score(classifier,ae_new_features, labels, scoring='f1', cv = 5)
np.mean(scores),np.std(scores)
```

    Train on 750 samples, validate on 141 samples
    Epoch 1/200
    750/750 [==============================] - 0s 375us/step - loss: 1.1590 - val_loss: 1.1222
    Epoch 2/200
    750/750 [==============================] - 0s 44us/step - loss: 1.1042 - val_loss: 1.0713
    Epoch 3/200
    750/750 [==============================] - 0s 43us/step - loss: 1.0534 - val_loss: 1.0264
    Epoch 4/200
    750/750 [==============================] - 0s 40us/step - loss: 1.0071 - val_loss: 0.9862
    Epoch 5/200
    750/750 [==============================] - 0s 31us/step - loss: 0.9651 - val_loss: 0.9490
    Epoch 6/200
    750/750 [==============================] - 0s 31us/step - loss: 0.9265 - val_loss: 0.9132
    Epoch 7/200
    750/750 [==============================] - 0s 32us/step - loss: 0.8904 - val_loss: 0.8790
    Epoch 8/200
    750/750 [==============================] - 0s 31us/step - loss: 0.8560 - val_loss: 0.8464
    Epoch 9/200
    750/750 [==============================] - 0s 33us/step - loss: 0.8230 - val_loss: 0.8151
    Epoch 10/200
    750/750 [==============================] - 0s 32us/step - loss: 0.7910 - val_loss: 0.7854
    Epoch 11/200
    750/750 [==============================] - 0s 32us/step - loss: 0.7612 - val_loss: 0.7566
    Epoch 12/200
    750/750 [==============================] - 0s 31us/step - loss: 0.7317 - val_loss: 0.7296
    Epoch 13/200
    750/750 [==============================] - 0s 32us/step - loss: 0.7036 - val_loss: 0.7039
    Epoch 14/200
    750/750 [==============================] - 0s 33us/step - loss: 0.6766 - val_loss: 0.6794
    Epoch 15/200
    750/750 [==============================] - 0s 32us/step - loss: 0.6513 - val_loss: 0.6564
    Epoch 16/200
    750/750 [==============================] - 0s 32us/step - loss: 0.6281 - val_loss: 0.6357
    Epoch 17/200
    750/750 [==============================] - 0s 33us/step - loss: 0.6073 - val_loss: 0.6173
    Epoch 18/200
    750/750 [==============================] - 0s 33us/step - loss: 0.5889 - val_loss: 0.6015
    Epoch 19/200
    750/750 [==============================] - 0s 32us/step - loss: 0.5728 - val_loss: 0.5876
    Epoch 20/200
    750/750 [==============================] - 0s 31us/step - loss: 0.5585 - val_loss: 0.5753
    Epoch 21/200
    750/750 [==============================] - 0s 32us/step - loss: 0.5460 - val_loss: 0.5642
    Epoch 22/200
    750/750 [==============================] - 0s 31us/step - loss: 0.5350 - val_loss: 0.5545
    Epoch 23/200
    750/750 [==============================] - 0s 31us/step - loss: 0.5252 - val_loss: 0.5456
    Epoch 24/200
    750/750 [==============================] - 0s 35us/step - loss: 0.5169 - val_loss: 0.5375
    Epoch 25/200
    750/750 [==============================] - 0s 35us/step - loss: 0.5093 - val_loss: 0.5302
    Epoch 26/200
    750/750 [==============================] - 0s 35us/step - loss: 0.5027 - val_loss: 0.5235
    Epoch 27/200
    750/750 [==============================] - 0s 32us/step - loss: 0.4966 - val_loss: 0.5174
    Epoch 28/200
    750/750 [==============================] - 0s 32us/step - loss: 0.4910 - val_loss: 0.5118
    Epoch 29/200
    750/750 [==============================] - 0s 41us/step - loss: 0.4859 - val_loss: 0.5066
    Epoch 30/200
    750/750 [==============================] - 0s 39us/step - loss: 0.4811 - val_loss: 0.5018
    Epoch 31/200
    750/750 [==============================] - 0s 29us/step - loss: 0.4766 - val_loss: 0.4972
    Epoch 32/200
    750/750 [==============================] - 0s 33us/step - loss: 0.4724 - val_loss: 0.4930
    Epoch 33/200
    750/750 [==============================] - 0s 35us/step - loss: 0.4686 - val_loss: 0.4890
    Epoch 34/200
    750/750 [==============================] - 0s 35us/step - loss: 0.4648 - val_loss: 0.4852
    Epoch 35/200
    750/750 [==============================] - 0s 31us/step - loss: 0.4613 - val_loss: 0.4816
    Epoch 36/200
    750/750 [==============================] - 0s 35us/step - loss: 0.4579 - val_loss: 0.4783
    Epoch 37/200
    750/750 [==============================] - 0s 36us/step - loss: 0.4547 - val_loss: 0.4751
    Epoch 38/200
    750/750 [==============================] - 0s 35us/step - loss: 0.4516 - val_loss: 0.4721
    Epoch 39/200
    750/750 [==============================] - 0s 29us/step - loss: 0.4487 - val_loss: 0.4694
    Epoch 40/200
    750/750 [==============================] - 0s 33us/step - loss: 0.4460 - val_loss: 0.4666
    Epoch 41/200
    750/750 [==============================] - 0s 37us/step - loss: 0.4434 - val_loss: 0.4641
    Epoch 42/200
    750/750 [==============================] - 0s 49us/step - loss: 0.4410 - val_loss: 0.4618
    Epoch 43/200
    750/750 [==============================] - 0s 43us/step - loss: 0.4386 - val_loss: 0.4596
    Epoch 44/200
    750/750 [==============================] - 0s 47us/step - loss: 0.4364 - val_loss: 0.4575
    Epoch 45/200
    750/750 [==============================] - 0s 31us/step - loss: 0.4343 - val_loss: 0.4556
    Epoch 46/200
    750/750 [==============================] - 0s 32us/step - loss: 0.4322 - val_loss: 0.4536
    Epoch 47/200
    750/750 [==============================] - 0s 31us/step - loss: 0.4303 - val_loss: 0.4518
    Epoch 48/200
    750/750 [==============================] - 0s 31us/step - loss: 0.4285 - val_loss: 0.4501
    Epoch 49/200
    750/750 [==============================] - 0s 31us/step - loss: 0.4267 - val_loss: 0.4484
    Epoch 50/200
    750/750 [==============================] - 0s 29us/step - loss: 0.4250 - val_loss: 0.4469
    Epoch 51/200
    750/750 [==============================] - 0s 29us/step - loss: 0.4234 - val_loss: 0.4454
    Epoch 52/200
    750/750 [==============================] - 0s 29us/step - loss: 0.4219 - val_loss: 0.4439
    Epoch 53/200
    750/750 [==============================] - 0s 29us/step - loss: 0.4204 - val_loss: 0.4426
    Epoch 54/200
    750/750 [==============================] - 0s 33us/step - loss: 0.4190 - val_loss: 0.4414
    Epoch 55/200
    750/750 [==============================] - 0s 32us/step - loss: 0.4176 - val_loss: 0.4400
    Epoch 56/200
    750/750 [==============================] - 0s 28us/step - loss: 0.4163 - val_loss: 0.4388
    Epoch 57/200
    750/750 [==============================] - 0s 29us/step - loss: 0.4150 - val_loss: 0.4377
    Epoch 58/200
    750/750 [==============================] - 0s 25us/step - loss: 0.4138 - val_loss: 0.4366
    Epoch 59/200
    750/750 [==============================] - 0s 29us/step - loss: 0.4126 - val_loss: 0.4355
    Epoch 60/200
    750/750 [==============================] - 0s 31us/step - loss: 0.4115 - val_loss: 0.4344
    Epoch 61/200
    750/750 [==============================] - 0s 29us/step - loss: 0.4104 - val_loss: 0.4333
    Epoch 62/200
    750/750 [==============================] - 0s 29us/step - loss: 0.4094 - val_loss: 0.4323
    Epoch 63/200
    750/750 [==============================] - 0s 31us/step - loss: 0.4084 - val_loss: 0.4314
    Epoch 64/200
    750/750 [==============================] - 0s 29us/step - loss: 0.4074 - val_loss: 0.4304
    Epoch 65/200
    750/750 [==============================] - 0s 28us/step - loss: 0.4064 - val_loss: 0.4295
    Epoch 66/200
    750/750 [==============================] - 0s 28us/step - loss: 0.4055 - val_loss: 0.4285
    Epoch 67/200
    750/750 [==============================] - 0s 31us/step - loss: 0.4046 - val_loss: 0.4277
    Epoch 68/200
    750/750 [==============================] - 0s 29us/step - loss: 0.4038 - val_loss: 0.4268
    Epoch 69/200
    750/750 [==============================] - 0s 32us/step - loss: 0.4030 - val_loss: 0.4259
    Epoch 70/200
    750/750 [==============================] - 0s 35us/step - loss: 0.4021 - val_loss: 0.4251
    Epoch 71/200
    750/750 [==============================] - 0s 31us/step - loss: 0.4013 - val_loss: 0.4243
    Epoch 72/200
    750/750 [==============================] - 0s 27us/step - loss: 0.4006 - val_loss: 0.4236
    Epoch 73/200
    750/750 [==============================] - 0s 31us/step - loss: 0.3998 - val_loss: 0.4227
    Epoch 74/200
    750/750 [==============================] - 0s 33us/step - loss: 0.3991 - val_loss: 0.4219
    Epoch 75/200
    750/750 [==============================] - 0s 36us/step - loss: 0.3984 - val_loss: 0.4211
    Epoch 76/200
    750/750 [==============================] - 0s 43us/step - loss: 0.3977 - val_loss: 0.4205
    Epoch 77/200
    750/750 [==============================] - 0s 41us/step - loss: 0.3970 - val_loss: 0.4198
    Epoch 78/200
    750/750 [==============================] - 0s 41us/step - loss: 0.3964 - val_loss: 0.4191
    Epoch 79/200
    750/750 [==============================] - 0s 44us/step - loss: 0.3957 - val_loss: 0.4183
    Epoch 80/200
    750/750 [==============================] - 0s 32us/step - loss: 0.3950 - val_loss: 0.4177
    Epoch 81/200
    750/750 [==============================] - 0s 33us/step - loss: 0.3945 - val_loss: 0.4170
    Epoch 82/200
    750/750 [==============================] - 0s 37us/step - loss: 0.3938 - val_loss: 0.4163
    Epoch 83/200
    750/750 [==============================] - 0s 33us/step - loss: 0.3933 - val_loss: 0.4157
    Epoch 84/200
    750/750 [==============================] - 0s 28us/step - loss: 0.3927 - val_loss: 0.4151
    Epoch 85/200
    750/750 [==============================] - 0s 33us/step - loss: 0.3921 - val_loss: 0.4144
    Epoch 86/200
    750/750 [==============================] - 0s 32us/step - loss: 0.3916 - val_loss: 0.4139
    Epoch 87/200
    750/750 [==============================] - 0s 36us/step - loss: 0.3910 - val_loss: 0.4133
    Epoch 88/200
    750/750 [==============================] - 0s 35us/step - loss: 0.3905 - val_loss: 0.4126
    Epoch 89/200
    750/750 [==============================] - 0s 33us/step - loss: 0.3900 - val_loss: 0.4121
    Epoch 90/200
    750/750 [==============================] - 0s 32us/step - loss: 0.3895 - val_loss: 0.4115
    Epoch 91/200
    750/750 [==============================] - 0s 28us/step - loss: 0.3890 - val_loss: 0.4111
    Epoch 92/200
    750/750 [==============================] - 0s 33us/step - loss: 0.3886 - val_loss: 0.4104
    Epoch 93/200
    750/750 [==============================] - 0s 41us/step - loss: 0.3881 - val_loss: 0.4100
    Epoch 94/200
    750/750 [==============================] - ETA: 0s - loss: 0.392 - 0s 41us/step - loss: 0.3876 - val_loss: 0.4095
    Epoch 95/200
    750/750 [==============================] - 0s 47us/step - loss: 0.3872 - val_loss: 0.4090
    Epoch 96/200
    750/750 [==============================] - 0s 36us/step - loss: 0.3868 - val_loss: 0.4084
    Epoch 97/200
    750/750 [==============================] - 0s 39us/step - loss: 0.3863 - val_loss: 0.4080
    Epoch 98/200
    750/750 [==============================] - 0s 36us/step - loss: 0.3859 - val_loss: 0.4076
    Epoch 99/200
    750/750 [==============================] - 0s 37us/step - loss: 0.3855 - val_loss: 0.4071
    Epoch 100/200
    750/750 [==============================] - 0s 33us/step - loss: 0.3851 - val_loss: 0.4067
    Epoch 101/200
    750/750 [==============================] - 0s 33us/step - loss: 0.3847 - val_loss: 0.4062
    Epoch 102/200
    750/750 [==============================] - 0s 40us/step - loss: 0.3843 - val_loss: 0.4058
    Epoch 103/200
    750/750 [==============================] - 0s 41us/step - loss: 0.3840 - val_loss: 0.4054
    Epoch 104/200
    750/750 [==============================] - 0s 39us/step - loss: 0.3836 - val_loss: 0.4049
    Epoch 105/200
    750/750 [==============================] - 0s 41us/step - loss: 0.3833 - val_loss: 0.4045
    Epoch 106/200
    750/750 [==============================] - 0s 43us/step - loss: 0.3829 - val_loss: 0.4043
    Epoch 107/200
    750/750 [==============================] - 0s 37us/step - loss: 0.3825 - val_loss: 0.4039
    Epoch 108/200
    750/750 [==============================] - 0s 33us/step - loss: 0.3822 - val_loss: 0.4035
    Epoch 109/200
    750/750 [==============================] - 0s 31us/step - loss: 0.3819 - val_loss: 0.4030
    Epoch 110/200
    750/750 [==============================] - 0s 31us/step - loss: 0.3815 - val_loss: 0.4027
    Epoch 111/200
    750/750 [==============================] - 0s 33us/step - loss: 0.3812 - val_loss: 0.4024
    Epoch 112/200
    750/750 [==============================] - 0s 32us/step - loss: 0.3809 - val_loss: 0.4021
    Epoch 113/200
    750/750 [==============================] - 0s 33us/step - loss: 0.3806 - val_loss: 0.4017
    Epoch 114/200
    750/750 [==============================] - 0s 33us/step - loss: 0.3803 - val_loss: 0.4013
    Epoch 115/200
    750/750 [==============================] - 0s 35us/step - loss: 0.3799 - val_loss: 0.4010
    Epoch 116/200
    750/750 [==============================] - 0s 44us/step - loss: 0.3797 - val_loss: 0.4007
    Epoch 117/200
    750/750 [==============================] - 0s 43us/step - loss: 0.3793 - val_loss: 0.4004
    Epoch 118/200
    750/750 [==============================] - 0s 44us/step - loss: 0.3790 - val_loss: 0.4000
    Epoch 119/200
    750/750 [==============================] - 0s 45us/step - loss: 0.3788 - val_loss: 0.3997
    Epoch 120/200
    750/750 [==============================] - 0s 39us/step - loss: 0.3785 - val_loss: 0.3994
    Epoch 121/200
    750/750 [==============================] - 0s 29us/step - loss: 0.3782 - val_loss: 0.3992
    Epoch 122/200
    750/750 [==============================] - 0s 35us/step - loss: 0.3779 - val_loss: 0.3988
    Epoch 123/200
    750/750 [==============================] - 0s 31us/step - loss: 0.3776 - val_loss: 0.3985
    Epoch 124/200
    750/750 [==============================] - 0s 35us/step - loss: 0.3773 - val_loss: 0.3983
    Epoch 125/200
    750/750 [==============================] - 0s 44us/step - loss: 0.3771 - val_loss: 0.3981
    Epoch 126/200
    750/750 [==============================] - 0s 44us/step - loss: 0.3768 - val_loss: 0.3978
    Epoch 127/200
    750/750 [==============================] - 0s 44us/step - loss: 0.3766 - val_loss: 0.3976
    Epoch 128/200
    750/750 [==============================] - 0s 43us/step - loss: 0.3763 - val_loss: 0.3973
    Epoch 129/200
    750/750 [==============================] - 0s 45us/step - loss: 0.3760 - val_loss: 0.3970
    Epoch 130/200
    750/750 [==============================] - 0s 35us/step - loss: 0.3758 - val_loss: 0.3968
    Epoch 131/200
    750/750 [==============================] - 0s 31us/step - loss: 0.3755 - val_loss: 0.3965
    Epoch 132/200
    750/750 [==============================] - 0s 33us/step - loss: 0.3753 - val_loss: 0.3962
    Epoch 133/200
    750/750 [==============================] - 0s 36us/step - loss: 0.3751 - val_loss: 0.3960
    Epoch 134/200
    750/750 [==============================] - 0s 39us/step - loss: 0.3748 - val_loss: 0.3958
    Epoch 135/200
    750/750 [==============================] - 0s 37us/step - loss: 0.3746 - val_loss: 0.3956
    Epoch 136/200
    750/750 [==============================] - 0s 47us/step - loss: 0.3744 - val_loss: 0.3954
    Epoch 137/200
    750/750 [==============================] - 0s 45us/step - loss: 0.3741 - val_loss: 0.3951
    Epoch 138/200
    750/750 [==============================] - 0s 35us/step - loss: 0.3739 - val_loss: 0.3949
    Epoch 139/200
    750/750 [==============================] - 0s 44us/step - loss: 0.3737 - val_loss: 0.3946
    Epoch 140/200
    750/750 [==============================] - 0s 33us/step - loss: 0.3734 - val_loss: 0.3944
    Epoch 141/200
    750/750 [==============================] - ETA: 0s - loss: 0.444 - 0s 40us/step - loss: 0.3732 - val_loss: 0.3942
    Epoch 142/200
    750/750 [==============================] - 0s 37us/step - loss: 0.3730 - val_loss: 0.3940
    Epoch 143/200
    750/750 [==============================] - 0s 44us/step - loss: 0.3728 - val_loss: 0.3939
    Epoch 144/200
    750/750 [==============================] - 0s 41us/step - loss: 0.3726 - val_loss: 0.3936
    Epoch 145/200
    750/750 [==============================] - 0s 40us/step - loss: 0.3724 - val_loss: 0.3934
    Epoch 146/200
    750/750 [==============================] - 0s 36us/step - loss: 0.3722 - val_loss: 0.3931
    Epoch 147/200
    750/750 [==============================] - 0s 36us/step - loss: 0.3720 - val_loss: 0.3930
    Epoch 148/200
    750/750 [==============================] - 0s 33us/step - loss: 0.3718 - val_loss: 0.3928
    Epoch 149/200
    750/750 [==============================] - 0s 40us/step - loss: 0.3716 - val_loss: 0.3926
    Epoch 150/200
    750/750 [==============================] - 0s 44us/step - loss: 0.3714 - val_loss: 0.3924
    Epoch 151/200
    750/750 [==============================] - 0s 45us/step - loss: 0.3712 - val_loss: 0.3922
    Epoch 152/200
    750/750 [==============================] - 0s 39us/step - loss: 0.3710 - val_loss: 0.3920
    Epoch 153/200
    750/750 [==============================] - 0s 33us/step - loss: 0.3708 - val_loss: 0.3919
    Epoch 154/200
    750/750 [==============================] - 0s 41us/step - loss: 0.3706 - val_loss: 0.3916
    Epoch 155/200
    750/750 [==============================] - 0s 40us/step - loss: 0.3704 - val_loss: 0.3915
    Epoch 156/200
    750/750 [==============================] - 0s 43us/step - loss: 0.3703 - val_loss: 0.3913
    Epoch 157/200
    750/750 [==============================] - 0s 44us/step - loss: 0.3701 - val_loss: 0.3912
    Epoch 158/200
    750/750 [==============================] - 0s 40us/step - loss: 0.3699 - val_loss: 0.3910
    Epoch 159/200
    750/750 [==============================] - 0s 43us/step - loss: 0.3697 - val_loss: 0.3908
    Epoch 160/200
    750/750 [==============================] - 0s 36us/step - loss: 0.3696 - val_loss: 0.3906
    Epoch 161/200
    750/750 [==============================] - 0s 31us/step - loss: 0.3694 - val_loss: 0.3905
    Epoch 162/200
    750/750 [==============================] - 0s 32us/step - loss: 0.3692 - val_loss: 0.3904
    Epoch 163/200
    750/750 [==============================] - 0s 33us/step - loss: 0.3691 - val_loss: 0.3902
    Epoch 164/200
    750/750 [==============================] - 0s 28us/step - loss: 0.3689 - val_loss: 0.3900
    Epoch 165/200
    750/750 [==============================] - 0s 28us/step - loss: 0.3688 - val_loss: 0.3898
    Epoch 166/200
    750/750 [==============================] - 0s 33us/step - loss: 0.3686 - val_loss: 0.3897
    Epoch 167/200
    750/750 [==============================] - 0s 35us/step - loss: 0.3684 - val_loss: 0.3896
    Epoch 168/200
    750/750 [==============================] - 0s 37us/step - loss: 0.3683 - val_loss: 0.3894
    Epoch 169/200
    750/750 [==============================] - 0s 40us/step - loss: 0.3681 - val_loss: 0.3893
    Epoch 170/200
    750/750 [==============================] - 0s 40us/step - loss: 0.3680 - val_loss: 0.3891
    Epoch 171/200
    750/750 [==============================] - 0s 44us/step - loss: 0.3678 - val_loss: 0.3890
    Epoch 172/200
    750/750 [==============================] - 0s 43us/step - loss: 0.3677 - val_loss: 0.3888
    Epoch 173/200
    750/750 [==============================] - 0s 36us/step - loss: 0.3675 - val_loss: 0.3887
    Epoch 174/200
    750/750 [==============================] - 0s 36us/step - loss: 0.3674 - val_loss: 0.3886
    Epoch 175/200
    750/750 [==============================] - 0s 32us/step - loss: 0.3672 - val_loss: 0.3885
    Epoch 176/200
    750/750 [==============================] - 0s 27us/step - loss: 0.3671 - val_loss: 0.3883
    Epoch 177/200
    750/750 [==============================] - 0s 28us/step - loss: 0.3669 - val_loss: 0.3882
    Epoch 178/200
    750/750 [==============================] - 0s 31us/step - loss: 0.3668 - val_loss: 0.3881
    Epoch 179/200
    750/750 [==============================] - 0s 25us/step - loss: 0.3667 - val_loss: 0.3879
    Epoch 180/200
    750/750 [==============================] - 0s 33us/step - loss: 0.3665 - val_loss: 0.3878
    Epoch 181/200
    750/750 [==============================] - 0s 35us/step - loss: 0.3664 - val_loss: 0.3877
    Epoch 182/200
    750/750 [==============================] - 0s 29us/step - loss: 0.3663 - val_loss: 0.3875
    Epoch 183/200
    750/750 [==============================] - 0s 31us/step - loss: 0.3661 - val_loss: 0.3874
    Epoch 184/200
    750/750 [==============================] - 0s 28us/step - loss: 0.3660 - val_loss: 0.3873
    Epoch 185/200
    750/750 [==============================] - 0s 29us/step - loss: 0.3659 - val_loss: 0.3872
    Epoch 186/200
    750/750 [==============================] - 0s 28us/step - loss: 0.3658 - val_loss: 0.3872
    Epoch 187/200
    750/750 [==============================] - 0s 29us/step - loss: 0.3656 - val_loss: 0.3870
    Epoch 188/200
    750/750 [==============================] - 0s 33us/step - loss: 0.3655 - val_loss: 0.3868
    Epoch 189/200
    750/750 [==============================] - 0s 29us/step - loss: 0.3654 - val_loss: 0.3868
    Epoch 190/200
    750/750 [==============================] - 0s 28us/step - loss: 0.3653 - val_loss: 0.3867
    Epoch 191/200
    750/750 [==============================] - 0s 31us/step - loss: 0.3652 - val_loss: 0.3865
    Epoch 192/200
    750/750 [==============================] - 0s 33us/step - loss: 0.3650 - val_loss: 0.3864
    Epoch 193/200
    750/750 [==============================] - 0s 32us/step - loss: 0.3649 - val_loss: 0.3863
    Epoch 194/200
    750/750 [==============================] - 0s 31us/step - loss: 0.3648 - val_loss: 0.3862
    Epoch 195/200
    750/750 [==============================] - 0s 29us/step - loss: 0.3647 - val_loss: 0.3861
    Epoch 196/200
    750/750 [==============================] - 0s 29us/step - loss: 0.3646 - val_loss: 0.3859
    Epoch 197/200
    750/750 [==============================] - 0s 31us/step - loss: 0.3645 - val_loss: 0.3859
    Epoch 198/200
    750/750 [==============================] - 0s 31us/step - loss: 0.3644 - val_loss: 0.3858
    Epoch 199/200
    750/750 [==============================] - 0s 28us/step - loss: 0.3642 - val_loss: 0.3857
    Epoch 200/200
    750/750 [==============================] - 0s 29us/step - loss: 0.3641 - val_loss: 0.3855
    




    (0.7752863876311468, 0.04550175059993495)




```python
#增强后
epochs=200
batch_size=128
input_dim=50

input_layer=Input(shape=(input_dim,))
encode_layer=Dense(20,activation='relu',name='encoder')(input_layer)
decode_layer=Dense(input_dim,activation='tanh')(encode_layer)

model=Model(inputs=input_layer,outputs=decode_layer)
#获取encode_layer层的输出
encode_layer_model = Model(inputs=model.input,outputs=model.get_layer('encoder').output)
model.compile('adam',loss='mse')

ae_standard_np=StandardScaler().fit_transform(features_select_top_50_df)
X_train=ae_standard_np[:1200]
X_eval=ae_standard_np[1200:]

#训练模型
model.fit(X_train,X_train,batch_size=batch_size,epochs=epochs,validation_data=[X_eval,X_eval])
ae_new_features=encode_layer_model.predict(ae_standard_np)
#gbdt
classifier=GradientBoostingClassifier()
scores = cross_val_score(classifier,ae_new_features[:891], labels, scoring='f1', cv = 5)
np.mean(scores),np.std(scores)
```

    Train on 1200 samples, validate on 109 samples
    Epoch 1/200
    1200/1200 [==============================] - 0s 274us/step - loss: 1.1714 - val_loss: 1.1603
    Epoch 2/200
    1200/1200 [==============================] - 0s 27us/step - loss: 1.0831 - val_loss: 1.0879
    Epoch 3/200
    1200/1200 [==============================] - 0s 29us/step - loss: 1.0114 - val_loss: 1.0262
    Epoch 4/200
    1200/1200 [==============================] - 0s 35us/step - loss: 0.9489 - val_loss: 0.9699
    Epoch 5/200
    1200/1200 [==============================] - 0s 42us/step - loss: 0.8911 - val_loss: 0.9141
    Epoch 6/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.8366 - val_loss: 0.8592
    Epoch 7/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.7843 - val_loss: 0.8059
    Epoch 8/200
    1200/1200 [==============================] - 0s 33us/step - loss: 0.7343 - val_loss: 0.7563
    Epoch 9/200
    1200/1200 [==============================] - 0s 26us/step - loss: 0.6885 - val_loss: 0.7123
    Epoch 10/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.6495 - val_loss: 0.6757
    Epoch 11/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.6175 - val_loss: 0.6482
    Epoch 12/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.5933 - val_loss: 0.6267
    Epoch 13/200
    1200/1200 [==============================] - 0s 26us/step - loss: 0.5740 - val_loss: 0.6095
    Epoch 14/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.5581 - val_loss: 0.5945
    Epoch 15/200
    1200/1200 [==============================] - 0s 25us/step - loss: 0.5441 - val_loss: 0.5817
    Epoch 16/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.5322 - val_loss: 0.5702
    Epoch 17/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.5212 - val_loss: 0.5597
    Epoch 18/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.5113 - val_loss: 0.5500
    Epoch 19/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.5021 - val_loss: 0.5409
    Epoch 20/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.4937 - val_loss: 0.5327
    Epoch 21/200
    1200/1200 [==============================] - 0s 26us/step - loss: 0.4860 - val_loss: 0.5249
    Epoch 22/200
    1200/1200 [==============================] - 0s 31us/step - loss: 0.4789 - val_loss: 0.5178
    Epoch 23/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.4725 - val_loss: 0.5112
    Epoch 24/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.4665 - val_loss: 0.5052
    Epoch 25/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.4610 - val_loss: 0.4993
    Epoch 26/200
    1200/1200 [==============================] - 0s 34us/step - loss: 0.4561 - val_loss: 0.4938
    Epoch 27/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.4514 - val_loss: 0.4889
    Epoch 28/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.4470 - val_loss: 0.4844
    Epoch 29/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.4431 - val_loss: 0.4799
    Epoch 30/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.4394 - val_loss: 0.4757
    Epoch 31/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.4359 - val_loss: 0.4719
    Epoch 32/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.4327 - val_loss: 0.4680
    Epoch 33/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.4296 - val_loss: 0.4647
    Epoch 34/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.4268 - val_loss: 0.4614
    Epoch 35/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.4242 - val_loss: 0.4584
    Epoch 36/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.4217 - val_loss: 0.4560
    Epoch 37/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.4194 - val_loss: 0.4533
    Epoch 38/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.4173 - val_loss: 0.4511
    Epoch 39/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.4153 - val_loss: 0.4488
    Epoch 40/200
    1200/1200 [==============================] - 0s 31us/step - loss: 0.4134 - val_loss: 0.4467
    Epoch 41/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.4116 - val_loss: 0.4446
    Epoch 42/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.4099 - val_loss: 0.4426
    Epoch 43/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.4083 - val_loss: 0.4408
    Epoch 44/200
    1200/1200 [==============================] - 0s 31us/step - loss: 0.4068 - val_loss: 0.4392
    Epoch 45/200
    1200/1200 [==============================] - 0s 25us/step - loss: 0.4053 - val_loss: 0.4375
    Epoch 46/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.4040 - val_loss: 0.4361
    Epoch 47/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.4027 - val_loss: 0.4347
    Epoch 48/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.4015 - val_loss: 0.4334
    Epoch 49/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.4003 - val_loss: 0.4321
    Epoch 50/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.3992 - val_loss: 0.4310
    Epoch 51/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3982 - val_loss: 0.4295
    Epoch 52/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.3972 - val_loss: 0.4285
    Epoch 53/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3963 - val_loss: 0.4274
    Epoch 54/200
    1200/1200 [==============================] - 0s 26us/step - loss: 0.3954 - val_loss: 0.4266
    Epoch 55/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3945 - val_loss: 0.4254
    Epoch 56/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3937 - val_loss: 0.4242
    Epoch 57/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3929 - val_loss: 0.4232
    Epoch 58/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.3921 - val_loss: 0.4223
    Epoch 59/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3914 - val_loss: 0.4214
    Epoch 60/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3907 - val_loss: 0.4206
    Epoch 61/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.3900 - val_loss: 0.4198
    Epoch 62/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.3893 - val_loss: 0.4191
    Epoch 63/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3887 - val_loss: 0.4181
    Epoch 64/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3880 - val_loss: 0.4173
    Epoch 65/200
    1200/1200 [==============================] - 0s 34us/step - loss: 0.3874 - val_loss: 0.4164
    Epoch 66/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.3868 - val_loss: 0.4157
    Epoch 67/200
    1200/1200 [==============================] - 0s 26us/step - loss: 0.3862 - val_loss: 0.4151
    Epoch 68/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3857 - val_loss: 0.4144
    Epoch 69/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.3852 - val_loss: 0.4136
    Epoch 70/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3846 - val_loss: 0.4131
    Epoch 71/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.3841 - val_loss: 0.4122
    Epoch 72/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.3836 - val_loss: 0.4114
    Epoch 73/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3831 - val_loss: 0.4107
    Epoch 74/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.3826 - val_loss: 0.4101
    Epoch 75/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.3821 - val_loss: 0.4097
    Epoch 76/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.3817 - val_loss: 0.4091
    Epoch 77/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.3812 - val_loss: 0.4085
    Epoch 78/200
    1200/1200 [==============================] - 0s 31us/step - loss: 0.3808 - val_loss: 0.4079
    Epoch 79/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.3804 - val_loss: 0.4073
    Epoch 80/200
    1200/1200 [==============================] - 0s 25us/step - loss: 0.3799 - val_loss: 0.4068
    Epoch 81/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.3795 - val_loss: 0.4064
    Epoch 82/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3791 - val_loss: 0.4059
    Epoch 83/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.3787 - val_loss: 0.4053
    Epoch 84/200
    1200/1200 [==============================] - 0s 31us/step - loss: 0.3784 - val_loss: 0.4048
    Epoch 85/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.3780 - val_loss: 0.4044
    Epoch 86/200
    1200/1200 [==============================] - 0s 32us/step - loss: 0.3776 - val_loss: 0.4038
    Epoch 87/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3773 - val_loss: 0.4033
    Epoch 88/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3769 - val_loss: 0.4031
    Epoch 89/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3766 - val_loss: 0.4026
    Epoch 90/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3763 - val_loss: 0.4022
    Epoch 91/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.3759 - val_loss: 0.4017
    Epoch 92/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.3756 - val_loss: 0.4013
    Epoch 93/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3753 - val_loss: 0.4010
    Epoch 94/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.3750 - val_loss: 0.4006
    Epoch 95/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3747 - val_loss: 0.4003
    Epoch 96/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3744 - val_loss: 0.3999
    Epoch 97/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.3741 - val_loss: 0.3997
    Epoch 98/200
    1200/1200 [==============================] - 0s 26us/step - loss: 0.3738 - val_loss: 0.3993
    Epoch 99/200
    1200/1200 [==============================] - ETA: 0s - loss: 0.360 - 0s 28us/step - loss: 0.3735 - val_loss: 0.3990
    Epoch 100/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3732 - val_loss: 0.3986
    Epoch 101/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3729 - val_loss: 0.3983
    Epoch 102/200
    1200/1200 [==============================] - 0s 26us/step - loss: 0.3726 - val_loss: 0.3980
    Epoch 103/200
    1200/1200 [==============================] - ETA: 0s - loss: 0.346 - 0s 27us/step - loss: 0.3723 - val_loss: 0.3976
    Epoch 104/200
    1200/1200 [==============================] - 0s 26us/step - loss: 0.3720 - val_loss: 0.3973
    Epoch 105/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3717 - val_loss: 0.3970
    Epoch 106/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.3714 - val_loss: 0.3967
    Epoch 107/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3711 - val_loss: 0.3963
    Epoch 108/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.3709 - val_loss: 0.3962
    Epoch 109/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.3706 - val_loss: 0.3958
    Epoch 110/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3703 - val_loss: 0.3955
    Epoch 111/200
    1200/1200 [==============================] - 0s 32us/step - loss: 0.3701 - val_loss: 0.3952
    Epoch 112/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3698 - val_loss: 0.3949
    Epoch 113/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.3696 - val_loss: 0.3947
    Epoch 114/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.3694 - val_loss: 0.3944
    Epoch 115/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3692 - val_loss: 0.3942
    Epoch 116/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.3690 - val_loss: 0.3940
    Epoch 117/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3688 - val_loss: 0.3938
    Epoch 118/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.3686 - val_loss: 0.3935
    Epoch 119/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3685 - val_loss: 0.3933
    Epoch 120/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3683 - val_loss: 0.3932
    Epoch 121/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.3681 - val_loss: 0.3929
    Epoch 122/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3679 - val_loss: 0.3928
    Epoch 123/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3678 - val_loss: 0.3926
    Epoch 124/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3676 - val_loss: 0.3925
    Epoch 125/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.3674 - val_loss: 0.3924
    Epoch 126/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.3673 - val_loss: 0.3921
    Epoch 127/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.3671 - val_loss: 0.3921
    Epoch 128/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.3670 - val_loss: 0.3919
    Epoch 129/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.3668 - val_loss: 0.3917
    Epoch 130/200
    1200/1200 [==============================] - 0s 34us/step - loss: 0.3667 - val_loss: 0.3916
    Epoch 131/200
    1200/1200 [==============================] - 0s 34us/step - loss: 0.3665 - val_loss: 0.3914
    Epoch 132/200
    1200/1200 [==============================] - 0s 39us/step - loss: 0.3664 - val_loss: 0.3913
    Epoch 133/200
    1200/1200 [==============================] - 0s 38us/step - loss: 0.3662 - val_loss: 0.3910
    Epoch 134/200
    1200/1200 [==============================] - 0s 35us/step - loss: 0.3661 - val_loss: 0.3910
    Epoch 135/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.3660 - val_loss: 0.3909
    Epoch 136/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3659 - val_loss: 0.3907
    Epoch 137/200
    1200/1200 [==============================] - 0s 33us/step - loss: 0.3657 - val_loss: 0.3905
    Epoch 138/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.3656 - val_loss: 0.3905
    Epoch 139/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3655 - val_loss: 0.3903
    Epoch 140/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.3653 - val_loss: 0.3903
    Epoch 141/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.3652 - val_loss: 0.3901
    Epoch 142/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3651 - val_loss: 0.3899
    Epoch 143/200
    1200/1200 [==============================] - 0s 26us/step - loss: 0.3650 - val_loss: 0.3899
    Epoch 144/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.3649 - val_loss: 0.3897
    Epoch 145/200
    1200/1200 [==============================] - ETA: 0s - loss: 0.341 - 0s 29us/step - loss: 0.3648 - val_loss: 0.3897
    Epoch 146/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3647 - val_loss: 0.3896
    Epoch 147/200
    1200/1200 [==============================] - 0s 31us/step - loss: 0.3646 - val_loss: 0.3894
    Epoch 148/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.3645 - val_loss: 0.3894
    Epoch 149/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3644 - val_loss: 0.3893
    Epoch 150/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.3642 - val_loss: 0.3892
    Epoch 151/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3641 - val_loss: 0.3891
    Epoch 152/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.3641 - val_loss: 0.3889
    Epoch 153/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.3639 - val_loss: 0.3889
    Epoch 154/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.3639 - val_loss: 0.3888
    Epoch 155/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3638 - val_loss: 0.3888
    Epoch 156/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.3637 - val_loss: 0.3886
    Epoch 157/200
    1200/1200 [==============================] - 0s 26us/step - loss: 0.3636 - val_loss: 0.3886
    Epoch 158/200
    1200/1200 [==============================] - 0s 26us/step - loss: 0.3635 - val_loss: 0.3886
    Epoch 159/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3634 - val_loss: 0.3884
    Epoch 160/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3633 - val_loss: 0.3884
    Epoch 161/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.3632 - val_loss: 0.3883
    Epoch 162/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.3631 - val_loss: 0.3882
    Epoch 163/200
    1200/1200 [==============================] - 0s 24us/step - loss: 0.3630 - val_loss: 0.3881
    Epoch 164/200
    1200/1200 [==============================] - 0s 32us/step - loss: 0.3629 - val_loss: 0.3880
    Epoch 165/200
    1200/1200 [==============================] - 0s 32us/step - loss: 0.3628 - val_loss: 0.3879
    Epoch 166/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3627 - val_loss: 0.3879
    Epoch 167/200
    1200/1200 [==============================] - 0s 32us/step - loss: 0.3626 - val_loss: 0.3878
    Epoch 168/200
    1200/1200 [==============================] - 0s 37us/step - loss: 0.3625 - val_loss: 0.3878
    Epoch 169/200
    1200/1200 [==============================] - 0s 39us/step - loss: 0.3624 - val_loss: 0.3878
    Epoch 170/200
    1200/1200 [==============================] - 0s 34us/step - loss: 0.3623 - val_loss: 0.3877
    Epoch 171/200
    1200/1200 [==============================] - 0s 35us/step - loss: 0.3622 - val_loss: 0.3876
    Epoch 172/200
    1200/1200 [==============================] - 0s 40us/step - loss: 0.3621 - val_loss: 0.3874
    Epoch 173/200
    1200/1200 [==============================] - 0s 37us/step - loss: 0.3621 - val_loss: 0.3875
    Epoch 174/200
    1200/1200 [==============================] - 0s 32us/step - loss: 0.3620 - val_loss: 0.3873
    Epoch 175/200
    1200/1200 [==============================] - 0s 92us/step - loss: 0.3619 - val_loss: 0.3872
    Epoch 176/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3618 - val_loss: 0.3872
    Epoch 177/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3617 - val_loss: 0.3871
    Epoch 178/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.3617 - val_loss: 0.3870
    Epoch 179/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.3616 - val_loss: 0.3868
    Epoch 180/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3615 - val_loss: 0.3868
    Epoch 181/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3614 - val_loss: 0.3868
    Epoch 182/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3614 - val_loss: 0.3868
    Epoch 183/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.3613 - val_loss: 0.3867
    Epoch 184/200
    1200/1200 [==============================] - 0s 31us/step - loss: 0.3612 - val_loss: 0.3866
    Epoch 185/200
    1200/1200 [==============================] - 0s 31us/step - loss: 0.3611 - val_loss: 0.3865
    Epoch 186/200
    1200/1200 [==============================] - 0s 31us/step - loss: 0.3611 - val_loss: 0.3864
    Epoch 187/200
    1200/1200 [==============================] - 0s 31us/step - loss: 0.3610 - val_loss: 0.3863
    Epoch 188/200
    1200/1200 [==============================] - 0s 32us/step - loss: 0.3609 - val_loss: 0.3863
    Epoch 189/200
    1200/1200 [==============================] - 0s 32us/step - loss: 0.3608 - val_loss: 0.3862
    Epoch 190/200
    1200/1200 [==============================] - 0s 34us/step - loss: 0.3607 - val_loss: 0.3862
    Epoch 191/200
    1200/1200 [==============================] - 0s 31us/step - loss: 0.3607 - val_loss: 0.3862
    Epoch 192/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.3606 - val_loss: 0.3860
    Epoch 193/200
    1200/1200 [==============================] - 0s 28us/step - loss: 0.3605 - val_loss: 0.3859
    Epoch 194/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3605 - val_loss: 0.3860
    Epoch 195/200
    1200/1200 [==============================] - 0s 29us/step - loss: 0.3604 - val_loss: 0.3858
    Epoch 196/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3603 - val_loss: 0.3858
    Epoch 197/200
    1200/1200 [==============================] - 0s 27us/step - loss: 0.3602 - val_loss: 0.3858
    Epoch 198/200
    1200/1200 [==============================] - ETA: 0s - loss: 0.232 - 0s 27us/step - loss: 0.3602 - val_loss: 0.3857
    Epoch 199/200
    1200/1200 [==============================] - 0s 34us/step - loss: 0.3601 - val_loss: 0.3856
    Epoch 200/200
    1200/1200 [==============================] - 0s 30us/step - loss: 0.3601 - val_loss: 0.3856
    




    (0.7675414890194835, 0.033262851771355864)



另外，对NaN的填充，仅用训练数据和用增强后的数据也可以做对比.....

##### 4.2.2 数据增强-过采样
这里推荐imblearn工具  
pip install imblearn


```python
from imblearn.over_sampling import SMOTE
kfold= KFold(n_splits=5,random_state=42,shuffle=True)
scores=[]
for train_index,test_index in kfold.split(features_select_top_50_df[:891],labels):
    X_train=features_select_top_50_df.loc[train_index]
    y_train=labels[train_index]
    X_test=features_select_top_50_df.loc[test_index]
    y_test=labels[test_index]
    
    X_resampled,y_resampled=SMOTE(k_neighbors=5).fit_sample(X_train,y_train)
    
    gbdt=GradientBoostingClassifier()
    gbdt.fit(X_resampled,y_resampled)
    y_predict=gbdt.predict(X_test)
    f1_score=metrics.f1_score(y_test,y_predict)
    scores.append(f1_score)
np.mean(scores),np.std(scores)
```




    (0.8023959131004276, 0.05676594835661363)



##### 4.2.2 数据增强-自定义规则
对每条训练数据做如下操作：  
（1）随机删掉某个特征（0替换）；  
（2）随机交换同class的某个特征的值；  
（3）随机交换非class的某个特征的值；  


```python
def extend_data(train_df,train_y):
    #删除操作
    rows,cols=train_df.shape
    delete_df=copy.deepcopy(train_df)
    for i in range(0,rows):
        j=random.choice(range(0,cols))
        delete_df.iloc[i,j]=0#注意：要用iloc[i,j]的方式才能成功赋值，loc[i,j],iloc[i][j],iloc[i,j]的方式都不行
    #替换操作
    replace_df=copy.deepcopy(train_df)
    zero_class_df=train_df[train_y==0]
    one_class_df=train_df[train_y==1]
    zero_rows,_=zero_class_df.shape
    one_rows,_=one_class_df.shape
    for i in range(0,rows):
        j=random.choice(range(0,cols))
        if train_y.tolist()[i]==0:
            new_i=random.choice(range(0,zero_rows))
            replace_df.iloc[i,j]=zero_class_df.iloc[new_i,j]
        else:
            new_i=random.choice(range(0,one_rows))
            replace_df.iloc[i,j]=one_class_df.iloc[new_i,j]
    #替换操作
    replace_df2=copy.deepcopy(train_df)
    for i in range(0,rows):
        j=random.choice(range(0,cols))
        if train_y.tolist()[i]==0:
            new_i=random.choice(range(0,one_rows))
            replace_df2.iloc[i,j]=one_class_df.iloc[new_i,j]
        else:
            new_i=random.choice(range(0,zero_rows))
            replace_df2.iloc[i,j]=zero_class_df.iloc[new_i,j]
    #合并
    return pd.concat([train_df,delete_df,replace_df,replace_df2]),train_y.tolist()*4
```


```python
kfold= KFold(n_splits=5,random_state=42,shuffle=True)
scores=[]
for train_index,test_index in kfold.split(features_select_top_50_df[:891],labels):
    X_train=features_select_top_50_df.loc[train_index]
    y_train=labels[train_index]
    X_test=features_select_top_50_df.loc[test_index]
    y_test=labels[test_index]
    
    X_extended,y_extended=extend_data(X_train,y_train)
    X_extended2,y_extended2=extend_data(X_train,y_train)
    X_extended3,y_extended3=extend_data(X_train,y_train)
    
    gbdt=GradientBoostingClassifier()
    gbdt.fit(pd.concat([X_train,X_extended,X_extended2,X_extended3]),y_train.tolist()+y_extended+y_extended2+y_extended3)
    y_predict=gbdt.predict(X_test)
    f1_score=metrics.f1_score(y_test,y_predict)
    scores.append(f1_score)
np.mean(scores),np.std(scores)
```




    (0.8050731990298431, 0.05334388437545418)



这里把数据扩了12倍，多次运行，绝大部分情况下f1>0.8，当然，我们也可以与过采样的方法结合起来

#### 4.3模型优化  
模型的优化，可以考虑：  
（1）单模型优化：超参搜索；  
（2）多模型集成：集成学习；  

##### 4.3.1 超参数搜索
超参是指需要人为设定的参数，比如前面gbdt中的```n_estimators,max_depth,learning_rate```等；目前常见的超参搜索有网格搜索、随机搜索、贝叶斯优化搜索，还有基于强化学习的，比如google vizier...，其实比较好的方法是“人工智能”搜索（只需要一个excel表，并记录到相关操作对结果的改变就好了<坏结果也要保留>），接下来我们就在features_select_top_50_df数据集以及gbdt模型的基础上演示网格搜索、随机搜索、贝叶斯搜索...

##### 4.3.1 超参数搜索-网格搜索
网格搜索是将超参搜索空间切分成许多网格，我们在这些交点上选择一个较优秀的超参，但由于优化目标往往非凸，最优参数往往会成为漏网之鱼，通常比较建议的一种方式是在大范围内进行初步搜索，然后再在小范围内精确搜索。


```python
from sklearn.model_selection import GridSearchCV
#定义搜索空间
gdbt_parameters = {'max_depth': [3,4,5],'learning_rate':[0.1,0.15,0.2],'n_estimators':[50,80,100,150]}
#定义模型
gbdt=GradientBoostingClassifier()
#进行搜索
grid = GridSearchCV(gbdt, gdbt_parameters,scoring='f1')
grid.fit(features_select_top_50_df[:891], labels)
```




    GridSearchCV(cv='warn', error_score='raise-deprecating',
           estimator=GradientBoostingClassifier(criterion='friedman_mse', init=None,
                  learning_rate=0.1, loss='deviance', max_depth=3,
                  max_features=None, max_leaf_nodes=None,
                  min_impurity_decrease=0.0, min_impurity_split=None,
                  min_samples_leaf=1, min_sampl...      subsample=1.0, tol=0.0001, validation_fraction=0.1,
                  verbose=0, warm_start=False),
           fit_params=None, iid='warn', n_jobs=None,
           param_grid={'max_depth': [3, 4, 5], 'learning_rate': [0.1, 0.15, 0.2], 'n_estimators': [50, 80, 100, 150]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring='f1', verbose=0)




```python
grid.best_score_
```




    0.7913674844734889




```python
grid.best_params_
```




    {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}




```python
#试一试这组参数
classifier=GradientBoostingClassifier(n_estimators=80,max_depth=3,learning_rate=0.15)
scores = cross_val_score(classifier,features_select_top_50_df[:891],labels, scoring='f1', cv = 5)
np.mean(scores),np.std(scores)
```




    (0.8049563435310698, 0.04026419511814796)



##### 4.3.1超参数搜索-随机搜索
更多：https://blog.csdn.net/qq_36810398/article/details/86699842


```python
from sklearn.model_selection import RandomizedSearchCV
#定义搜索空间
gdbt_parameters = {'max_depth': [3,4,5],'learning_rate':[0.1,0.15,0.2],'n_estimators':[50,80,100,150]}
#定义模型
gbdt=GradientBoostingClassifier()
#进行搜索
random_search = RandomizedSearchCV(gbdt, gdbt_parameters,scoring='f1')
random_search.fit(features_select_top_50_df[:891], labels)
```




    RandomizedSearchCV(cv='warn', error_score='raise-deprecating',
              estimator=GradientBoostingClassifier(criterion='friedman_mse', init=None,
                  learning_rate=0.1, loss='deviance', max_depth=3,
                  max_features=None, max_leaf_nodes=None,
                  min_impurity_decrease=0.0, min_impurity_split=None,
                  min_samples_leaf=1, min_sampl...      subsample=1.0, tol=0.0001, validation_fraction=0.1,
                  verbose=0, warm_start=False),
              fit_params=None, iid='warn', n_iter=10, n_jobs=None,
              param_distributions={'max_depth': [3, 4, 5], 'learning_rate': [0.1, 0.15, 0.2], 'n_estimators': [50, 80, 100, 150]},
              pre_dispatch='2*n_jobs', random_state=None, refit=True,
              return_train_score='warn', scoring='f1', verbose=0)




```python
random_search.best_params_
```




    {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.15}




```python
#试一试这组参数
classifier=GradientBoostingClassifier(n_estimators=80,max_depth=3,learning_rate=0.1)
scores = cross_val_score(classifier,features_select_top_50_df[:891],labels, scoring='f1', cv = 5)
np.mean(scores),np.std(scores)
```




    (0.7953503302636137, 0.05131962771636548)



#### 4.3.1超参数搜索-贝叶斯优化
这里推荐使用Hyperopt工具  
更多：https://www.jianshu.com/p/35eed1567463


```python
from hyperopt import fmin, tpe, hp,STATUS_OK,Trials

#定义loss函数
def hyperopt_train_test(params):
    clf = GradientBoostingClassifier(**params)
    return cross_val_score(clf, features_select_top_50_df[:891],labels,cv=5,scoring='f1').mean()
#定义搜索空间
space4gbdt = {
    'max_depth': hp.choice('max_depth', [3,4,5]),
    'n_estimators': hp.choice('n_estimators', [50,80,100,150]),
    'learning_rate': hp.choice('learning_rate', [0.1,0.15,0.2])
}
#定义优化目标-最小化-f1
def f(params):
    f1 = hyperopt_train_test(params)
    return {'loss': -f1, 'status': STATUS_OK}
#查找最佳参数
trials = Trials()
best = fmin(f, space4gbdt, algo=tpe.suggest, max_evals=300, trials=trials)
print('best:',best)
```

    100%|████████████████████████████████████████████████| 300/300 [03:25<00:00,  1.59it/s, best loss: -0.8084986135045984]
    best: {'learning_rate': 1, 'max_depth': 0, 'n_estimators': 0}
    


```python
#试一试这组参数
classifier=GradientBoostingClassifier(n_estimators=50,max_depth=3,learning_rate=0.2)
scores = cross_val_score(classifier,features_select_top_50_df[:891],labels, scoring='f1', cv = 5)
np.mean(scores),np.std(scores)
```




    (0.8028268945820634, 0.04982023136963339)



从前面的几组参数来看，可以发现learning_rate在0.1到0.15,n_estimators在50,80之间都可取，max_depth都选择为3，接下来，我们换更小的步长进行搜索...


```python
from sklearn.model_selection import RandomizedSearchCV
#定义搜索空间
gdbt_parameters = {'learning_rate':[item/100 for item in list(range(10,21))],'n_estimators':range(50,81)}
#定义模型
gbdt=GradientBoostingClassifier()
#进行搜索
random_search = RandomizedSearchCV(gbdt, gdbt_parameters,scoring='f1')
random_search.fit(features_select_top_50_df[:891], labels)
```




    RandomizedSearchCV(cv='warn', error_score='raise-deprecating',
              estimator=GradientBoostingClassifier(criterion='friedman_mse', init=None,
                  learning_rate=0.1, loss='deviance', max_depth=3,
                  max_features=None, max_leaf_nodes=None,
                  min_impurity_decrease=0.0, min_impurity_split=None,
                  min_samples_leaf=1, min_sampl...      subsample=1.0, tol=0.0001, validation_fraction=0.1,
                  verbose=0, warm_start=False),
              fit_params=None, iid='warn', n_iter=10, n_jobs=None,
              param_distributions={'learning_rate': [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2], 'n_estimators': range(50, 81)},
              pre_dispatch='2*n_jobs', random_state=None, refit=True,
              return_train_score='warn', scoring='f1', verbose=0)




```python
random_search.best_params_
```




    {'n_estimators': 52, 'learning_rate': 0.12}




```python
#试一试这组参数
classifier=GradientBoostingClassifier(n_estimators=50,max_depth=3,learning_rate=0.17)
scores = cross_val_score(classifier,features_select_top_50_df[:891],labels, scoring='f1', cv = 5)
np.mean(scores),np.std(scores)
```




    (0.7901044801644943, 0.05243792278711915)



### 4.3.2 集成学习
最后我们还可以将多个模型的输出结果进行集成，常见的bagging(代表是rf),boosting(代表是gbdt)；另外gbdt的多种实现版本，大家可以在各种竞赛(特别是kaggle)中经常见到，比如xgboost,lightgbm,catboost等，这里我介绍另外一种比较暴力的集成学习方法：**stacking**，它将模型的预测结果作为上一层模型的特征输入，结构如图：  
![avatar](./source/stacking.jpg)

更多： https://github.com/zhulei227/Stacking_Ensembles  
更多stacking集成工具：https://www.jianshu.com/p/59313f43916f


```python
from stacking_classifier import *
#定义模型结构
classifier = StackingClassifier(
    base_classifiers=[
        RandomForestClassifier(),
        AdaBoostClassifier(),
        BaggingClassifier(),
        GradientBoostingClassifier(),
        LightGBMClassifier(),
        SVMClassifier(),
        NaiveBayesClassifier(),
    ],
    meta_classifier=LogisticRegression(),
    subsample_features_rate=0.9,
    n_jobs=-1
)
classifier.build_model()
```


```python
X_train,X_test, y_train, y_test =train_test_split(features_select_top_50_df[:891], labels,test_size=0.2, random_state=42)
classifier.fit(X_train,y_train)
y_predict=classifier.predict(X_test)
f1_score=metrics.f1_score(y_test,y_predict)
f1_score
```

    3 35
    2 1
    2 5
    




    0.8085106382978724


