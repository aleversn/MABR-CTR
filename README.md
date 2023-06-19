# MABR-CTR

融合行为检索与记忆增强的点击率预测算法

## 使用数据集：
## 1、JD_Computers & JD_Applicances

## 下载链接:

https://drive.google.com/open?id=1kz7Dkq5jQ8WR82xBITyQ7S4vUgBFNsDn.

## 数据集介绍:


| Dataset | JD-Applicances | JD-Computers | Desc                            |
| ---------- | ---------- | ---------- | ------------------------------- |
| Users | 6,166,916    | 3,191,573 | users                           |
| Products | 169,856 | 419,388 | SKUs            |
| Categories | 103 | 93 | Leaf categories |
| Number of Micro behaviors      | 176,483,033 | 88,766,833 | micro behaviors |

 "**sku + behavior_type + category + time_interval + dwell_time**"
 
For example, "1993092+7+870+22+27" is a micro-behavior, which means that a user read the specification of product "1993092" (in the leaf category "870"). The time interval between this micro behavior and next micro behavior is 22 seconds. The user spends 27 seconds in this product.


2、UserBehavior

## 下载链接:

https://tianchi.aliyun.com/dataset/649

## 数据集介绍:


| Dataset | UserBehavior | Desc                            |
| ---------- | ---------- | ------------------------------- |
| Users | 987,994    | users                           |
| Products | 4,162,024 | SKUs            |
| Categories | 9,439 | Leaf categories |
| Number of Micro behaviors      | 100,150,807 | micro behaviors |


3、lastfm-1k

## 下载链接:

http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html

## 数据集介绍:

| Dataset | lastfm | Desc                            |
| ---------- | ---------- | ------------------------------- |
| Users | 937    | users                           |
| Products | 64,181 | SKUs            |
| Number of Micro behaviors      | 12,566,899 | micro behaviors |
