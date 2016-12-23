import numpy as np
import math
import decimal
from decimal import (Decimal, ROUND_HALF_UP)


# 小数点切り捨て
def round_down1(value):
  value = Decimal(value).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
  return str(value)


def round_down2(value):
  value = Decimal(value).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
  return str(value)


def round_down3(value):
  value = Decimal(value).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
  return str(value)


# 平均を求める関数
# argument: np.array, np.array
def get_average(x, frequency=None):
  if (frequency is not None and len(x) == len(frequency)):
    total_frequency = np.sum(frequency)
    ave = np.sum(x * frequency) / total_frequency
    return ave
  else:
    total_frequency = len(x)
    ave = np.sum(x) / total_frequency
    return ave


# 分散を求める関数
# argument: np.array, np.array
def get_dispersion(x, frequency=None):
  if (frequency is not None and len(x) == len(frequency)):
    total_frequency = np.sum(frequency)
    ave = np.sum(x * frequency) / total_frequency
    return (np.sum((x ** 2) * frequency) / total_frequency) - (ave ** 2)
  else:
    total_frequency = len(x)
    ave = np.sum(x) / total_frequency
    return (x.dot(x) / len(x)) - (ave ** 2)


# 標準偏差を求める関数
# argument: dispersion(分散)
def get_standard_deviation(dispersion):
  return np.sqrt(dispersion)


# 偏差値を求める関数
def get_standard_score(average_score, score, standard_deviation):
  return (50 + 10 * ((score - average_score) / standard_deviation))


# 共分散を求める関数
# argument : np.array, np.array
def get_covariance(x, y):
  sum_x = np.sum(x)
  sum_y = np.sum(y)

  x_ave = sum_x / len(x)
  y_ave = sum_y / len(y)
  xy = x * y
  sum_xy = np.sum(xy)
  sum_xy_ave = sum_xy / len(xy)

  return sum_xy_ave - (x_ave * y_ave)


# 相関係数を求める関数
# argument: np.array, np.array
# return Correlation_Coefficient : 相関係数
def get_correlation_coefficient(x, y, x_frequency=None, y_frequency=None):
  # 分散
  x_dispersion = get_dispersion(x, x_frequency)
  y_dispersion = get_dispersion(y, y_frequency)

  # 標準偏差
  x_standard_deviation = get_standard_deviation(x_dispersion)
  y_standard_deviation = get_standard_deviation(y_dispersion)

  # xの標準偏差 * yの標準偏差
  denominator = x_standard_deviation * y_standard_deviation

  # 共分散
  numerator = get_covariance(x, y)

  Correlation_Coefficient = numerator / denominator
  return Correlation_Coefficient


# YのXへの回帰直線を求める関数
# argument x,y : np.array, np.array
# return a,b : 傾き, 切片
def get_regression_line(x, y):
  x_ave = get_average(x)
  y_ave = get_average(y)

  # 分散
  dispersion = get_dispersion(x)

  # 共分散
  covariance = get_covariance(x, y)

  # 共分散 / 分散
  a = covariance / dispersion

  b = -(a * x_ave) + y_ave
  # a : 傾き, b : 切片
  return a, b
