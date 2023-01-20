import itertools
import json
import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import polyval
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

RESOURCES_FOLDER = 'res'
DATASET_PATH = os.path.join(RESOURCES_FOLDER, 'weather_history_in_szeged.csv')
REPORT_PATH = os.path.join(RESOURCES_FOLDER, 'report.json')
MIN_P_STD = 0.1
BLOCK_PLOTS = True
PLOT = True
RUN_LR = True
RUN_PR = True
RUN_MLR = True
RUN_MPR = True
MAX_DEGREE = 5
TOP_CORRELATED = 3
CORRELATION_THRESHOLD = 0.8
LOOKING_FOR_HIGH_CORRELATION = True


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_results_report(rep):
    try:
        with open(REPORT_PATH, 'w') as f:
            json.dump(rep, f, cls=NumpyEncoder, indent=2)
    except Exception as e:
        print(e)


df = pd.read_csv(DATASET_PATH)

DATE_COLUMN = 'Formatted Date'
TEXT_COLUMNS = ['Summary', 'Daily Summary', 'Precip Type']

print(f'Dataset size {df.shape[0]}')
df.dropna(inplace=True)
print(f'Dataset size {df.shape[0]}, after cleaning')

print(df.head())

df = df[df.columns.difference(TEXT_COLUMNS)]

df[DATE_COLUMN] = df[DATE_COLUMN].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f %z'))

to_drop = []
for column in df.columns.difference([DATE_COLUMN]):
    print(f'Summary for column {column}:')
    min_val = df[column].min()
    max_val = df[column].max()
    unique = df[column].nunique()
    std = df[column].std()
    amplitude = abs(abs(min_val) - abs(max_val))
    std_per = std / amplitude if amplitude > 0 else 0
    mean = df[column].mean()
    median = df[column].median()
    per_25 = df[column].quantile(0.25)
    per_50 = df[column].quantile(0.5)
    per_75 = df[column].quantile(0.75)
    per_90 = df[column].quantile(0.9)
    per_95 = df[column].quantile(0.95)
    per_99 = df[column].quantile(0.99)
    print(f'\tUnique values: {unique}')
    print(f'\tMean: {mean}')
    print(f'\tMedian: {median}')
    print(f'\tStandard Deviation: {std}')
    print(f'\tStandard Deviation %: {std_per}')
    print(f'\tMin value: {min_val}')
    print(f'\t25 Percentile: {per_25}')
    print(f'\t25 Percentile: {per_50}')
    print(f'\t25 Percentile: {per_75}')
    print(f'\t25 Percentile: {per_90}')
    print(f'\t25 Percentile: {per_95}')
    print(f'\t25 Percentile: {per_99}')
    print(f'\tMax value: {max_val}')
    print(f'\tAmplitude: {amplitude}')

    if PLOT:
        plt.scatter(df[DATE_COLUMN], df[column], s=5)
        plt.title(f'{column.capitalize()} over time in Szeged, Hungary')
        plt.xlabel('Date')
        plt.ylabel(column)
        plt.tight_layout()
        plt.show(block=BLOCK_PLOTS)

        if amplitude > 0:
            amount_bins = 10
            bin_width = (math.ceil(max_val) - math.floor(min_val)) / amount_bins
            bins = np.arange(math.floor(min_val), math.ceil(max_val) + bin_width, bin_width)
            ax = df[column].plot.hist(bins=bins, title=f'{column.capitalize()} histogram', rwidth=0.5,
                                      figsize=(12, 4.8))
            ax.set_xlabel(f'{column}')
            plt.tight_layout()
            plt.xticks(bins)
            plt.show(block=BLOCK_PLOTS)
    print()
    print()

    if std < MIN_P_STD:
        print(f'Dropping {column} due to low variance')
        to_drop.append(column)

df.drop(to_drop, axis=1, inplace=True)
print(f'Removed {len(to_drop)} columns, {to_drop}')

data_columns = df.columns.difference([DATE_COLUMN])

# print('Normalizing data')
# df.loc[:, data_columns] = (df[data_columns] - df[data_columns].min()) / (
#         df[data_columns].max() - df[data_columns].min())

correlation_matrix = df.corr(numeric_only=True)
print(correlation_matrix)

if PLOT:
    plt.matshow(correlation_matrix.abs(), cmap='RdBu')
    plt.colorbar()  # location='bottom'
    variables = correlation_matrix.columns
    plt.xticks(np.arange(0, len(variables)), variables, **{'rotation': 20, 'fontsize': 7})
    plt.yticks(np.arange(0, len(variables)), variables, **{'rotation': 70, 'fontsize': 7})
    plt.title(f'Correlation matrix')
    # plt.tight_layout()
    plt.show(block=BLOCK_PLOTS)

upper_diagonal_mat = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool_))


def check_column(up_diag_mat, col):
    if LOOKING_FOR_HIGH_CORRELATION:
        any(up_diag_mat[col] > CORRELATION_THRESHOLD)
    else:
        any(up_diag_mat[col] < CORRELATION_THRESHOLD)


good_columns = [column for column in upper_diagonal_mat.columns if check_column(upper_diagonal_mat, column)]

print(
    f'Found {good_columns} columns with correlation {"above" if LOOKING_FOR_HIGH_CORRELATION else "below"} {CORRELATION_THRESHOLD}.')

if len(good_columns) < 2:
    good_columns = set()
    np_upper_diagonal_mat = upper_diagonal_mat.to_numpy()
    np_upper_diagonal_mat[np.isnan(np_upper_diagonal_mat)] = -666  # make nan become negative
    flat_upper_diagonal_mat = np_upper_diagonal_mat.flatten()
    flat_indexes = flat_upper_diagonal_mat.argsort()[-TOP_CORRELATED:]
    if LOOKING_FOR_HIGH_CORRELATION:
        flat_indexes = flat_indexes[::-1]
    x_idx, y_idx = np.unravel_index(flat_indexes, upper_diagonal_mat.shape)
    for x, y, in zip(x_idx, y_idx):
        print(f'Correlation between `{upper_diagonal_mat.columns[x]}` and `{upper_diagonal_mat.columns[y]}` '
              f'is {np_upper_diagonal_mat[x][y]}')
        good_columns.add(upper_diagonal_mat.columns[x])
        good_columns.add(upper_diagonal_mat.columns[y])
    good_columns = list(good_columns)

    print(
        f'Found {good_columns} columns {"high" if LOOKING_FOR_HIGH_CORRELATION else "low"} correlation, top {TOP_CORRELATED}.')

report = {}
if RUN_PR or RUN_LR:
    for i, (x_col, y_col) in enumerate(itertools.combinations(good_columns, 2)):
        print(f'Trying to predict `{y_col}` using `{x_col}` using linear regression...')
        x = df[x_col].to_numpy().reshape(-1, 1)
        y = df[y_col].to_numpy()
        max_val = x.max()
        min_val = x.min()
        x_ori = x.copy()
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True)
        x_train_old = x_train.copy()
        x_test_old = x_test.copy()
        x_scaler = MinMaxScaler(copy=True)
        y_scaler = MinMaxScaler(copy=True)
        x_scaler.fit(x)
        y_scaler.fit(y.reshape(-1, 1))
        if RUN_LR:
            x_train = x_scaler.transform(x_train)
            x_test = x_scaler.transform(x_test)
            y_train = y_scaler.transform(y_train.reshape(-1, 1)).reshape(-1)
            y_test = y_scaler.transform(y_test.reshape(-1, 1)).reshape(-1)
            lr = LinearRegression()
            lr.fit(x_train, y_train)
            r2_train = lr.score(x_train, y_train)
            r2_test = lr.score(x_test, y_test)
            angular_coef = lr.coef_[0]
            linear_coef = lr.intercept_
            y_pred = lr.predict(x_test)
            sample_points = np.arange(min_val * 1.1, max_val * 1.1, max_val / 500)
            line = polyval(sample_points, [linear_coef, angular_coef])

            print(f'Train R²: {r2_train}')
            print(f'Test R²: {r2_train}')

            if PLOT:
                plt.scatter(np.squeeze(x), y,
                            s=10, alpha=0.8, zorder=0,
                            label='_nolegend_')
                plt.plot(sample_points, line, zorder=1, label='linear regression curve', color='red')
                plt.title(f'{x_col} x {y_col} - All data')
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.tight_layout()
                plt.legend()
                plt.show(block=BLOCK_PLOTS)

                plt.scatter(np.arange(y_test.shape[0]), y_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1),
                            s=10,
                            alpha=0.8, zorder=0,
                            label='truth')
                plt.scatter(np.arange(y_pred.shape[0]), y_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1),
                            s=10,
                            alpha=0.8, zorder=1,
                            label='pred')
                plt.title(f'Forecast for {y_col} - R² {r2_train} - LR')
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.tight_layout()
                plt.legend()
                plt.show(block=BLOCK_PLOTS)

            report[f'linear_regression-{i}'] = {
                'x': x_col,
                'y': y_col,
                'r2_train': r2_train,
                'r2_test': r2_test,
                'angular_coef': angular_coef,
                'linear_coef': linear_coef,
            }

        if RUN_PR:
            for j in range(1, MAX_DEGREE, 1):
                print(f'Trying to predict `{y_col}` using `{x_col}` using polynomial regression of degree {j}...')
                pf = PolynomialFeatures(degree=j, include_bias=True)
                poly_x = pf.fit_transform(x_scaler.transform(x_ori.copy()))  # all data
                poly_x_train = pf.transform(x_train_old.copy())
                poly_x_test = pf.transform(x_test_old.copy())
                lr = LinearRegression()
                lr.fit(poly_x_train, y_train)
                r2_train = lr.score(poly_x_train, y_train)
                r2_test = lr.score(poly_x_test, y_test)
                angular_coefs = lr.coef_
                linear_coef = lr.intercept_
                y_pred = lr.predict(poly_x_test)
                poly_x_inv = x_scaler.inverse_transform(poly_x.copy())
                line = polyval(sample_points, [linear_coef] + angular_coefs)

                print(f'Train R²: {r2_train}')
                print(f'Test R²: {r2_train}')

                if PLOT:
                    x_t = np.transpose(poly_x_inv)
                    for k, dim_x_inv in enumerate(x_t):
                        plt.scatter(dim_x_inv, y,
                                    s=5, alpha=0.8, zorder=0,
                                    label=f'feature {k}')
                    plt.title(f'{x_col} x {y_col} - All data')
                    plt.xlabel(x_col)
                    plt.ylabel(y_col)
                    plt.tight_layout()
                    plt.legend()
                    plt.show(block=BLOCK_PLOTS)

                    plt.plot(sample_points, line, zorder=1, color='black')
                    plt.title(f'poly regre d{j} curve')
                    plt.xlabel(x_col)
                    plt.ylabel(y_col)
                    plt.tight_layout()
                    plt.show(block=BLOCK_PLOTS)

                    plt.scatter(np.arange(y_test.shape[0]), y_test, s=10,
                                alpha=0.8, zorder=0,
                                label='truth')
                    plt.scatter(np.arange(y_pred.shape[0]), y_pred, s=10,
                                alpha=0.8, zorder=1,
                                label='pred')
                    plt.title(f'Forecast for {y_col} - R² {r2_train} - PR')
                    plt.xlabel(x_col)
                    plt.ylabel(y_col)
                    plt.tight_layout()
                    plt.legend()
                    plt.show(block=BLOCK_PLOTS)

                report[f'polynomial_regression_d{j}-{i}'] = {
                    'x': x_col,
                    'y': y_col,
                    'r2_train': r2_train,
                    'r2_test': r2_test,
                    'angular_coefs': angular_coefs,
                    'linear_coef': linear_coef,
                }

x_col = [good_columns[0], good_columns[2]]
y_col = good_columns[1]
print(f'Trying to predict `{y_col}` using `{x_col}` using multi variable linear regression...')
x = df[x_col].to_numpy()
y = df[y_col].to_numpy()
max_val = x.max()
min_val = x.min()
x_ori = x.copy()
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True)
x_train_old = x_train.copy()
x_test_old = x_test.copy()
x_scaler = MinMaxScaler(copy=True)
x_scaler.fit(x)
y_scaler = MinMaxScaler(copy=True)
y_scaler.fit(y.reshape(-1, 1))

if RUN_MLR:
    x_train = x_scaler.transform(x_train)
    x_test = x_scaler.transform(x_test)
    y_train = y_scaler.transform(y_train.reshape(-1, 1)).reshape(-1)
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).reshape(-1)
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    r2_train = lr.score(x_train, y_train)
    r2_test = lr.score(x_test, y_test)
    angular_coefs = lr.coef_
    linear_coef = lr.intercept_
    y_pred = lr.predict(x_test)
    sample_points = np.arange(min_val * 1.1, max_val * 1.1, max_val / 500)
    line = polyval(sample_points, [linear_coef] + angular_coefs)

    print(f'Train R²: {r2_train}')
    print(f'Test R²: {r2_train}')

    if PLOT:
        x_inv = np.transpose(x_scaler.inverse_transform(x))
        plt.scatter(x_inv[0], y_scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1),
                    s=10, alpha=0.8, zorder=0,
                    label=f'{x_col[0]}')
        plt.scatter(x_inv[1], y_scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1),
                    s=10, alpha=0.8, zorder=0,
                    label=f'{x_col[1]}')
        plt.title(f'x columns x {y_col} - All data')
        plt.xlabel(str(x_col))
        plt.ylabel(y_col)
        plt.tight_layout()
        plt.legend()
        plt.show(block=BLOCK_PLOTS)

        plt.plot(sample_points, line, zorder=1, color='black')
        plt.title('multi linear regre curve')
        plt.xlabel(str(x_col))
        plt.ylabel(y_col)
        plt.tight_layout()
        plt.show(block=BLOCK_PLOTS)

        plt.scatter(np.arange(y_test.shape[0]), y_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1), s=10,
                    alpha=0.8, zorder=0,
                    label='truth')
        plt.scatter(np.arange(y_pred.shape[0]), y_scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1), s=10,
                    alpha=0.8, zorder=1,
                    label='pred')
        plt.title(f'Forecast for {y_col} - R² {r2_train} - MLR')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.tight_layout()
        plt.legend()
        plt.show(block=BLOCK_PLOTS)

    report[f'linear_regression_multi_var'] = {
        'x': x_col,
        'y': y_col,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'angular_coefs': angular_coefs,
        'linear_coef': linear_coef,
    }

if RUN_MPR:
    for j in range(1, MAX_DEGREE, 1):
        try:
            print(f'Trying to predict `{y_col}` using `{x_col}` using multi polynomial regression of degree {j}...')
            pf = PolynomialFeatures(degree=j, include_bias=True)
            poly_x = pf.fit_transform(x_scaler.transform(x_ori.copy()))  # all data
            poly_x_train = pf.transform(x_train_old.copy())
            poly_x_test = pf.transform(x_test_old.copy())
            lr = LinearRegression()
            lr.fit(poly_x_train, y_train)
            r2_train = lr.score(poly_x_train, y_train)
            r2_test = lr.score(poly_x_test, y_test)
            angular_coefs = lr.coef_
            linear_coef = lr.intercept_
            y_pred = lr.predict(poly_x_test)
            # poly_x_inv = x_scaler.inverse_transform(poly_x.copy())
            # poly_x_inv = poly_x.copy() - x_scaler.min_
            # poly_x_inv /= x_scaler.scale_
            pf = PolynomialFeatures(degree=j, include_bias=True)
            poly_x2 = pf.fit_transform(x_ori.copy())
            scaler_x2 = MinMaxScaler()
            scaler_x2.fit(poly_x2)
            poly_x_inv = scaler_x2.inverse_transform(poly_x)
            max_val = x.max()
            min_val = x.min()

            sample_points = np.arange(min_val * 1.1, max_val * 1.1, max_val / 500)
            line = polyval(sample_points, [linear_coef] + angular_coefs)

            print(f'Train R²: {r2_train}')
            print(f'Test R²: {r2_train}')

            if PLOT:
                x_t = np.transpose(poly_x_inv)
                for k, dim_x_inv in enumerate(x_t):
                    plt.scatter(dim_x_inv, y,
                                s=5, alpha=0.8, zorder=0,
                                label=f'feature {k}')
                plt.title(f'{x_col} x {y_col} - All data')
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.tight_layout()
                plt.legend()
                plt.show(block=BLOCK_PLOTS)

                plt.plot(sample_points, line, zorder=1, color='black')
                plt.title(f'multi poly regre d{j} curve')
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.tight_layout()
                plt.show(block=BLOCK_PLOTS)

                plt.scatter(np.arange(y_test.shape[0]), y_test, s=10,
                            alpha=0.8, zorder=0,
                            label='truth')
                plt.scatter(np.arange(y_pred.shape[0]), y_pred, s=10,
                            alpha=0.8, zorder=1,
                            label='pred')
                plt.title(f'Forecast for {y_col} - R² {r2_train} - PR')
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.tight_layout()
                plt.legend()
                plt.show(block=BLOCK_PLOTS)

            report[f'polynomial_regression_multi_var_d{j}-{i}'] = {
                'x': x_col,
                'y': y_col,
                'r2_train': r2_train,
                'r2_test': r2_test,
                'angular_coefs': angular_coefs,
                'linear_coef': linear_coef,
            }
        except:
            pass
save_results_report(report)

print()
