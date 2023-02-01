"""
Collection of decorators for common dataframe transformations
"""
from functools import wraps
from typing import ParamSpec, TypeVar, Callable, TypeAlias

import pandas as pd

P = ParamSpec("P")
R = TypeVar("R")


def add_column(
    col_name: str,
) -> Callable[[Callable[[pd.DataFrame], pd.Series]], Callable[[pd.DataFrame], pd.DataFrame]]:
    def my_decorator(f: Callable[[pd.DataFrame], pd.Series]) -> Callable[[pd.DataFrame], pd.DataFrame]:
        @wraps(f)
        def wrapper(df_: pd.DataFrame) -> pd.DataFrame:
            series = f(df_)
            df_[col_name] = series
            return df_

        return wrapper

    return my_decorator


def add_cols(
    *,
    new_col_name: str,
    src_col: str | None = None,
    src_cols: list[str] | None = None,
    ptype_fn: Callable[[pd.Series], bool] | None = None,
) -> Callable[[Callable[[pd.DataFrame], pd.Series]], Callable[[pd.DataFrame], pd.DataFrame]]:

    src_col_specified = src_col or src_cols
    if src_col_specified:
        assert not (
            bool(src_cols) and bool(src_col)
        ), "Error src_col and src_cols params are mutual exclusive"

    def my_decorator(
        f: Callable[[pd.DataFrame | pd.Series], pd.Series]
    ) -> Callable[[pd.DataFrame], pd.DataFrame]:
        @wraps(f)
        def wrapper(df_or_s: pd.DataFrame) -> pd.DataFrame:
            df_subset = df_or_s
            if src_col_specified:
                check_columns = [src_col] if src_col is not None else src_cols
                df_columns = set(df_or_s.columns)
                assert set(check_columns).issubset(df_columns), (
                    f"df doesn't contain all required columns {df_or_s.columns}"
                    f", df is missing {set(check_columns) - df_columns} columns"
                )
                df_subset = df_or_s[src_col] if src_col is not None else df_or_s[src_cols]

            new_column_data = f(df_subset)
            assert isinstance(new_column_data, pd.Series), (
                f"return value of applying transform was a " f"{type(new_column_data)} but expected a series"
            )
            if ptype_fn is not None:
                assert ptype_fn(
                    new_column_data
                ), f"ptype assertion {ptype_fn.__name__} failed on new column {new_column_data}"

            df_or_s[new_col_name] = new_column_data
            return df_or_s

        return wrapper

    return my_decorator


def update_col(
    *, col: str, ptype_fn: Callable[[pd.Series], bool] | None = None
) -> Callable[[Callable[[pd.DataFrame], pd.Series]], Callable[[pd.DataFrame], pd.DataFrame]]:
    return add_cols(new_col_name=col, src_col=col, ptype_fn=ptype_fn)
