import re

import pandas as pd

def _python_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def analyze_time_series(
    df,
    value_col="OBS_VALUE",
    time_col="TIME_PERIOD",
):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    missing_columns = [
        column for column in (value_col, time_col) if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            "DataFrame is missing required column(s): "
            + ", ".join(missing_columns)
        )
    if df.empty:
        raise ValueError("Cannot analyze an empty time series")
    if df[time_col].isna().any():
        raise ValueError(f"{time_col} contains missing periods")

    series = df.copy()
    try:
        series[value_col] = pd.to_numeric(series[value_col], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{value_col} must contain numeric values") from error

    series = series.dropna(subset=[value_col])
    if series.empty:
        raise ValueError(f"{value_col} contains no numeric observations")

    period_strings = series[time_col].astype(str)
    annual_periods = period_strings.map(
        lambda value: re.fullmatch(r"\d{4}", value) is not None
    ).all()
    if annual_periods:
        series["_time_sort_key"] = period_strings.astype(int)
    else:
        parsed_periods = pd.to_datetime(period_strings, errors="coerce")
        series["_time_sort_key"] = (
            parsed_periods
            if not parsed_periods.isna().any()
            else period_strings
        )
    series = series.sort_values(
        ["_time_sort_key", time_col],
        kind="stable",
    ).drop(columns="_time_sort_key")

    if series[time_col].duplicated().any():
        duplicate_periods = (
            series.loc[series[time_col].duplicated(keep=False), time_col]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            "Time series must have one observation per period; duplicates: "
            + ", ".join(duplicate_periods)
        )

    values = series[value_col]
    periods = series[time_col].tolist()
    first_value = float(values.iloc[0])
    last_value = float(values.iloc[-1])
    absolute_change = last_value - first_value
    percent_change = (
        absolute_change / first_value * 100
        if first_value != 0
        else None
    )
    min_index = values.idxmin()
    max_index = values.idxmax()

    year_over_year = []
    if annual_periods:
        for index in range(1, len(series)):
            previous_period = int(str(series.iloc[index - 1][time_col]))
            current_period = int(str(series.iloc[index][time_col]))
            if current_period - previous_period != 1:
                continue

            previous_value = float(series.iloc[index - 1][value_col])
            current_value = float(series.iloc[index][value_col])
            change = current_value - previous_value
            year_over_year.append({
                "previous_period": _python_value(
                    series.iloc[index - 1][time_col]
                ),
                "period": _python_value(series.iloc[index][time_col]),
                "previous_value": previous_value,
                "value": current_value,
                "absolute_change": change,
                "percent_change": (
                    change / previous_value * 100
                    if previous_value != 0
                    else None
                ),
            })

    dataset_ids = (
        df["dataset_id"].dropna().astype(str).drop_duplicates().tolist()
        if "dataset_id" in df.columns
        else []
    )
    if len(dataset_ids) > 1:
        raise ValueError("DataFrame contains multiple dataset_id values")

    declared_dimensions = df.attrs.get("analysis_dimension_columns")
    if declared_dimensions is None:
        dimension_columns = [
            column
            for column in df.columns
            if column not in {
                "dataset_id",
                value_col,
                time_col,
                f"{time_col}_LABEL",
            }
            and not column.endswith("_LABEL")
            and column not in {
                "HSPA",
                "METHODOLOGY",
                "TIME_RANGE",
                "SURVEY_VERSION",
                "DECIMALS",
                "OBS_STATUS",
                "OBS_STATUS2",
                "OBS_STATUS3",
                "UNIT_MULT",
            }
        ]
    else:
        dimension_columns = [
            column
            for column in declared_dimensions
            if column in df.columns
        ]
    filters = {}
    selected_measure = None
    for column in dimension_columns:
        codes = df[column].dropna().astype(str).drop_duplicates().tolist()
        if len(codes) > 1:
            raise ValueError(
                f"DataFrame contains multiple values for dimension {column}"
            )
        if not codes:
            continue

        label_column = f"{column}_LABEL"
        labels = (
            df[label_column].dropna().astype(str).drop_duplicates().tolist()
            if label_column in df.columns
            else []
        )
        dimension = {
            "code": codes[0],
            "label": labels[0] if labels else None,
        }
        if column.upper() in {"MEASURE", "INDICATOR"}:
            selected_measure = {
                "dimension_id": column,
                **dimension,
            }
        else:
            filters[column] = dimension

    if selected_measure is None and "INDICATOR" in df.columns:
        codes = (
            df["INDICATOR"].dropna().astype(str).drop_duplicates().tolist()
        )
        if len(codes) > 1:
            raise ValueError(
                "DataFrame contains multiple values for dimension INDICATOR"
            )
        if codes:
            labels = (
                df["INDICATOR_LABEL"]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .tolist()
                if "INDICATOR_LABEL" in df.columns
                else []
            )
            selected_measure = {
                "dimension_id": "INDICATOR",
                "code": codes[0],
                "label": labels[0] if labels else None,
            }

    return {
        "dataset_id": dataset_ids[0] if dataset_ids else None,
        "selected_measure": selected_measure,
        "filters": filters,
        "source_columns": {
            "value": value_col,
            "time": time_col,
        },
        "statistics": {
            "observation_count": len(series),
            "first_period": _python_value(periods[0]),
            "first_value": first_value,
            "last_period": _python_value(periods[-1]),
            "last_value": last_value,
            "absolute_change": absolute_change,
            "percent_change": percent_change,
            "minimum": {
                "period": _python_value(series.loc[min_index, time_col]),
                "value": float(values.loc[min_index]),
            },
            "maximum": {
                "period": _python_value(series.loc[max_index, time_col]),
                "value": float(values.loc[max_index]),
            },
            "mean": float(values.mean()),
            "year_over_year_changes": year_over_year,
        },
    }


def analyze_data_source_time_series(
    data_source,
    dataset_id,
    filters,
    start_period=None,
    end_period=None,
    value_col="OBS_VALUE",
    time_col="TIME_PERIOD",
    limit=1000,
):
    if not isinstance(filters, dict):
        raise TypeError("filters must map dimension/field IDs to values")

    structure = data_source.get_cached_structure(dataset_id)
    if structure is None:
        structure = data_source.get_structure(dataset_id)
    observation_result = data_source.retrieve_data(
        dataset_id,
        filters,
        start_period=start_period,
        end_period=end_period,
        limit=limit,
    )
    dataframe = data_source.normalize_dataframe(
        observation_result,
        structure,
    )
    analysis = analyze_time_series(
        dataframe,
        value_col=value_col,
        time_col=time_col,
    )
    observations = []
    for row in dataframe.to_dict(orient="records"):
        observations.append({
            key: _python_value(value)
            for key, value in row.items()
        })

    return {
        "request": {
            "dataset_id": dataset_id,
            "filters": dict(filters),
            "start_period": start_period,
            "end_period": end_period,
            "limit": limit,
        },
        "measure": analysis["selected_measure"],
        "filters": analysis["filters"],
        "observations": observations,
        "statistics": analysis["statistics"],
        "provenance": data_source.get_provenance(
            dataset_id,
            structure,
            observation_result,
        ),
    }


def analyze_oecd_time_series(
    dataset_id,
    measure_code,
    filters,
    start_period=None,
    end_period=None,
):
    if not isinstance(filters, dict):
        raise TypeError("filters must map OECD dimension IDs to code IDs")

    from data_sources.oecd import OECDDataSource

    data_source = OECDDataSource()
    structure = data_source.get_indexed_structure(dataset_id)
    dimensions = structure["data_structure"]["dimensions"]
    measure_dimensions = [
        dimension
        for dimension in dimensions
        if dimension.get("type") == "MeasureDimension"
        or dimension.get("id", "").upper() in {"MEASURE", "INDICATOR"}
    ]
    if len(measure_dimensions) != 1:
        raise ValueError(
            "Expected exactly one measure dimension in the OECD structure"
        )

    measure_dimension = measure_dimensions[0]
    if measure_dimension["id"] in filters:
        raise ValueError(
            f"Pass {measure_dimension['id']} through measure_code, not filters"
        )

    request_filters = {
        **filters,
        measure_dimension["id"]: measure_code,
    }
    result = analyze_data_source_time_series(
        data_source,
        dataset_id,
        request_filters,
        start_period=start_period,
        end_period=end_period,
    )
    result["request"] = {
        "dataset_id": dataset_id,
        "measure_code": measure_code,
        "filters": dict(filters),
        "start_period": start_period,
        "end_period": end_period,
    }
    return result


if __name__ == "__main__":
    dataset_id = "OECD.ELS.HD,DSD_HCQO@DF_PC,2.1"
    result = analyze_oecd_time_series(
        dataset_id,
        measure_code="ADMRASTH",
        filters={
            "REF_AREA": "AUS",
            "FREQ": "A",
            "UNIT_MEASURE": "10P5HB",
            "AGE": "Y_GE15",
            "SEX": "_T",
            "STATISTICAL_OPERATION": "OBS",
            "INCOME_GROUP": "_Z",
            "HEALTH_PROF": "_Z",
        },
        start_period="2016",
        end_period="2023",
    )
    statistics = result["statistics"]
    assert statistics["first_period"] == "2016"
    assert statistics["last_period"] == "2023"
    assert statistics["first_value"] == 70.4
    assert statistics["last_value"] == 52.0
    assert abs(statistics["absolute_change"] - (-18.4)) < 1e-9
    assert abs(statistics["percent_change"] - (-26.1363636364)) < 1e-8
    assert abs(statistics["mean"] - 55.5) < 1e-9
    print("Request:")
    print(result["request"])
    print("Measure:")
    print(result["measure"])
    print("Decoded filters:")
    print(result["filters"])
    print("Provenance:")
    print(result["provenance"])
    print("\nSummary:")
    for key, value in statistics.items():
        if key != "year_over_year_changes":
            print(f"  {key}: {value}")
    print("\nYear-by-year changes:")
    for change in statistics["year_over_year_changes"]:
        print(
            f"  {change['previous_period']} -> {change['period']}: "
            f"{change['absolute_change']:+.1f} "
            f"({change['percent_change']:+.2f}%)"
        )
    print(f"\nObservation rows: {len(result['observations'])}")
