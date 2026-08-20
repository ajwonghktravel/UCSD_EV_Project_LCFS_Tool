import argparse
import glob
import os
import re
from datetime import datetime
from pathlib import Path
import dateparser
from dateutil import parser
import numpy as np
import pandas as pd


def parse_quarter_year(text):
    """Parse a (quarter, year) pair out of a string such as 'Q1 2024',
    '2024-Q1', '1Q24', 'Q1_2024', etc.

    Returns a tuple (quarter:int, year:int), or None if no pair could be
    found in the given text.
    """
    if text is None:
        return None
    text = str(text).strip()

    # Q1 2024 / Q1-2024 / Q1_2024 / Q1/2024 / Q12024
    m = re.search(r'Q\s*([1-4])\D{0,3}(\d{2,4})', text, re.IGNORECASE)
    if m:
        q, y = int(m.group(1)), int(m.group(2))
        return q, (y + 2000 if y < 100 else y)

    # 2024 Q1 / 2024-Q1 / 2024_Q1
    m = re.search(r'(\d{2,4})\D{0,3}Q\s*([1-4])', text, re.IGNORECASE)
    if m:
        y, q = int(m.group(1)), int(m.group(2))
        return q, (y + 2000 if y < 100 else y)

    # 1Q24 / 1Q2024
    m = re.search(r'([1-4])Q\s*(\d{2,4})', text, re.IGNORECASE)
    if m:
        q, y = int(m.group(1)), int(m.group(2))
        return q, (y + 2000 if y < 100 else y)

    return None


def find_ci_column(carbon_intensity_df, quarter, year):
    """Locate the column in carbon_intensity_df whose header encodes the
    requested quarter/year (e.g. quarter=1, year=2024 -> a column header
    like 'Q1 2024').
    """
    matches = [
        col for col in carbon_intensity_df.columns
        if parse_quarter_year(col) == (quarter, year)
    ]

    if not matches:
        parsed_headers = {col: parse_quarter_year(col) for col in carbon_intensity_df.columns}
        raise ValueError(
            f"No column found for Q{quarter} {year} in the carbon intensity "
            f"sheet. Column headers found: {parsed_headers}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple columns matched Q{quarter} {year}: {matches}. "
            "Column headers must uniquely identify a quarter/year."
        )
    return matches[0]


def align_carbon_intensity(carbon_intensity_df, target_index, quarter, year):
    """Select the carbon-intensity column for the requested quarter/year
    and align it with target_index. Assumes the carbon intensity sheet has
    one column per quarter, with rows sorted by hour (0-23, top to bottom).
    """
    ci_column = find_ci_column(carbon_intensity_df, quarter, year)
    ci_series = pd.to_numeric(carbon_intensity_df[ci_column], errors='coerce')

    # Handle missing values
    if ci_series.isna().any():
        print(f"Warning: {ci_series.isna().sum()} missing values found in carbon intensity data")
        ci_series = ci_series.fillna(ci_series.mean())

    # Handle length mismatches
    if len(ci_series) > len(target_index):
        ci_series = ci_series.iloc[:len(target_index)]
    elif len(ci_series) < len(target_index):
        print(f"Warning: ci_series length ({len(ci_series)}) shorter than target ({len(target_index)})")
        last_value = ci_series.iloc[-1] if len(ci_series) > 0 else 0
        ci_series = pd.concat([ci_series, pd.Series([last_value] * (len(target_index) - len(ci_series)))])

    ci_series.index = target_index

    # Final NaN check
    if ci_series.isna().any():
        print(f"Warning: {ci_series.isna().sum()} NaN values remain. Filling with 0.")
        ci_series = ci_series.fillna(0)
    ci_column = find_ci_column(carbon_intensity_df, quarter, year)
    print(f"Using carbon intensity column: '{ci_column}' for Q{quarter} {year}")
    print(f"ci_series head:\n{ci_series.head()}")
    return ci_series



def get_latest_file(data_folder, explicit_name=None, pattern=None, quarter = None, year = None):
    """Generic function to get the latest file."""
    candidate_files = []

    if explicit_name:
        explicit_path = os.path.join(data_folder, explicit_name)
        if os.path.exists(explicit_path):
            candidate_files.append(explicit_path)

    if pattern:
        pattern_path = os.path.join(data_folder, pattern)
        candidate_files.extend(glob.glob(pattern_path))

    if not candidate_files:
        raise FileNotFoundError(f"No files found matching explicit_name='{explicit_name}' or pattern='{pattern}'")
    return max(candidate_files, key=os.path.getmtime)

def lcfs_filing_manual(data_folder, quarter, year, ci_benchmark, grid_avg, eer_value, energy_density):
    # Load all three files
    df_lcfs_pf = pd.read_csv(
        get_latest_file(data_folder, explicit_name="powerflex_data.csv", pattern="UCSD-All Sites - *.csv"),
        dtype=str
    )

    df_lcfs_chargepoint = pd.read_csv(
        get_latest_file(data_folder, explicit_name="chargepoint_data.txt", pattern="Session-Details-Meter-*.txt"),
        sep='\t',
        dtype=str
    )
    df_lcfs_flipturn = pd.read_csv(
        get_latest_file(data_folder, explicit_name="flipturn_data.csv", pattern="intervals-*.csv"),
        dtype=str
    )
    df_evseid_sn = pd.read_csv(
        get_latest_file(data_folder, explicit_name="evseid_sn_data.csv"), dtype=str
    )
    df_registered_fse = pd.read_csv(
        get_latest_file(data_folder, explicit_name="Registered FSE.csv", pattern="Registered FSE-*.csv"), dtype=str
    )
    carbon_intensity = pd.read_excel(os.path.join(data_folder, "ca_carbon_intensity_values.xlsx"))

    output_dir = Path("reports/LCFS") / f"Q{quarter}_{year}"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_registered_fse.columns = df_registered_fse.columns.str.strip()  # Remove leading/trailing spaces from column names
    df_registered_fse['End_Date'] = pd.to_datetime(
        df_registered_fse['End_Date'],
        format='mixed',   # adjust if your actual format differs, e.g. '%m/%d/%Y' for 4-digit years
        errors='coerce'
    )
    # Verify nothing failed to parse and nothing looks suspicious
    bad_dates = df_registered_fse['End_Date'].isna().sum()
    print(f"Number of invalid dates: {bad_dates}")
    print(df_registered_fse[['EVSE_Serial_No', 'End_Date']].drop_duplicates().sort_values('End_Date'))
    df_registered_fse = df_registered_fse[df_registered_fse['End_Date'].isna() | (df_registered_fse['End_Date'] > pd.Timestamp.now())]

    # Transformation for Powerflex
    df_lcfs_pf['Interval start'] = df_lcfs_pf['Interval start'].str.replace(' PST', '', case=False, regex=False)
    df_lcfs_pf['Interval start'] = df_lcfs_pf['Interval start'].str.replace(' PDT', '', case=False, regex=False)
    df_lcfs_pf['Interval start'] = pd.to_datetime(
        df_lcfs_pf['Interval start'],
        format='%m-%d-%Y %H:%M:%S',
        errors='coerce'
    )
    # Extract the hour
    df_lcfs_pf['hour'] = df_lcfs_pf['Interval start'].dt.hour
    # Group by hour and serial_number, summing the relevant columns
    agg_cols = ['Interval kWh']
    # Ensure numeric, coerce errors to NaN
    for col in agg_cols:
        df_lcfs_pf[col] = pd.to_numeric(df_lcfs_pf[col], errors='coerce')
    # Now group and sum
    df_lcfs_pf_reporting = df_lcfs_pf.pivot_table(
        index="hour",
        columns="FSE ID",
        values="Interval kWh",
        aggfunc="sum",
        fill_value=0
    )

    # Transform Chargepoint data
    df_lcfs_chargepoint["System S/N"] = pd.merge(df_lcfs_chargepoint, df_evseid_sn, left_on="EVSE ID", right_on="EVSE ID", how="left")["System S/N"]
    df_lcfs_chargepoint['FSE ID'] = pd.merge(df_lcfs_chargepoint, df_registered_fse, left_on="System S/N", right_on="EVSE_Serial_No", how="left")["FSE_ID"]

    df_lcfs_chargepoint.columns = df_lcfs_chargepoint.columns.str.strip()  # Remove leading/trailing spaces from column names
    df_lcfs_chargepoint['Power Start Time'] = pd.to_datetime(df_lcfs_chargepoint["Power Start Time"], errors='coerce')
    df_lcfs_chargepoint['hour'] = df_lcfs_chargepoint['Power Start Time'].dt.hour
    df_lcfs_chargepoint['Energy Consumed (AC kWh)'] = pd.to_numeric(df_lcfs_chargepoint['Energy Consumed (AC kWh)'], errors='coerce')
    df_lcfs_chargepoint_reporting = df_lcfs_chargepoint.pivot_table(
        index="hour",
        columns="FSE ID",
        values="Energy Consumed (AC kWh)",
        aggfunc="sum",
        fill_value=0
    )

    # Transform Flipturn data
    df_lcfs_flipturn['IntervalStartDateTime'] = pd.to_datetime(df_lcfs_flipturn['IntervalStartDateTime'], errors='coerce')
    df_lcfs_flipturn['IntervalStartDateTime'] = df_lcfs_flipturn['IntervalStartDateTime'].dt.tz_convert('America/Los_Angeles')
    df_lcfs_flipturn['FSE ID'] = pd.merge(df_lcfs_flipturn, df_registered_fse, left_on="ChargerSerialNumber", right_on="EVSE_Serial_No", how="left")["FSE_ID"]
    df_lcfs_flipturn['hour'] = df_lcfs_flipturn['IntervalStartDateTime'].dt.hour
    df_lcfs_flipturn['IntervalEnergyConsumedkWh'] = pd.to_numeric(df_lcfs_flipturn['IntervalEnergyConsumedkWh'], errors='coerce')
    df_lcfs_flipturn_reporting = df_lcfs_flipturn.pivot_table(
        index="hour",
        columns="FSE ID",
        values="IntervalEnergyConsumedkWh",
        aggfunc="sum",
        fill_value=0
    )

    df_combined = pd.concat([df_lcfs_pf_reporting, df_lcfs_chargepoint_reporting, df_lcfs_flipturn_reporting], axis=1).fillna(0)
    df_combined.to_csv(os.path.join(output_dir, "combined_lcfs_data.csv"), index=True)

    ci_series = align_carbon_intensity(carbon_intensity, df_combined.index, quarter, year)
    print(f"ci_series length: {len(ci_series)}")
    print(f"df_combined length: {len(df_combined)}")
    print(f"ci_series index: {ci_series.index[:5]}")  # First 5 values
    print(f"df_combined index: {df_combined.index[:5]}")  # First 5 values

    df_allsmart = df_combined.mul((ci_benchmark - (ci_series / eer_value)) * eer_value * energy_density / 1000000, axis=0)
    df_allgridavg = df_combined.mul((ci_benchmark - (grid_avg / eer_value)) * eer_value * energy_density / 1000000, axis=0)
    smart_charging_sums = df_allsmart.sum()
    grid_avg_sums = df_allgridavg.sum()
    smart_is_higher = smart_charging_sums > grid_avg_sums
    df_smart_wins = df_combined.loc[:, smart_is_higher]
    df_gridavg_wins = df_combined.loc[:, ~smart_is_higher]

    # Create optimized credits dataframe
    df_optimized_credits = pd.DataFrame(
        np.where(smart_is_higher.values, df_allsmart.values, df_allgridavg.values),
        index=df_allsmart.index,
        columns=df_allsmart.columns
    )
    # Create a separate dataframe showing which method was used
    df_method_used = pd.DataFrame(
        ['smart' if x else 'grid_avg' for x in smart_is_higher],
        index=smart_is_higher.index,
        columns=['method_used']
    ).T

    # Transform grid average into copy-pastable values
    df_gridavg_sums = df_gridavg_wins.sum()
    df_gridavg_sums = df_gridavg_sums.reset_index()
    df_gridavg_sums.columns = ['FSE ID', 'Grid Average kWh']
    df_gridavg_sums = df_gridavg_sums[df_gridavg_sums['Grid Average kWh'] != 0]
    df_gridavg_sums.to_csv(output_dir / f"gridavg_sums_reporting{datetime.now().strftime('%Y%m%d')}.csv", index=True)

    # Or add the method as a row in the optimized credits dataframe
    df_optimized_with_method = pd.concat([df_optimized_credits, df_method_used])
    timestamp = datetime.now().strftime('%Y%m%d')
    fse_columns_final = pd.DataFrame(df_combined.columns, columns=['FSE ID'])
    #fse_columns_final.to_csv(output_dir / f"fse_columns_{timestamp}.csv", index=False)
    df_allsmart.to_csv(output_dir / "all_smart_charging.csv", index=False)
    df_allgridavg.to_csv(output_dir / "all_grid_average.csv", index=False)
    df_smart_wins.to_csv(output_dir / f"interval_sessions_smartcharging_lcfs_{timestamp}.csv")
    df_gridavg_wins.to_csv(output_dir / f"interval_sessions_gridavg_{timestamp}.csv")
    df_optimized_with_method.to_csv(output_dir / f"optimized_credits_with_method_{timestamp}.csv")

    print(f"Total credits (smart charging): {df_allsmart.sum().sum():,.2f}")
    print(f"Total credits (grid average): {df_allgridavg.sum().sum():,.2f}")
    print(f"Total credits (optimized): {df_optimized_credits.sum().sum():,.2f}")
    print(f"Optimization benefit: {(df_optimized_credits.sum().sum() - df_allgridavg.sum().sum()):,.2f}")
    print("FSEs using smart:", smart_is_higher.sum(), "| grid:", (~smart_is_higher).sum())
    print(f"Total kWh Delivered: {df_combined.to_numpy().sum():,.2f}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run the LCFS filing calculation for a given quarter."
    )
    parser.add_argument(
        "--quarter",
        required=True,
        help="Reporting quarter, e.g. 'Q1 2026', '2026-Q1', or '1Q26'. Must match a column header in ca_carbon_intensity_values.xlsx.",
    )
    parser.add_argument(
        "--ci-benchmark",
        type=float,
        required=True,
        dest="ci_benchmark",
        help="LCFS carbon intensity benchmark (e.g. 75.16).",
    )
    parser.add_argument(
        "--grid-avg",
        type=float,
        required=True,
        dest="grid_avg",
        help="Grid average carbon intensity (e.g. 65.07).",
    )
    parser.add_argument(
        "--eer-value",
        type=float,
        default=3.4,
        dest="eer_value",
        help="Energy Economy Ratio value (default: 3.4).",
    )
    parser.add_argument(
        "--energy-density",
        type=float,
        default=3.6,
        dest="energy_density",
        help="Energy density conversion factor (default: 3.6).",
    )
    parser.add_argument(
        "--data-folder",
        default=None,
        dest="data_folder",
        help="Path to the folder containing input data files "
             "(default: <script_dir>/Data).",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    parsed_quarter = parse_quarter_year(args.quarter)
    if parsed_quarter is None:
        raise SystemExit(
            f"Could not parse --quarter value '{args.quarter}'. "
            "Expected something like 'Q1 2026', '2026-Q1', or '1Q26'."
        )
    quarter_num, year_num = parsed_quarter

    BASE_DIR = Path(__file__).resolve().parent
    data_folder = Path(args.data_folder) if args.data_folder else BASE_DIR / "Data"

    lcfs_filing_manual(
        data_folder=data_folder,
        quarter=quarter_num,
        year=year_num,
        ci_benchmark=args.ci_benchmark,
        grid_avg=args.grid_avg,
        eer_value=args.eer_value,
        energy_density=args.energy_density,
    )