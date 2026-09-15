import pandas as pd
from datetime import datetime, date, time, timedelta
from pathlib import Path
import os
import numpy as np
def lcfs_filing_manual(data_folder, ci_benchmark, grid_avg, eer_value, energy_density):
    df_lcfs_pf = pd.read_csv(
        os.path.join(data_folder, "powerflex_data.csv"), dtype=str
    )
    df_lcfs_chargepoint = pd.read_csv(
        os.path.join(data_folder, "chargepoint_data.txt"), sep='\t', dtype=str
    )
    df_lcfs_flipturn = pd.read_csv(
        os.path.join(data_folder, "flipturn_data.csv"), dtype=str
    )
    df_evseid_sn = pd.read_csv(
        os.path.join(data_folder, "evseid_sn_data.csv"), dtype=str
    )
    df_registered_fse = pd.read_csv(
        os.path.join(data_folder, "Registered FSE.csv"), dtype=str
    )
    output_dir = Path("reports/LCFS")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Transformation for Powerflex
    df_lcfs_pf['Interval start'] = df_lcfs_pf['Interval start'].str.replace(' PST', '', case=False, regex=False)
    df_lcfs_pf['Interval start'] = df_lcfs_pf['Interval start'].str.replace(' PDT', '', case=False, regex=False)
    df_lcfs_pf['Interval start'] = pd.to_datetime(
        df_lcfs_pf['Interval start'], 
        format='%m-%d-%Y %H:%M:%S', 
        errors='coerce'
    )    # Extract the hour
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
    print(df_combined.head())
    df_combined.to_csv(os.path.join(output_dir, "combined_lcfs_data.csv"), index=True)
    carbon_intensity = pd.read_excel(os.path.join(data_folder, "ca_carbon_intensity_values.xlsx"))
    ci_series = carbon_intensity.iloc[:, 1]
    ci_series.index = df_combined.index  # force alignment
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

    # Or add the method as a row in the optimized credits dataframe
    df_optimized_with_method = pd.concat([df_optimized_credits, df_method_used])
    timestamp = datetime.now().strftime('%Y%m%d')
    df_allsmart.to_csv(output_dir / "all_smart_charging.csv", index=False)
    df_allgridavg.to_csv(output_dir / "all_grid_average.csv", index=False)
    df_smart_wins.to_csv(output_dir / f"interval_sessions_smartcharging_lcfs_{timestamp}.csv")
    df_gridavg_wins.to_csv(output_dir / f"interval_sessions_gridavg_{timestamp}.csv")
    df_optimized_with_method.to_csv(output_dir / f"optimized_credits_with_method_{timestamp}.csv")
    print(f"Total credits (smart charging): {df_allsmart.sum().sum():,.2f}")
    print(f"Total credits (grid average): {df_allgridavg.sum().sum():,.2f}")
    print(f"Total credits (optimized): {df_optimized_credits.sum().sum():,.2f}")
    print(f"Optimization benefit: {(df_optimized_credits.sum().sum() - df_allgridavg.sum().sum()):,.2f}")
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    data_folder = BASE_DIR / "Data"
    lcfs_filing_manual(
        data_folder=data_folder,
        ci_benchmark=75.16,
        grid_avg=65.07,
        eer_value=3.4,
        energy_density=3.6
    )