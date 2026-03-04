"""Analytics module for route KPI computation."""

import pandas as pd


def compute_route_kpis(orders_df, threshold=7, group_by_col='State/Province'):
    """
    Compute KPIs for routes: Factory -> group_by_col.

    Returns an aggregated DataFrame sorted by efficiency score.
    """
    if orders_df.empty:
        return pd.DataFrame()

    # Columns for routing KPIs: Factory and group_by_col
    route_cols = ['FACTORY', group_by_col]

    missing_cols = [
        col for col in route_cols + ['lead_time_days']
        if col not in orders_df.columns
    ]
    if missing_cols:
        raise ValueError(
            f"Missing required columns for routing KPIs: {missing_cols}"
        )

    grouped = orders_df.groupby(route_cols)

    agg_df = grouped.agg(  # Performs multiple calculations per route.
        route_volume=('Order ID', 'count'),
        avg_lead_time=('lead_time_days', 'mean'),
        lead_time_std=('lead_time_days', 'std'),
        delayed_orders=('lead_time_days', lambda x: (x > threshold).sum())
    ).reset_index()

    agg_df['delay_frequency'] = (
        agg_df['delayed_orders'] / agg_df['route_volume']
    ) * 100

    agg_df['lead_time_std'] = agg_df['lead_time_std'].fillna(0)

    max_lead_time = agg_df['avg_lead_time'].max()
    max_delay_freq = agg_df['delay_frequency'].max()

    norm_lead = (  # Normalizes lead time to a 0–1 scale
        agg_df['avg_lead_time'] / max_lead_time
        if max_lead_time > 0 else 0
    )
    norm_delay = (  # Normalizes delay frequency to a 0–1 scale
        agg_df['delay_frequency'] / max_delay_freq
        if max_delay_freq > 0 else 0
    )

    # Computes the final efficiency score (higher = better)
    agg_df['route_efficiency_score'] = (
        (1 - norm_lead) + (1 - norm_delay)
    ) / 2

    # Sort by efficiency score descending
    agg_df = agg_df.sort_values(
        by='route_efficiency_score', ascending=False
    ).reset_index(drop=True)

    return agg_df


def merge_factory_data(orders_df, product_factories_df):
    """Merge factory information into orders based on Division."""
    return pd.merge(
        orders_df, product_factories_df,
        left_on='Division', right_on='DIVISION', how='left'
    )


def prepare_and_save_kpis(orders_path, mapping_path, output_path):
    """Load processed orders, compute route KPIs, and save to CSV."""
    orders = pd.read_csv(orders_path)
    mapping = pd.read_csv(mapping_path)

    orders_with_factory = merge_factory_data(orders, mapping)  # Merge

    kpi_df = compute_route_kpis(
        orders_with_factory, threshold=7, group_by_col='State/Province'
    )

    kpi_df.to_csv(output_path, index=False)
    print(f"Saved aggregated KPIs to {output_path}")


if __name__ == "__main__":
    import os
    PROJECT_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
    RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

    prepare_and_save_kpis(
        os.path.join(PROCESSED_DIR, "cleaned_orders.csv"),
        os.path.join(RAW_DIR, "product_factories.csv"),
        os.path.join(PROCESSED_DIR, "routes_aggregated.csv")
    )
