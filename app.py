from __future__ import annotations

import numpy as np
import pandas as pd
import altair as alt
import folium
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium


st.set_page_config(
    page_title="Chennai Dark Store Optimizer",
    page_icon="DS",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    :root {
        --canvas: #c8d2cc;
        --ink: #1b292b;
        --muted: #667578;
        --line: #c4cec8;
        --signal: #a4ba39;
    }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 50% 0%, rgba(255, 255, 255, 0.32), transparent 58%),
            linear-gradient(135deg, #d6ded9 0%, #c3cec7 52%, #b9c6bf 100%);
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1680px; padding-top: 1.35rem; padding-bottom: 2rem; }
    h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
    h1 { font-size: 1.85rem !important; font-weight: 760 !important; }
    h2 { font-size: 1.15rem !important; font-weight: 700 !important; }
    h3 { font-size: 1rem !important; font-weight: 700 !important; }
    .title-banner {
        margin: 0 0 0.35rem;
        padding: 0.8rem 1rem;
        background: linear-gradient(110deg, #fbfcfa 0%, #f1f5f1 100%);
        border: 1px solid #aebbb3;
        border-top: 4px solid var(--signal);
        border-radius: 6px;
        text-align: center;
        box-shadow:
            inset 0 1px 0 rgba(255, 255, 255, 0.95),
            0 2px 0 rgba(26, 42, 38, 0.08),
            0 10px 24px rgba(26, 42, 38, 0.16);
    }
    .title-banner h1 {
        margin: 0;
        color: #1b292b;
        text-align: center;
    }
    .scenario-table-heading {
        margin: 0.25rem 0 0.5rem;
        color: #132125;
        font-family: sans-serif;
        font-size: 1rem;
        font-weight: 720;
    }
    .scenario-heading {
        margin: 0.4rem 0 0.25rem;
        color: var(--ink);
        font-size: 1.25rem;
        font-weight: 700;
        line-height: 1.35;
    }
    [data-testid="stSidebar"] {
        background: #c4cec8;
        border-right: 1px solid #aebbb3;
    }
    [data-testid="stSidebar"] h2 { font-size: 1.05rem !important; }
    [data-testid="stMetric"] {
        min-height: 100px;
        padding: 13px 15px;
        background: #fbfcfa;
        border: 1px solid var(--line);
        border-left: 3px solid var(--signal);
        border-radius: 5px;
        box-shadow: 0 3px 10px rgba(26, 42, 38, 0.1);
    }
    [data-testid="stMetricLabel"] p { color: var(--muted); font-size: 0.76rem; }
    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-size: 1.3rem !important;
        font-weight: 720;
        line-height: 1.2;
        overflow-wrap: anywhere;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #293438;
        border-radius: 4px;
        overflow: hidden;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stNumberInput"] [data-baseweb="input"] {
        border-color: #b8c3bc;
        border-radius: 3px;
    }
    [data-testid="stExpander"] {
        border: 1px solid var(--line);
        border-radius: 3px;
        background: rgba(255, 255, 255, 0.5);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

CITY_CENTER = (13.0475, 80.2090)
EARTH_RADIUS_KM = 6371.0088
SCENARIO_BUDGETS = (2, 5, 10, 20, 25)
STORE_COLORS = [
    "#087f5b", "#e67700", "#1971c2", "#c2255c", "#6741d9",
    "#0b7285", "#a61e4d", "#5c940d", "#d9480f", "#364fc7",
]
CHENNAI_LAND_POLYGON = np.array(
    [
        [13.182, 80.265],
        [13.180, 80.205],
        [13.150, 80.125],
        [13.080, 80.095],
        [13.000, 80.095],
        [12.930, 80.155],
        [12.910, 80.235],
        [12.905, 80.300],
        [12.955, 80.335],
        [13.045, 80.338],
        [13.125, 80.330],
        [13.180, 80.318],
    ]
)


def point_in_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool:
    x, y = point
    inside = False
    n_points = len(polygon)
    for idx, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(idx + 1) % n_points]
        if (y1 > y) != (y2 > y):
            x_intersection = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_intersection:
                inside = not inside
    return inside


def _clip_polygon_to_convex_polygon(
    subject: list[tuple[float, float]],
    clipper: np.ndarray,
) -> list[tuple[float, float]]:
    signed_area = sum(
        clipper[index, 0] * clipper[(index + 1) % len(clipper), 1]
        - clipper[(index + 1) % len(clipper), 0] * clipper[index, 1]
        for index in range(len(clipper))
    )
    if abs(signed_area) <= 1e-12:
        raise ValueError("The land boundary must have a non-zero area.")
    orientation = 1.0 if signed_area > 0.0 else -1.0
    clipped = subject

    for edge_index, edge_start in enumerate(clipper):
        edge_end = clipper[(edge_index + 1) % len(clipper)]
        edge = edge_end - edge_start

        def signed_distance(point: tuple[float, float]) -> float:
            relative = np.asarray(point) - edge_start
            return orientation * float(edge[0] * relative[1] - edge[1] * relative[0])

        if not clipped:
            break

        output: list[tuple[float, float]] = []
        previous = clipped[-1]
        previous_distance = signed_distance(previous)
        for current in clipped:
            current_distance = signed_distance(current)
            previous_inside = previous_distance >= -1e-12
            current_inside = current_distance >= -1e-12

            if previous_inside != current_inside:
                fraction = previous_distance / (previous_distance - current_distance)
                intersection = (
                    previous[0] + fraction * (current[0] - previous[0]),
                    previous[1] + fraction * (current[1] - previous[1]),
                )
                output.append(intersection)
            if current_inside:
                output.append(current)

            previous = current
            previous_distance = current_distance
        clipped = output

    return clipped


def clip_service_radius_to_land(
    latitude: float,
    longitude: float,
    max_distance_km: float,
) -> list[tuple[float, float]]:
    if max_distance_km <= 0.0:
        raise ValueError("Maximum service distance must be greater than zero.")

    radius_m = max_distance_km * 1000.0
    earth_radius_m = EARTH_RADIUS_KM * 1000.0
    latitude_radians = np.radians(latitude)
    angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    circle = [
        (
            latitude + np.degrees(radius_m * np.cos(angle) / earth_radius_m),
            longitude
            + np.degrees(
                radius_m * np.sin(angle)
                / (earth_radius_m * np.cos(latitude_radians))
            ),
        )
        for angle in angles
    ]
    return _clip_polygon_to_convex_polygon(circle, CHENNAI_LAND_POLYGON)


def generate_land_points(
    rng: np.random.Generator,
    point_count: int,
    zone_weights: np.ndarray,
    polygon: np.ndarray,
    x_scale: float,
    y_scale: float,
) -> np.ndarray:
    points = np.empty((point_count, 2), dtype=np.float64)
    filled = 0
    attempts = 0
    while filled < point_count and attempts < 50_000:
        attempts += 1
        zone_index = int(rng.choice(len(zone_weights), p=zone_weights))
        point = ZONES[zone_index] + rng.normal(loc=0.0, scale=(x_scale, y_scale))
        if not point_in_polygon((point[0], point[1]), polygon):
            continue
        points[filled] = point
        filled += 1
    if filled != point_count:
        raise RuntimeError("Unable to generate enough land-constrained points for the scenario.")
    return points


ZONES = np.array(
    [
        [13.165, 80.301],  # Tiruvottiyur
        [13.120, 80.231],  # Madhavaram
        [13.095, 80.209],  # Anna Nagar
        [13.080, 80.268],  # Perambur
        [13.065, 80.260],  # Egmore
        [13.041, 80.234],  # T Nagar
        [13.012, 80.221],  # Saidapet
        [13.035, 80.159],  # Porur
        [12.978, 80.220],  # Velachery
        [12.958, 80.247],  # Thoraipakkam
        [12.925, 80.131],  # Tambaram
        [12.915, 80.238],  # Sholinganallur
    ],
    dtype=np.float64,
)
ZONE_WEIGHTS = np.array([0.06, 0.06, 0.10, 0.08, 0.08, 0.14, 0.07, 0.08, 0.12, 0.08, 0.06, 0.07])
REVENUE_PER_ORDER_RS = 500.0
VARIABLE_COST_PER_ORDER_RS = 445.0
DELIVERY_COST_PER_KM_RS = 5.0
STORE_SETUP_COST_RS = 600_000.0
STORE_OPERATING_COST_PER_DAY_RS = 5_000.0
SETUP_RECOVERY_YEARS = 3


def get_scenario_budgets(max_stores: int) -> tuple[int, ...]:
    budgets = {budget for budget in SCENARIO_BUDGETS if budget < max_stores}
    budgets.add(max_stores)
    return tuple(sorted(budgets))


@st.cache_data(show_spinner=False)
def generate_data(candidate_count: int, customer_count: int, seed: int):
    rng = np.random.default_rng(seed)
    candidate_points = generate_land_points(
        rng=rng,
        point_count=candidate_count,
        zone_weights=ZONE_WEIGHTS,
        polygon=CHENNAI_LAND_POLYGON,
        x_scale=0.010,
        y_scale=0.012,
    )
    customer_points = generate_land_points(
        rng=rng,
        point_count=customer_count,
        zone_weights=ZONE_WEIGHTS,
        polygon=CHENNAI_LAND_POLYGON,
        x_scale=0.014,
        y_scale=0.016,
    )

    customers = pd.DataFrame(
        {
            "customer_id": [f"C{i + 1:05d}" for i in range(customer_count)],
            "latitude": customer_points[:, 0],
            "longitude": customer_points[:, 1],
            "orders_per_day": np.clip(rng.gamma(2.0, 0.55, customer_count), 0.15, 4.0),
        }
    )
    candidates = pd.DataFrame(
        {
            "store_id": [f"S{i + 1:03d}" for i in range(candidate_count)],
            "latitude": candidate_points[:, 0],
            "longitude": candidate_points[:, 1],
        }
    )
    return candidates, customers


def haversine_matrix(customers: pd.DataFrame, candidates: pd.DataFrame) -> np.ndarray:
    customer_lat = np.radians(customers["latitude"].to_numpy())[:, None]
    customer_lon = np.radians(customers["longitude"].to_numpy())[:, None]
    candidate_lat = np.radians(candidates["latitude"].to_numpy())[None, :]
    candidate_lon = np.radians(candidates["longitude"].to_numpy())[None, :]

    lat_delta = candidate_lat - customer_lat
    lon_delta = candidate_lon - customer_lon
    haversine = (
        np.sin(lat_delta / 2.0) ** 2
        + np.cos(customer_lat) * np.cos(candidate_lat) * np.sin(lon_delta / 2.0) ** 2
    )
    return (2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))).astype(
        np.float32
    )


def greedy_store_sequence(
    distances_km: np.ndarray,
    orders_per_day: np.ndarray,
    max_distance_km: float,
    margin_per_order_rs: float,
    delivery_cost_per_km_rs: float,
    daily_store_cost_rs: float,
    max_stores: int,
) -> tuple[list[int], np.ndarray]:
    per_order_profit = margin_per_order_rs - delivery_cost_per_km_rs * distances_km
    eligible = (distances_km <= max_distance_km) & (per_order_profit > 0.0)
    profit = np.where(eligible, per_order_profit * orders_per_day[:, None], 0.0).astype(
        np.float32
    )

    best_profit = np.zeros(profit.shape[0], dtype=np.float32)
    remaining = np.ones(profit.shape[1], dtype=bool)
    opened: list[int] = []

    for _ in range(min(max_stores, profit.shape[1])):
        candidate_indexes = np.flatnonzero(remaining)
        gains = np.maximum(profit[:, candidate_indexes], best_profit[:, None]) - best_profit[:, None]
        candidate_gains = gains.sum(axis=0) - daily_store_cost_rs
        best_position = int(np.argmax(candidate_gains))
        best_gain = float(candidate_gains[best_position])
        if best_gain <= 1e-6:
            break

        chosen = int(candidate_indexes[best_position])
        opened.append(chosen)
        remaining[chosen] = False
        best_profit = np.maximum(best_profit, profit[:, chosen])

    return opened, profit


def evaluate_budget(
    opened: list[int],
    profit: np.ndarray,
    distances_km: np.ndarray,
    daily_store_cost_rs: float,
) -> dict:
    if not opened:
        assignments = np.full(profit.shape[0], -1, dtype=np.int32)
        assignment_profit = np.zeros(profit.shape[0], dtype=np.float32)
        assigned_distance = np.full(profit.shape[0], np.nan, dtype=np.float32)
    else:
        selected_profit = profit[:, opened]
        local_assignment = selected_profit.argmax(axis=1)
        assignment_profit = selected_profit.max(axis=1)
        assignments = np.asarray(opened, dtype=np.int32)[local_assignment]
        unserved = assignment_profit <= 0.0
        assignments[unserved] = -1
        assignment_profit[unserved] = 0.0
        rows = np.arange(profit.shape[0])
        assigned_distance = distances_km[rows, np.maximum(assignments, 0)]
        assigned_distance[assignments < 0] = np.nan

    return {
        "opened": opened,
        "assignments": assignments,
        "assignment_profit": assignment_profit,
        "assigned_distance": assigned_distance,
        "profit_per_day": float(assignment_profit.sum() - len(opened) * daily_store_cost_rs),
    }


def summarize_store_performance(
    candidates: pd.DataFrame,
    customers: pd.DataFrame,
    solution: dict,
    operating_cost_per_day_rs: float,
    setup_recovery_per_day_rs: float,
) -> pd.DataFrame:
    rows = []
    assignments = solution["assignments"]
    for store_index in solution["opened"]:
        assigned = assignments == store_index
        served_distances = solution["assigned_distance"][assigned]
        contribution_after_delivery = float(solution["assignment_profit"][assigned].sum())
        operating_cost = operating_cost_per_day_rs
        setup_recovery = setup_recovery_per_day_rs
        store = candidates.iloc[store_index]
        rows.append(
            {
                "Store": store["store_id"],
                "Latitude": float(store["latitude"]),
                "Longitude": float(store["longitude"]),
                "Customers served": int(assigned.sum()),
                "Orders / day": float(customers.loc[assigned, "orders_per_day"].sum()),
                "Avg. distance (km)": (
                    float(served_distances.mean()) if len(served_distances) else np.nan
                ),
                "Contribution after delivery (Rs/day)": contribution_after_delivery,
                "Operating cost (Rs/day)": operating_cost,
                "Setup recovery (Rs/day)": setup_recovery,
                "Estimated net profit (Rs/day)": (
                    contribution_after_delivery - operating_cost - setup_recovery
                ),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Optimizing candidate locations...")
def solve_scenarios(
    candidates: pd.DataFrame,
    customers: pd.DataFrame,
    max_distance_km: float,
    margin_per_order_rs: float,
    delivery_cost_per_km_rs: float,
    daily_store_cost_rs: float,
    max_stores: int,
):
    distances = haversine_matrix(customers, candidates)
    sequence, profit = greedy_store_sequence(
        distances,
        customers["orders_per_day"].to_numpy(dtype=np.float32),
        max_distance_km,
        margin_per_order_rs,
        delivery_cost_per_km_rs,
        daily_store_cost_rs,
        max_stores,
    )

    results = {}
    for budget in get_scenario_budgets(max_stores):
        opened = sequence[: min(budget, len(candidates))]
        results[budget] = evaluate_budget(opened, profit, distances, daily_store_cost_rs)
    return results


def build_map(
    candidates: pd.DataFrame,
    customers: pd.DataFrame,
    solution: dict,
    max_distance_km: float,
    seed: int,
) -> folium.Map:
    map_view = folium.Map(
        location=CITY_CENTER,
        zoom_start=11,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    customer_layer = folium.FeatureGroup(name="Customer demand heatmap", show=True)
    heat_data = customers[["latitude", "longitude", "orders_per_day"]].to_numpy().tolist()
    HeatMap(
        heat_data,
        radius=13,
        blur=17,
        min_opacity=0.18,
        name="Demand",
        gradient={
            0.2: "#2b83ba",
            0.4: "#abdda4",
            0.6: "#ffffbf",
            0.8: "#fdae61",
            1.0: "#d7191c",
        },
    ).add_to(customer_layer)
    customer_layer.add_to(map_view)

    assignments = solution["assignments"]
    assigned_indexes = np.flatnonzero(assignments >= 0)
    rng = np.random.default_rng(seed + 1)
    if len(assigned_indexes) > 700:
        assigned_indexes = np.sort(rng.choice(assigned_indexes, size=700, replace=False))

    assignment_layer = folium.FeatureGroup(
        name=f"Sampled customer assignments ({len(assigned_indexes):,})", show=False
    )
    for customer_index in assigned_indexes:
        store_index = int(assignments[customer_index])
        store = candidates.iloc[store_index]
        customer = customers.iloc[customer_index]
        color = STORE_COLORS[store_index % len(STORE_COLORS)]
        folium.PolyLine(
            locations=[
                (customer["latitude"], customer["longitude"]),
                (store["latitude"], store["longitude"]),
            ],
            color=color,
            weight=1.1,
            opacity=0.48,
            tooltip=f"{customer['customer_id']} to {store['store_id']}",
        ).add_to(assignment_layer)
        folium.CircleMarker(
            location=(customer["latitude"], customer["longitude"]),
            radius=2.2,
            color=color,
            fill=True,
            fill_opacity=0.65,
            weight=0,
        ).add_to(assignment_layer)
    assignment_layer.add_to(map_view)

    service_layer = folium.FeatureGroup(name="Maximum land service area", show=False)
    open_indexes = set(solution["opened"])
    for store_index in open_indexes:
        store = candidates.iloc[store_index]
        color = STORE_COLORS[store_index % len(STORE_COLORS)]
        service_area = clip_service_radius_to_land(
            float(store["latitude"]),
            float(store["longitude"]),
            max_distance_km,
        )
        if len(service_area) < 3:
            raise RuntimeError(
                f"Could not construct a land-constrained service area for {store['store_id']}."
            )
        folium.Polygon(
            locations=service_area,
            color=color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.08,
            opacity=0.42,
            tooltip=f"{store['store_id']} maximum service distance on land: {max_distance_km:g} km",
        ).add_to(service_layer)
    service_layer.add_to(map_view)

    candidate_layer = folium.FeatureGroup(name="Candidate locations", show=True)
    for store_index, store in candidates.iterrows():
        is_open = store_index in open_indexes
        color = STORE_COLORS[store_index % len(STORE_COLORS)] if is_open else "#687078"
        status = "OPEN" if is_open else "not selected"
        folium.CircleMarker(
            location=(store["latitude"], store["longitude"]),
            radius=8 if is_open else 5,
            color="white",
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.95,
            tooltip=f"{store['store_id']} | {status}",
            popup=f"<b>{store['store_id']}</b><br>Status: {status}",
        ).add_to(candidate_layer)
    candidate_layer.add_to(map_view)
    folium.LayerControl(collapsed=False).add_to(map_view)
    map_text_styles = """
    <style>
    .leaflet-control-layers label,
    .leaflet-control-attribution {
        font: 12px/1.35 Arial, sans-serif !important;
    }
    </style>
    """
    map_view.get_root().html.add_child(folium.Element(map_text_styles))
    legend_html = """
        <div style="position: fixed; bottom: 28px; right: 14px; left: auto; z-index: 9999;
                                width: 185px; padding: 8px 9px; background: rgba(250, 251, 249, 0.96);
                                border: 1px solid #cbd3cd; border-left: 3px solid #a4ba39;
                                border-radius: 3px; color: #263437; font: 11px/1.3 Arial, sans-serif;
                                box-shadow: 0 2px 8px rgba(21, 35, 32, 0.18);">
            <div style="font-size: 11px; font-weight: 700; margin-bottom: 5px;">MAP KEY</div>
            <div style="margin-bottom: 2px;">Demand density (orders/day)</div>
            <div style="height: 7px; border-radius: 2px; margin-bottom: 2px;
                                    background: linear-gradient(90deg, #2b83ba, #abdda4, #ffffbf, #fdae61, #d7191c);"></div>
            <div style="display: flex; justify-content: space-between; color: #667578; font-size: 10px;">
                <span>Lower</span><span>Higher</span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; margin-top: 5px;">
                <span style="width: 16px; height: 3px; background: linear-gradient(90deg, #087f5b, #e67700, #1971c2, #c2255c);"></span>
                <span>Assignment color = store</span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; margin-top: 4px;">
                <span style="width: 8px; height: 8px; border-radius: 50%; background: linear-gradient(135deg, #087f5b, #1971c2); border: 1px solid white;"></span>
                <span>Opened store</span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; margin-top: 4px;">
                <span style="width: 8px; height: 8px; border-radius: 50%; background: #687078; border: 1px solid white;"></span>
                <span>Candidate not selected</span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; margin-top: 4px;">
                <span style="width: 10px; height: 10px; border: 1px solid #687078; border-radius: 50%;"></span>
                <span>Maximum service distance (land only)</span>
            </div>
        </div>
        """
    map_view.get_root().html.add_child(folium.Element(legend_html))
    return map_view


st.markdown(
    '<div class="title-banner"><h1>Chennai Dark Store Location Optimizer</h1></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Scenario inputs")
    candidate_count = st.number_input(
        "Number of candidate dark store locations",
        min_value=10,
        max_value=100,
        value=50,
        step=1,
    )
    max_stores_allowed = st.number_input(
        "Maximum stores allowed to open",
        min_value=1,
        max_value=int(candidate_count),
        value=min(25, int(candidate_count)),
        step=1,
    )
    customer_count = st.number_input(
        "Number of customers",
        min_value=500,
        max_value=20000,
        value=10000,
        step=100,
    )
    max_distance_km = st.slider("Maximum customer distance (km)", 1.0, 15.0, 5.0, step=0.5)
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
    st.divider()
    st.header("Model assumptions")
    revenue_per_order = st.number_input(
        "Revenue per order (Rs)",
        min_value=0.0,
        max_value=5000.0,
        value=REVENUE_PER_ORDER_RS,
        step=10.0,
    )
    variable_cost_per_order = st.number_input(
        "Product and other variable costs (Rs/order)",
        min_value=0.0,
        max_value=5000.0,
        value=VARIABLE_COST_PER_ORDER_RS,
        step=10.0,
    )
    delivery_cost_per_km = st.number_input(
        "Delivery cost (Rs/order/km)",
        min_value=0.0,
        max_value=100.0,
        value=DELIVERY_COST_PER_KM_RS,
        step=0.5,
    )
    store_operating_cost_per_day = st.number_input(
        "Store operating cost (Rs/store/day)",
        min_value=0.0,
        max_value=100_000.0,
        value=STORE_OPERATING_COST_PER_DAY_RS,
        step=500.0,
    )
    store_setup_cost = st.number_input(
        "Store setup cost (Rs/store)",
        min_value=0.0,
        max_value=50_000_000.0,
        value=STORE_SETUP_COST_RS,
        step=50_000.0,
    )
    setup_recovery_years = st.number_input(
        "Setup cost recovery period (years)",
        min_value=1,
        max_value=20,
        value=SETUP_RECOVERY_YEARS,
        step=1,
    )
    margin_per_order = revenue_per_order - variable_cost_per_order
    daily_setup_cost_per_store = store_setup_cost / (setup_recovery_years * 365)
    daily_store_cost = store_operating_cost_per_day + daily_setup_cost_per_store
    st.divider()
    st.caption("How the estimates are calculated")
    st.markdown(
        f"""
        - **Per-order economics:** Rs {revenue_per_order:,.2f} revenue minus Rs
          {variable_cost_per_order:,.2f} variable costs = Rs {margin_per_order:,.2f}
          contribution before delivery.
        - **Delivery:** Rs {delivery_cost_per_km:,.2f} per order per km.
        - **Daily cost per store:** Rs {daily_store_cost:,.2f}, including Rs
          {store_operating_cost_per_day:,.2f} operating cost and Rs
          {daily_setup_cost_per_store:,.2f} setup-cost recovery.
        - Setup cost is spread evenly over {setup_recovery_years} years (no interest
          or inflation).
        - Revenue, variable costs, delivery, operating costs, and setup costs are
          illustrative assumptions; adjust them above to match your estimates.
        - **Demand and locations:** synthetic estimates around Chennai, not observed
          customer or store data.
        - **Customer demand:** generated orders average about 1.1 per day and are
          limited to 0.15–4 orders per customer per day.
        - **Distance:** straight-line distance, not road or travel distance.
        - **Optimization:** greedy profit optimization with a strict customer
          service-distance limit.
        - The model does not include taxes, competition, or changes in demand when
          assumptions change. Reported profit is an estimate, not a financial forecast.
        - **Location selection:** a greedy heuristic; it may not find the globally
          best store locations.
        """
    )

with st.spinner("Generating Chennai demand and solving store scenarios..."):
    candidate_data, customer_data = generate_data(
        candidate_count, customer_count, int(seed)
    )
    scenarios = solve_scenarios(
        candidate_data,
        customer_data,
        max_distance_km,
        margin_per_order,
        delivery_cost_per_km,
        daily_store_cost,
        int(max_stores_allowed),
    )

scenario_rows = []
for budget, solution in scenarios.items():
    assigned = solution["assignments"] >= 0
    distances = solution["assigned_distance"][assigned]
    scenario_rows.append(
        {
            "Store budget": budget,
            "Stores opened": len(solution["opened"]),
            "Customers served": int(assigned.sum()),
            "Coverage (%)": 100.0 * float(assigned.mean()),
            "Avg. distance (km)": float(distances.mean()) if len(distances) else 0.0,
            "Estimated net profit (Rs/day)": solution["profit_per_day"],
        }
    )
scenario_table = pd.DataFrame(scenario_rows)

st.markdown(
    '<div class="scenario-heading">Map and KPI scenario</div>',
    unsafe_allow_html=True,
)
scenario_choice = st.selectbox(
    "Map and KPI scenario",
    options=list(scenarios),
    index=min(
        range(len(scenarios)),
        key=lambda index: abs(list(scenarios)[index] - min(10, max_stores_allowed)),
    ),
    format_func=lambda budget: f"Up to {budget} store{'s' if budget != 1 else ''}",
    label_visibility="collapsed",
)
chosen = scenarios[scenario_choice]
served = chosen["assignments"] >= 0
served_distances = chosen["assigned_distance"][served]
coverage = 100.0 * float(served.mean())

kpi_columns = st.columns(4)
kpi_columns[0].metric("Estimated net profit / day", f"Rs {chosen['profit_per_day']:,.0f}")
kpi_columns[1].metric(
    "Customers served",
    f"{int(served.sum()):,}",
    help=f"Out of {customer_count:,} modeled customers",
)
kpi_columns[2].metric("Coverage", f"{coverage:.1f}%")
kpi_columns[3].metric(
    "Mean distance",
    f"{served_distances.mean():.2f} km" if len(served_distances) else "N/A",
)

if len(chosen["opened"]) < min(scenario_choice, candidate_count):
    st.info("The optimizer opened fewer stores because no remaining location added positive profit.")
elif scenario_choice == max_stores_allowed:
    st.info(f"The selected scenario allows up to {max_stores_allowed} stores to open.")

st.subheader("Chennai assignment map")
st.caption(
    "Heat intensity includes all customers. Connections show a reproducible sample of "
    "up to 700 assignments; toggle layers in the map to inspect candidates and land-only "
    "maximum service areas."
)
solution_map = build_map(
    candidate_data, customer_data, chosen, max_distance_km, int(seed)
)
st_folium(solution_map, height=560, use_container_width=True, returned_objects=[])

profit_column, coverage_column = st.columns(2, gap="large")
with profit_column:
    st.subheader("Estimated net profit by store budget")
    profit_chart = (
        alt.Chart(scenario_table)
        .mark_bar(color="#087f5b")
        .encode(
            x=alt.X(
                "Store budget:O",
                sort="ascending",
                title="Store budget (up to)",
                axis=alt.Axis(labelColor="#293538", titleColor="#293538"),
            ),
            y=alt.Y(
                "Estimated net profit (Rs/day):Q",
                title="Estimated net profit (Rs/day)",
                axis=alt.Axis(
                    format=",.0f",
                    labelColor="#293538",
                    titleColor="#293538",
                ),
            ),
            tooltip=[
                alt.Tooltip("Store budget:O", title="Store budget (up to)"),
                alt.Tooltip("Stores opened:Q"),
                alt.Tooltip("Estimated net profit (Rs/day):Q", format=",.0f"),
            ],
        )
        .properties(height=280)
        .configure_axis(
            gridColor="#d2d9d4",
            gridOpacity=0.85,
            tickColor="#73817d",
            domainColor="#73817d",
            labelFontSize=14,
            titleFontSize=15,
        )
        .configure_view(strokeOpacity=0)
    )
    st.altair_chart(profit_chart, width="stretch")

with coverage_column:
    st.subheader("Customer coverage by store budget")
    coverage_chart = (
        alt.Chart(scenario_table)
        .mark_line(
            color="#b64d2d",
            point=alt.OverlayMarkDef(filled=True, size=65),
            strokeWidth=3,
        )
        .encode(
            x=alt.X(
                "Store budget:O",
                sort="ascending",
                title="Store budget (up to)",
                axis=alt.Axis(labelColor="#293538", titleColor="#293538"),
            ),
            y=alt.Y(
                "Coverage (%):Q",
                title="Customers served (%)",
                scale=alt.Scale(domain=[0, 150], nice=False),
                axis=alt.Axis(
                    format=".0f",
                    tickCount=4,
                    labelColor="#293538",
                    titleColor="#293538",
                ),
            ),
            tooltip=[
                alt.Tooltip("Store budget:O", title="Store budget (up to)"),
                alt.Tooltip("Customers served:Q", format=",.0f"),
                alt.Tooltip("Coverage (%):Q", format=".1f"),
                alt.Tooltip("Avg. distance (km):Q", format=".2f", title="Mean distance (km)"),
            ],
        )
        .properties(height=230)
        .configure_axis(
            gridColor="#d2d9d4",
            gridOpacity=0.85,
            tickColor="#73817d",
            domainColor="#73817d",
            labelFontSize=14,
            titleFontSize=15,
        )
        .configure_view(strokeOpacity=0)
    )
    st.altair_chart(coverage_chart, width="stretch")

display_scenario_table = scenario_table.rename(
    columns={
        "Store budget": "Budget",
        "Stores opened": "Open",
        "Customers served": "Served",
        "Coverage (%)": "Coverage",
        "Avg. distance (km)": "Mean km",
        "Estimated net profit (Rs/day)": "Net profit / day",
    }
)
styled_scenario_table = display_scenario_table.style.apply(
    lambda row: [
        "background-color: #f7f9f6; color: #253236; border-color: #d2d9d4"
        if row.name % 2 == 0
        else "background-color: #e8ede8; color: #253236; border-color: #d2d9d4"
        for _ in row
    ],
    axis=1,
).format(
    {
        "Budget": "{:.0f}",
        "Open": "{:,.0f}",
        "Served": "{:,.0f}",
        "Coverage": "{:.1f}%",
        "Mean km": "{:.2f} km",
        "Net profit / day": lambda value: f"Rs {value:,.0f}",
    }
).set_table_styles(
    [
        {
            "selector": "th",
            "props": [
                ("background-color", "#263337"),
                ("color", "#f4f6f2"),
                ("font-weight", "700"),
                ("font-size", "14px"),
                ("font-family", "sans-serif"),
                ("text-align", "center"),
                ("border-color", "#d2d9d4"),
                ("border-bottom", "2px solid #a4ba39"),
            ],
        },
        {
            "selector": "td",
            "props": [
                ("padding", "0.55rem 0.45rem"),
                ("font-size", "14px"),
                ("font-family", "sans-serif"),
                ("text-align", "center"),
                ("vertical-align", "middle"),
                ("border-bottom", "1px solid #d2d9d4"),
            ],
        },
        {
            "selector": "table",
            "props": [
                ("width", "100%"),
                ("table-layout", "fixed"),
                ("border-collapse", "separate"),
                ("border-spacing", "0"),
                ("border", "1px solid #d2d9d4"),
                ("border-radius", "4px"),
            ],
        },
    ]
).set_table_attributes('style="width: 100%; table-layout: fixed;"').hide(axis="index")
st.subheader("Scenario comparison")
st.html(styled_scenario_table.to_html(), width="stretch")
st.download_button(
    "Download scenario data (CSV)",
    data=scenario_table.to_csv(index=False),
    file_name="chennai_store_scenarios.csv",
    mime="text/csv",
)
st.subheader("Store-level contribution and outcomes")
store_performance = summarize_store_performance(
    candidate_data,
    customer_data,
    chosen,
    store_operating_cost_per_day,
    daily_setup_cost_per_store,
)
if store_performance.empty:
    st.info("No stores were opened in this scenario.")
else:
    styled_store_performance = store_performance.style.apply(
        lambda row: [
            "background-color: #f7f9f6; color: #253236; border-color: #d2d9d4"
            if row.name % 2 == 0
            else "background-color: #e8ede8; color: #253236; border-color: #d2d9d4"
            for _ in row
        ],
        axis=1,
    ).format(
        {
            "Latitude": "{:.4f}",
            "Longitude": "{:.4f}",
            "Customers served": "{:,.0f}",
            "Orders / day": "{:,.1f}",
            "Avg. distance (km)": "{:.2f}",
            "Contribution after delivery (Rs/day)": "Rs {:,.0f}",
            "Operating cost (Rs/day)": "Rs {:,.0f}",
            "Setup recovery (Rs/day)": "Rs {:,.0f}",
            "Estimated net profit (Rs/day)": "Rs {:,.0f}",
        },
        na_rep="N/A",
    ).set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#263337"),
                    ("color", "#f4f6f2"),
                    ("font-weight", "700"),
                    ("font-size", "14px"),
                    ("font-family", "sans-serif"),
                    ("text-align", "center"),
                    ("border-color", "#d2d9d4"),
                    ("border-bottom", "2px solid #a4ba39"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("padding", "0.55rem 0.45rem"),
                    ("font-size", "14px"),
                    ("font-family", "sans-serif"),
                    ("text-align", "center"),
                    ("vertical-align", "middle"),
                    ("border-bottom", "1px solid #d2d9d4"),
                ],
            },
            {
                "selector": "table",
                "props": [
                    ("width", "100%"),
                    ("table-layout", "fixed"),
                    ("border-collapse", "separate"),
                    ("border-spacing", "0"),
                    ("border", "1px solid #d2d9d4"),
                    ("border-radius", "4px"),
                ],
            },
        ]
    ).set_table_attributes('style="width: 100%; table-layout: fixed;"').hide(axis="index")
    st.html(styled_store_performance.to_html(), width="stretch")
    st.download_button(
        "Download store-level results (CSV)",
        data=store_performance.to_csv(index=False),
        file_name="chennai_store_performance.csv",
        mime="text/csv",
    )
st.caption(
    "Contribution after delivery is the sum for customers assigned to each store, "
    "before fixed store costs. Estimated net profit subtracts that store's daily "
    "operating cost and setup-cost recovery."
)
st.caption(
    "Estimated net profit includes order contribution after distance-based delivery "
    "costs, minus daily store operating costs and setup-cost recovery. The optimizer "
    "uses a greedy marginal-profit heuristic; unprofitable or out-of-range assignments "
    "remain unserved."
)
