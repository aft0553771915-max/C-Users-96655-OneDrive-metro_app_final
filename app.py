"""
Streamlit Metro Energy & Power Explorer
- Headway sensitive
- 24:00 safe
- Time View: Specific Time / Full Day / Full Week (week = 7x day totals)
- GROQ Chatbot with chat-style UI
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta, date, time
from collections import defaultdict
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from groq import Groq
import json
from fpdf import FPDF

# ---------------------------------------------------------------------
# إعداد الصفحة
# ---------------------------------------------------------------------
st.set_page_config(page_title="Metro Energy & Power Explorer", layout="wide")

# ⚠️ IMPORTANT: مفتاح GROQ
# قراءة مفتاح GROQ من secrets أو من متغيّر بيئة
try:
    # في Streamlit Cloud نقرأ المفتاح من secrets
    groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    # لو تشغل محلي على لابتوبك نقرأه من متغير بيئة
    groq_api_key = os.getenv("GROQ_API_KEY")

groq_client = None
if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)


# =============================================================================
# Utility: Safe 24:00 handling for custom schedule periods
# =============================================================================
def make_datetime_allow_24(date_obj, hour, minute=0):
    hour = int(hour)
    minute = int(minute)
    if not (0 <= minute <= 59):
        raise ValueError(f"minute must be in 0..59, got {minute}")
    if hour < 0 or hour > 24:
        raise ValueError(
            f"hour must be in 0..24 (24 means midnight next day), got {hour}"
        )
    if hour == 24:
        return datetime.combine(date_obj + timedelta(days=1), time(0, minute))
    return datetime.combine(date_obj, time(hour, minute))


# =============================================================================
# File loading
# =============================================================================
@st.cache_data
def load_excel(excel_file):
    xls = pd.ExcelFile(excel_file)
    stations_df = pd.read_excel(xls, "Stations")
    lines_df = pd.read_excel(xls, "Lines")
    crossing_df = (
        pd.read_excel(xls, "Crossing")
        if "Crossing" in xls.sheet_names
        else pd.DataFrame()
    )

    stations_df["station"] = stations_df["station"].astype(str)
    lines_df["Station A"] = lines_df["Station A"].astype(str)
    lines_df["Station B"] = lines_df["Station B"].astype(str)

    stations_df = stations_df.drop_duplicates(subset=["station"]).reset_index(
        drop=True
    )
    return stations_df, lines_df, crossing_df


# =============================================================================
# Physics / Energy Model
# =============================================================================
def segment_energy_and_power(
    distance_km,
    grade,
    m_kg,
    eta,
    vmax_kmh,
    a_mps2,
    stop_s,
    coeff_a,
    coeff_b,
    coeff_c,
):
    d = distance_km * 1000.0
    vmax = vmax_kmh / 3.6
    g = 9.81

    F_res = coeff_a + coeff_b * vmax + coeff_c * vmax**2
    F_grade = m_kg * g * grade

    t_acc = vmax / a_mps2
    s_acc = 0.5 * a_mps2 * t_acc**2

    if 2 * s_acc > d:
        s_acc = d / 2
        t_acc = math.sqrt(2 * s_acc / a_mps2)
        t_cruise = 0
        triangular = True
    else:
        s_cruise = d - 2 * s_acc
        t_cruise = s_cruise / vmax
        triangular = False

    t_total_movement = 2 * t_acc + t_cruise
    t_total = t_total_movement + stop_s

    F_total_cruise = F_res + F_grade
    P_cruise = max(F_total_cruise * vmax / eta, 0.0)

    F_total_accel = F_res + F_grade + m_kg * a_mps2
    P_accel = max(F_total_accel * (vmax / 2.0) / eta, 0.0)
    P_decel = P_accel  # ممكن نضيف Regenerative لاحقاً

    if triangular:
        E_acc = P_accel * t_acc
        E_decel = P_decel * t_acc
        E_cruise = 0.0
    else:
        E_acc = P_accel * t_acc
        E_decel = P_decel * t_acc
        E_cruise = P_cruise * t_cruise

    E_total_j = E_acc + E_cruise + E_decel
    E_total_kwh = E_total_j / 3.6e6

    avg_power_w = E_total_j / t_total_movement if t_total_movement > 0 else 0.0

    return t_total, avg_power_w, E_total_kwh


# =============================================================================
# Timetable construction helpers
# =============================================================================
def hhmm_range(start_dt, end_dt, headway_min):
    current = start_dt
    delta = timedelta(minutes=headway_min)
    while current < end_dt:
        yield current
        current += delta


def parameter_section(line_colors):
    st.sidebar.markdown("## Train Parameters")

    mode = st.sidebar.radio(
        "Parameter mode:",
        ["Uniform (same for all lines)", "Per Line Color"],
        index=0,
    )

    default_mass = 40000
    default_eta = 0.9
    default_vmax = 80
    default_accel = 0.8
    default_stop = 30
    default_a = 10.0
    default_b = 1.5
    default_c = 0.02

    params_by_color = {}

    if mode == "Uniform (same for all lines)":
        st.sidebar.caption("All lines share the same vehicle parameters.")

        mass = st.sidebar.number_input(
            "Train mass (kg)", 10000, 200000, default_mass, step=1000
        )
        eta = st.sidebar.slider(
            "Overall efficiency (η)", 0.5, 1.0, default_eta, 0.01
        )
        vmax = st.sidebar.number_input(
            "Max speed (km/h)", 40, 140, default_vmax, step=5
        )
        accel = st.sidebar.number_input(
            "Acceleration/Deceleration (m/s²)",
            0.2,
            2.0,
            default_accel,
            step=0.1,
        )
        stop_s = st.sidebar.number_input(
            "Station dwell time (s)", 5, 120, default_stop, step=5
        )
        coeff_a = st.sidebar.number_input(
            "Rolling res. a (N)", 0.0, 100.0, default_a, step=1.0
        )
        coeff_b = st.sidebar.number_input(
            "Rolling res. b (N·s/m)", 0.0, 10.0, default_b, step=0.1
        )
        coeff_c = st.sidebar.number_input(
            "Rolling res. c (N·s²/m²)", 0.0, 0.5, default_c, step=0.01
        )

        for lc in line_colors:
            params_by_color[lc] = dict(
                mass_kg=mass,
                eta=eta,
                vmax_kmh=vmax,
                accel_mps2=accel,
                stop_s=stop_s,
                a=coeff_a,
                b=coeff_b,
                c=coeff_c,
            )
    else:
        st.sidebar.caption(
            "Adjust parameters for each line color independently."
        )
        for lc in line_colors:
            with st.sidebar.expander(
                f"Line {lc} parameters", expanded=False
            ):
                mass = st.number_input(
                    f"[{lc}] Train mass (kg)",
                    10000,
                    200000,
                    default_mass,
                    step=1000,
                    key=f"mass_{lc}",
                )
                eta = st.slider(
                    f"[{lc}] Efficiency (η)",
                    0.5,
                    1.0,
                    default_eta,
                    0.01,
                    key=f"eta_{lc}",
                )
                vmax = st.number_input(
                    f"[{lc}] Max speed (km/h)",
                    40,
                    140,
                    default_vmax,
                    step=5,
                    key=f"vmax_{lc}",
                )
                accel = st.number_input(
                    f"[{lc}] Accel/Decel (m/s²)",
                    0.2,
                    2.0,
                    default_accel,
                    step=0.1,
                    key=f"accel_{lc}",
                )
                stop_s = st.number_input(
                    f"[{lc}] Dwell time (s)",
                    5,
                    120,
                    default_stop,
                    step=5,
                    key=f"stop_{lc}",
                )
                coeff_a = st.number_input(
                    f"[{lc}] Rolling a (N)",
                    0.0,
                    100.0,
                    default_a,
                    step=1.0,
                    key=f"a_{lc}",
                )
                coeff_b = st.number_input(
                    f"[{lc}] Rolling b (N·s/m)",
                    0.0,
                    10.0,
                    default_b,
                    step=0.1,
                    key=f"b_{lc}",
                )
                coeff_c = st.number_input(
                    f"[{lc}] Rolling c (N·s²/m²)",
                    0.0,
                    0.5,
                    default_c,
                    step=0.01,
                    key=f"c_{lc}",
                )

                params_by_color[lc] = dict(
                    mass_kg=mass,
                    eta=eta,
                    vmax_kmh=vmax,
                    accel_mps2=accel,
                    stop_s=stop_s,
                    a=coeff_a,
                    b=coeff_b,
                    c=coeff_c,
                )

    return params_by_color


def schedule_section(line_colors):
    st.sidebar.markdown("## Schedule Configuration")
    st.sidebar.write(
        "Default assumption: 24-hour operation with fixed headway. "
        "You can also define different headways for peak/off-peak windows."
    )

    schedule_mode = st.sidebar.radio(
        "Schedule mode", ["24h Fixed Headway", "Custom Periods"], index=0
    )

    today = date.today()

    if schedule_mode == "24h Fixed Headway":
        st.sidebar.caption(
            "All day: from 00:00 to 24:00 with single headway per line."
        )

        per_line = st.sidebar.radio(
            "Headway style",
            ["Uniform headway for all lines", "Per-line headway"],
            index=0,
            key="fixed_hdw_style",
        )

        if per_line == "Uniform headway for all lines":
            hw = st.sidebar.number_input(
                "Headway for all lines (min)", 1, 60, 5, step=1
            )
            headways = {lc: hw for lc in line_colors}
        else:
            headways = {}
            for lc in line_colors:
                hw = st.sidebar.number_input(
                    f"[{lc}] Headway (min)",
                    1,
                    60,
                    5,
                    step=1,
                    key=f"hdw_{lc}",
                )
                headways[lc] = hw

        schedule_cfg = dict(mode="fixed", day=today, headways=headways)
    else:
        st.sidebar.caption(
            "Define peak/off-peak windows (e.g. 05:30–07:00, 07:00–10:00, etc.) and headways per period."
        )
        per_line = st.sidebar.radio(
            "Headway style in periods",
            ["Uniform across lines", "Per-line per period"],
            index=0,
            key="custom_hdw_style",
        )

        default_periods = [
            ("05:30", "07:00"),
            ("07:00", "10:00"),
            ("10:00", "14:00"),
            ("14:00", "18:00"),
            ("18:00", "24:00"),
        ]
        period_headways = {}

        for i, (start_str, end_str) in enumerate(default_periods, start=1):
            st.sidebar.markdown(f"**Period {i}: {start_str}–{end_str}**")
            if per_line == "Uniform across lines":
                hw = st.sidebar.number_input(
                    f"Headway (min) for {start_str}–{end_str}",
                    1,
                    60,
                    5,
                    step=1,
                    key=f"period_{i}_hw",
                )
                for lc in line_colors:
                    period_headways.setdefault(lc, []).append(
                        dict(start=start_str, end=end_str, headway=hw)
                    )
            else:
                for lc in line_colors:
                    hw = st.sidebar.number_input(
                        f"[{lc}] {start_str}–{end_str} (min)",
                        1,
                        60,
                        5,
                        step=1,
                        key=f"period_{i}_hw_{lc}",
                    )
                    period_headways.setdefault(lc, []).append(
                        dict(start=start_str, end=end_str, headway=hw)
                    )

        schedule_cfg = dict(
            mode="custom", day=today, periods=period_headways
        )

    return schedule_cfg


def build_timetable(lines_df, params_by_color, schedule_cfg):
    """
    جدول رحلات ليوم واحد فقط (24 ساعة).
    """
    line_colors = sorted(lines_df["Line Color A"].unique())

    line_station_sequences = {}
    for lc in line_colors:
        df_l = lines_df[lines_df["Line Color A"] == lc].copy()
        if "Segment Index" in df_l.columns:
            df_l = df_l.sort_values("Segment Index")
        seq = [df_l.iloc[0]["Station A"]]
        for _, r in df_l.iterrows():
            seq.append(r["Station B"])
        line_station_sequences[lc] = seq

    day = schedule_cfg["day"]
    timetable = []

    if schedule_cfg["mode"] == "fixed":
        headways = schedule_cfg["headways"]
        for lc in line_colors:
            headway_min = headways[lc]
            start_dt = datetime.combine(day, time(0, 0))
            end_dt = datetime.combine(day + timedelta(days=1), time(0, 0))
            for dep in hhmm_range(start_dt, end_dt, headway_min):
                timetable.append(
                    dict(
                        line=lc,
                        direction="forward",
                        depart_dt=dep,
                        stops=line_station_sequences[lc],
                    )
                )
                timetable.append(
                    dict(
                        line=lc,
                        direction="backward",
                        depart_dt=dep,
                        stops=list(reversed(line_station_sequences[lc])),
                    )
                )
    else:
        periods = schedule_cfg["periods"]
        for lc in line_colors:
            line_periods = periods.get(lc, [])
            seq = line_station_sequences[lc]
            for p in line_periods:
                start_str = p["start"]
                end_str = p["end"]
                headway_min = p["headway"]
                sh, sm = map(int, start_str.split(":"))
                eh, em = map(int, end_str.split(":"))
                start_dt = make_datetime_allow_24(day, sh, sm)
                end_dt = make_datetime_allow_24(day, eh, em)
                for dep in hhmm_range(start_dt, end_dt, headway_min):
                    timetable.append(
                        dict(
                            line=lc,
                            direction="forward",
                            depart_dt=dep,
                            stops=seq,
                        )
                    )
                    timetable.append(
                        dict(
                            line=lc,
                            direction="backward",
                            depart_dt=dep,
                            stops=list(reversed(seq)),
                        )
                    )

    timetable.sort(key=lambda x: x["depart_dt"])
    return timetable


def compute_energy(lines_df, timetable, params_by_color, eta_factor=1.0):
    """
    eta_factor:
        = 1.0  → قطار كهربائي عادي من الشبكة
        = batt_eff → قطار ببطارية (كفاءة أقل، استهلاك أعلى)
    """
    line_segments = {}
    for _, row in lines_df.iterrows():
        lc = row["Line Color A"]
        a = str(row["Station A"])
        b = str(row["Station B"])
        d_km = float(row["Distance In Km"])
        line_segments.setdefault(lc, {})
        line_segments[lc][(a, b)] = d_km
        line_segments[lc][(b, a)] = d_km

    energy_per_line_total = {
        lc: 0.0 for lc in lines_df["Line Color A"].unique()
    }
    station_power_time = defaultdict(list)
    line_energy_time = {
        lc: defaultdict(float) for lc in lines_df["Line Color A"].unique()
    }

    for trip in timetable:
        lc = trip["line"]
        stops = trip["stops"]
        depart_dt = trip["depart_dt"]
        pparams = params_by_color[lc]

        mass_kg = pparams["mass_kg"]
        eta_base = pparams["eta"]
        vmax_kmh = pparams["vmax_kmh"]
        accel_mps2 = pparams["accel_mps2"]
        stop_s = pparams["stop_s"]
        a = pparams["a"]
        b = pparams["b"]
        c = pparams["c"]

        # كفاءة فعالة (شبكة أو بطارية)
        eta_eff = eta_base * eta_factor

        running_s = 0.0
        trip_energy_total = 0.0

        for i in range(len(stops) - 1):
            sA = str(stops[i])
            sB = str(stops[i + 1])
            dist_km = line_segments[lc].get((sA, sB), None)
            if dist_km is None:
                continue

            grade = 0.0
            (
                seg_time_s,
                seg_avg_power_w,
                seg_e_kwh,
            ) = segment_energy_and_power(
                distance_km=dist_km,
                grade=grade,
                m_kg=mass_kg,
                eta=eta_eff,
                vmax_kmh=vmax_kmh,
                a_mps2=accel_mps2,
                stop_s=stop_s,
                coeff_a=a,
                coeff_b=b,
                coeff_c=c,
            )

            seg_start_dt = depart_dt + timedelta(seconds=running_s)
            time_str = seg_start_dt.strftime("%H:%M")

            p_kw = seg_avg_power_w / 1000.0
            station_power_time[(sA, time_str)].append(p_kw)
            line_energy_time[lc][time_str] += seg_e_kwh
            trip_energy_total += seg_e_kwh
            running_s += seg_time_s

        energy_per_line_total[lc] += trip_energy_total

    return energy_per_line_total, station_power_time, line_energy_time


# =============================================================================
# Map construction
# =============================================================================
def build_map(stations_df, lines_df, energy_per_line_total_display,
              station_power_time, display_time, label_suffix):
    # مركز الخريطة
    center_lat = stations_df["lat"].mean()
    center_lon = stations_df["long"].mean()

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="cartodbpositron",
    )

    # رسم الخطوط
    line_segments = defaultdict(list)
    for _, row in lines_df.iterrows():
        lc = row["Line Color A"]
        a = row["Station A"]
        b = row["Station B"]

        station_a = stations_df[stations_df["station"] == a]
        station_b = stations_df[stations_df["station"] == b]
        if station_a.empty or station_b.empty:
            continue

        latA, lonA = station_a.iloc[0]["lat"], station_a.iloc[0]["long"]
        latB, lonB = station_b.iloc[0]["lat"], station_b.iloc[0]["long"]
        line_segments[lc].append(((latA, lonA), (latB, lonB)))

    for lc, segs in line_segments.items():
        color = lc.lower()
        energy_mwh = energy_per_line_total_display.get(lc, 0.0) / 1000.0
        for (latA, lonA), (latB, lonB) in segs:
            folium.PolyLine(
                locations=[(latA, lonA), (latB, lonB)],
                color=color,
                weight=4,
                opacity=0.7,
                popup=f"{lc} line segment<br>Energy ({label_suffix}): {energy_mwh:.2f} MWh",
            ).add_to(fmap)

    # حساب القدرة لكل محطة في وقت معيّن
    t = display_time
    max_power_at_time = 0.0
    power_by_station = {}
    for (station, t_str), vals in station_power_time.items():
        if t_str == t:
            p_sum = float(sum(vals))
            power_by_station[station] = p_sum
            if p_sum > max_power_at_time:
                max_power_at_time = p_sum

    # رسم الدوائر على المحطات
    for _, row in stations_df.iterrows():
        st_name = row["station"]
        lat, lon = row["lat"], row["long"]
        h = str(row.get("height", "G"))

        p_kw = power_by_station.get(st_name, 0.0)

        # كل الدوائر نفس الحجم
        radius = 8

        # لون ثابت رصاصي
        color = "#888888"

        popup_html = f"<b>{st_name}</b><br>H: {h}<br>Power @ {t}: {p_kw:.1f} kW"

        folium.CircleMarker(
            location=[lat, lon],
            radius=0.8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=popup_html,
        ).add_to(fmap)

    return fmap


# =============================================================================
# Plotting
# =============================================================================
LINE_COLORS = {
    "Blue": "#1f77b4",
    "Red": "#d62728",
    "Green": "#2ca02c",
    "Orange": "#ff7f0e",
    "Purple": "#9467bd",
    "Yellow": "#bcbd22",
}
def plot_total_energy(energy_per_line_total_display, label_suffix):
    lines = sorted(energy_per_line_total_display.keys())
    energies_mwh = [
        energy_per_line_total_display[lc] / 1000.0 for lc in lines
    ]
    fig = go.Figure(
        data=[
            go.Bar(
              x=lines,
              y=energies_mwh,
              marker_color=[LINE_COLORS.get(lc, "#333") for lc in lines],
             text=[f"{e:.2f}" for e in energies_mwh],
             textposition="auto"
               )
            ]
    )
    fig.update_layout(
        title=f"Total Energy per Line ({label_suffix})",
        xaxis_title="Line Color",
        yaxis_title="Energy (MWh)",
        template="plotly_white",
    )
    return fig


def plot_mass_vs_energy(
    params_by_color,
    lines_df,
    timetable,
    base_mass_kg,
    label_suffix,
    eta_factor,
    mode="total",
    selected_line=None,
):
    """
    Sweep على وزن القطار حول القيمة الحالية:
    من 0.5×الوزن إلى 2.5×الوزن بعدد نقاط ثابت.
    mode = "total" → إجمالي الطاقة لكل الشبكة
    mode = "line"  → طاقة خط معيّن فقط
    """
    min_mass = max(10000, int(base_mass_kg * 0.5))
    max_mass = int(base_mass_kg * 2.5)
    num_points = 12

    masses_kg = list(np.linspace(min_mass, max_mass, num_points))
    masses_ton = [m / 1000.0 for m in masses_kg]

    energies_mwh = []

    for mass_kg in masses_kg:
        params_sweep = {}
        for lc, p in params_by_color.items():
            p2 = p.copy()
            if mode == "line" and lc != selected_line:
                p2["mass_kg"] = p["mass_kg"]
            else:
                p2["mass_kg"] = mass_kg
            params_sweep[lc] = p2

        energy_per_line_sweep, _, _ = compute_energy(
            lines_df, timetable, params_sweep, eta_factor=eta_factor
        )

        if mode == "total":
            total_energy_kwh = sum(energy_per_line_sweep.values())
        else:
            total_energy_kwh = energy_per_line_sweep.get(selected_line, 0.0)

        energies_mwh.append(total_energy_kwh / 1000.0)

    if mode == "total":
        title = f"Train Mass vs Total Energy (All Lines, {label_suffix})"
    else:
        title = (
            f"Train Mass vs Energy ({selected_line} line, {label_suffix})"
        )

    fig = go.Figure(
        data=[
            go.Scatter(
                x=masses_ton,
                y=energies_mwh,
                mode="lines+markers",
                hovertemplate=(
                    "Mass: %{x:.1f} ton<br>"
                    "Energy: %{y:.2f} MWh<br><extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title="Train mass (ton)",
        yaxis_title="Total Energy (MWh)",
        template="plotly_white",
    )

    return fig


def plot_station_power_at_time(stations_df, station_power_time, time_str):
    station_names = stations_df["station"].tolist()
    power_vals = []
    for st_name in station_names:
        vals = []
        for (s, t_s), p_list in station_power_time.items():
            if s == st_name and t_s == time_str:
                vals.extend(p_list)
        power_vals.append(sum(vals) if vals else 0.0)

    fig = go.Figure(data=[go.Bar(x=station_names, y=power_vals)])
    fig.update_layout(
        title=f"Station Power at {time_str}",
        xaxis_title="Station",
        yaxis_title="Power (kW)",
        xaxis_tickangle=45,
        template="plotly_white",
    )
    return fig


def plot_cumulative_energy(line_energy_time_display, label_suffix):
    fig = go.Figure()

    for lc, t_dict in line_energy_time_display.items():
        times_sorted = sorted(t_dict.keys())
        cumulative = []
        total = 0.0

        for t_str in times_sorted:
            total += t_dict[t_str]
            cumulative.append(total)

        fig.add_trace(
            go.Scatter(
                x=times_sorted,
                y=cumulative,
                mode="lines",
                name=lc,
                line=dict(color=LINE_COLORS.get(lc, "#333"))
            )
        )

    fig.update_layout(
        title=f"Cumulative Energy per Line ({label_suffix}, kWh)",
        xaxis_title="Time (HH:MM)",
        yaxis_title="Cumulative Energy (kWh)",
        template="plotly_white",
    )

    return fig



def plot_station_time_series(stations_df, station_power_time, station_name):
    times_sorted = sorted(
        {t_str for (s, t_str) in station_power_time.keys() if s == station_name}
    )
    series = []
    for t_str in times_sorted:
        vals = []
        for (s, tt), p_list in station_power_time.items():
            if s == station_name and tt == t_str:
                vals.extend(p_list)
        series.append(sum(vals) if vals else 0.0)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=times_sorted, y=series, mode="lines+markers", name=station_name
            )
        ]
    )
    fig.update_layout(
        title=f"Station Power Time-Series: {station_name}",
        xaxis_title="Time (HH:MM)",
        yaxis_title="Power (kW)",
        template="plotly_white",
    )
    return fig


def plot_station_heatmap(stations_df, station_power_time):
    stations = stations_df["station"].tolist()
    times = sorted({t_str for (_, t_str) in station_power_time.keys()})
    data_matrix = []
    for st_name in stations:
        row = []
        for t_str in times:
            vals = []
            for (s, tt), p_list in station_power_time.items():
                if s == st_name and tt == t_str:
                    vals.extend(p_list)
            row.append(sum(vals) if vals else 0.0)
        data_matrix.append(row)

    df_heat = pd.DataFrame(data_matrix, index=stations, columns=times)
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_heat, cmap="viridis")
    plt.title("Station Power Heatmap (kW)")
    plt.xlabel("Time (HH:MM)")
    plt.ylabel("Station")
    st.pyplot(plt.gcf())
    plt.close()


# =============================================================================
# GROQ Chatbot helper
# =============================================================================
def build_metro_context(
    energy_per_line_total_display,
    station_power_time,
    display_time,
    max_station_name,
    max_station_power,
    max_station_time,
    peak_time,
    peak_system_power,
    total_energy_kwh_display,
    total_energy_mwh_display,
    trips_per_line_display,
    total_trips_display,
    days_factor,
    label_suffix,
    emission_factor,
    total_co2_t_display,
    time_view_mode,
):
    snapshot = {}
    for (station, t_str), vals in station_power_time.items():
        if t_str == display_time:
            snapshot[station] = float(sum(vals))

    ctx = {
        "time_view_mode": time_view_mode,
        "days_factor": int(days_factor),
        "label_suffix": label_suffix,
        "display_time": display_time,
        "total_energy_kwh": float(total_energy_kwh_display),
        "total_energy_mwh": float(total_energy_mwh_display),
        "energy_per_line_kwh": {
            str(k): float(v) for k, v in energy_per_line_total_display.items()
        },
        "trips": {
            "per_line": {
                str(k): int(v) for k, v in trips_per_line_display.items()
            },
            "total": int(total_trips_display),
        },
        "peak_station": {
            "name": max_station_name,
            "power_kw": float(max_station_power),
            "time": max_station_time,
        },
        "system_peak": {
            "time": peak_time,
            "power_kw": float(peak_system_power),
        },
        "station_power_at_display_time_kw": snapshot,
        "emissions": {
            "emission_factor_kg_per_kwh": float(emission_factor),
            "total_co2_t_per_period": float(total_co2_t_display),
        },
        "cost_assumption_SAR_per_kWh": 0.2,
    }
    return ctx


def call_groq_chat(user_message, context_json, chat_history):
    if groq_client is None:
        return "GROQ API is not configured."

    system_prompt = (
        "You are a helpful assistant embedded in a Streamlit metro energy dashboard. "
        "You answer questions about metro lines, power, energy, DAILY/WEEKLY totals, "
        "number of trips, and simple cost and CO2 estimates. "
        "Use ONLY the data given in the 'metro_context' JSON, and basic physics/maths. "
        "If user asks about cost, assume electricity price = 0.2 SAR per kWh unless user "
        "explicitly provides another value. "
        "If something is unknown, be honest and say you are approximating.\n\n"
        f"metro_context = {context_json}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_message})

    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.2,
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()


# =============================================================================
# Main App Layout
# =============================================================================
st.title("🚆 Metro Energy & Power Explorer")
st.write(
    "Upload your metro network Excel (Stations + Lines + optional Crossing), "
    "choose train parameters & headways, and explore energy and power across the day or the week."
)

# --------- Train type (Grid vs Battery) ----------
st.sidebar.markdown("## Train Type")
train_type = st.sidebar.radio(
    "Select train type:", ["Grid-fed Electric", "Battery Electric"], index=0
)

if train_type == "Battery Electric":
    st.sidebar.markdown("### Battery Parameters")
    trips_per_charge = st.sidebar.number_input(
        "Trips per charge (round-trips)",
        min_value=1,
        max_value=50,
        value=4,
        step=1,
    )
    dod = (
        st.sidebar.slider(
            "Depth of Discharge (DoD %)",
            min_value=20,
            max_value=100,
            value=80,
        )
        / 100.0
    )
    batt_eff = (
        st.sidebar.slider(
            "Battery efficiency (%)", min_value=70, max_value=100, value=95
        )
        / 100.0
    )
else:
    trips_per_charge = None
    dod = None
    batt_eff = None

# --------- Time view mode ----------
st.sidebar.markdown("## Time View Mode")
time_view_mode = st.sidebar.radio(
    "Select time view mode:",
    ("Specific Time", "Full Day", "Full Week"),
    index=0,
)
days_factor = 7 if time_view_mode == "Full Week" else 1
label_suffix = "per week" if days_factor == 7 else "per day"

# --------- Upload Excel ----------
uploaded_file = st.file_uploader(
    "Upload Excel file with 'Stations' and 'Lines' sheets", type=["xlsx"]
)

if not uploaded_file:
    st.info("Please upload an Excel file to proceed.")
    st.stop()

stations_df, lines_df, crossing_df = load_excel(uploaded_file)
st.success("Excel data loaded successfully.")

line_colors = sorted(lines_df["Line Color A"].unique())

# Train parameters & schedule
params_by_color = parameter_section(line_colors)
schedule_cfg = schedule_section(line_colors)

# ----------------- نحسب دائماً ليوم واحد -----------------
with st.spinner("Building 1-day timetable and computing energy..."):
    timetable = build_timetable(lines_df, params_by_color, schedule_cfg)

    # لو قطار ببطارية: نطبّق كفاءة البطارية على الـ eta
    if train_type == "Battery Electric" and batt_eff is not None:
        eta_factor = batt_eff
    else:
        eta_factor = 1.0

    (
        energy_per_line_total_day,
        station_power_time,
        line_energy_time_day,
    ) = compute_energy(
        lines_df,
        timetable,
        params_by_color,
        eta_factor=eta_factor,
    )

# ----------------- نحسب قيم العرض حسب اليوم/الأسبوع -------------------
energy_per_line_total_display = {
    lc: val * days_factor for lc, val in energy_per_line_total_day.items()
}
line_energy_time_display = {
    lc: {t: e * days_factor for t, e in t_dict.items()}
    for lc, t_dict in line_energy_time_day.items()
}
total_energy_kwh_display = float(sum(energy_per_line_total_display.values()))
total_energy_mwh_display = total_energy_kwh_display / 1000.0

# ---------------------------- Dashboard Summary ------------------------------
st.markdown("### Dashboard Summary")

trips_per_line_day = (
    pd.Series([t["line"] for t in timetable]).value_counts().sort_index()
)
trips_per_line_display = trips_per_line_day * days_factor
total_trips_display = int(len(timetable) * days_factor)

# ---------------- Battery sizing (if Battery Electric) ----------------
battery_results = {}
if train_type == "Battery Electric":
    for lc in sorted(energy_per_line_total_day.keys()):
        daily_energy_kwh = energy_per_line_total_day[lc]
        daily_trips = trips_per_line_day.get(lc, 1)

        energy_per_trip = daily_energy_kwh / max(daily_trips, 1)
        round_trip_energy = energy_per_trip * 2

        required_batt = (round_trip_energy * trips_per_charge) / (
            dod * batt_eff
        )
        battery_results[lc] = required_batt

total_lines = len(lines_df["Line Color A"].unique())
total_stations = len(stations_df)

max_station_power = 0.0
max_station_name = None
max_station_time = None
for (station, t_str), vals in station_power_time.items():
    p = float(sum(vals))
    if p > max_station_power:
        max_station_power = p
        max_station_name = station
        max_station_time = t_str

time_totals = defaultdict(float)
for (station, t_str), vals in station_power_time.items():
    time_totals[t_str] += float(sum(vals))

peak_time = None
peak_system_power = 0.0
if time_totals:
    peak_time, peak_system_power = max(
        time_totals.items(), key=lambda kv: kv[1]
    )

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Lines", total_lines)
col2.metric("Stations", total_stations)
col3.metric(
    f"Total Trips ({label_suffix})", f"{total_trips_display:,}"
)
col4.metric(
    f"Total Energy (All Lines, {label_suffix})",
    f"{total_energy_mwh_display:.2f} MWh",
    help=f"{total_energy_kwh_display:.0f} kWh",
)

emission_factor = 0.55
total_co2_kg_display = total_energy_kwh_display * emission_factor
total_co2_t_display = total_co2_kg_display / 1000.0

col5, col6, col7 = st.columns(3)
if max_station_name is not None:
    col5.metric(
        "Peak Station Power (kW)",
        f"{max_station_power:,.0f}",
        help=f"{max_station_name} at {max_station_time}",
    )
if peak_time is not None:
    col6.metric(
        "Peak System Power (kW)",
        f"{peak_system_power:,.0f}",
        help=f"At {peak_time} (sum over all stations)",
    )
col7.metric(
    f"CO₂ equivalent (tCO₂/{'week' if days_factor == 7 else 'day'})",
    f"{total_co2_t_display:.1f}",
    help=f"Assuming {emission_factor} kg CO₂ per kWh",
)

st.markdown("### Data Summary")

# ---------------------------- Map & Visualizations ---------------------------
time_keys = sorted({t_str for (_, t_str) in station_power_time.keys()})
if not time_keys:
    st.warning("No time-based station power found.")
    st.stop()

if time_view_mode == "Specific Time":
    default_idx = len(time_keys) // 2
    display_time = st.sidebar.selectbox(
        "Select time (HH:MM) for map and plots:",
        time_keys,
        index=default_idx,
    )
else:
    if peak_time in time_keys:
        display_time = peak_time
    else:
        display_time = time_keys[len(time_keys) // 2]

if time_view_mode == "Full Day":
    time_mode_label = f"Full day view, using system peak time: {display_time}"
elif time_view_mode == "Full Week":
    time_mode_label = (
        f"Full week view (7x totals), using peak time: {display_time}"
    )
else:
    time_mode_label = f"View at selected time: {display_time}"

st.markdown(f"**Time View:** {time_mode_label}")

with st.expander("Interactive Map", expanded=True):
    fmap = build_map(
        stations_df,
        lines_df,
        energy_per_line_total_display,
        station_power_time,
        display_time,
        label_suffix,
    )
    st_folium(fmap, width=900, height=600)

st.markdown("### Visualization Panel")

vis_options = [
    "Total Energy per Line (Bar)",
    "Total Energy (All Lines)",
    "Cumulative Energy by Line (Time-Series)",
    "Station Time Series (Select Station)",
    "Train Mass vs Energy per Line",
]

selection = st.selectbox("Choose a plot", vis_options, index=0)

if selection == "Total Energy per Line (Bar)":
    st.plotly_chart(
        plot_total_energy(energy_per_line_total_display, label_suffix),
        use_container_width=True,
    )

elif selection == "Total Energy (All Lines)":
    st.subheader(f"⚡ Total Energy Consumption (All Lines, {label_suffix})")

    col1_, col2_ = st.columns(2)
    col1_.metric("Total Energy (kWh)", f"{total_energy_kwh_display:,.2f}")
    col2_.metric("Total Energy (MWh)", f"{total_energy_mwh_display:,.2f}")

    # نجهّز الليبلز والقيم
    labels = list(energy_per_line_total_display.keys())
    values = list(energy_per_line_total_display.values())

    # Pie chart بألوان متطابقة مع لون كل خط
    fig_total = go.Figure(
        data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker=dict(
                colors=[LINE_COLORS.get(l, "#333333") for l in labels]
            ),
        )]
    )
    fig_total.update_layout(
        title=f"Energy Distribution per Line ({label_suffix})",
        template="plotly_white",
    )
    st.plotly_chart(fig_total, use_container_width=True)


elif selection == "Cumulative Energy by Line (Time-Series)":
    st.plotly_chart(
        plot_cumulative_energy(line_energy_time_display, label_suffix),
        use_container_width=True,
    )

elif selection == "Station Time Series (Select Station)":
    station_choice = st.selectbox(
        "Select station", stations_df["station"].tolist()
    )
    st.plotly_chart(
        plot_station_time_series(
            stations_df, station_power_time, station_choice
        ),
        use_container_width=True,
    )

elif selection == "Train Mass vs Energy per Line":
    mode_choice = st.radio(
        "Mass–Energy mode",
        ["Total (all lines)", "Single line"],
        index=0,
        horizontal=True,
    )

    selected_line = None
    if mode_choice == "Single line":
        selected_line = st.selectbox(
            "Select line color", sorted(params_by_color.keys())
        )

    if mode_choice == "Single line" and selected_line is not None:
        base_mass_kg = params_by_color[selected_line]["mass_kg"]
        mode = "line"
    else:
        any_lc = sorted(params_by_color.keys())[0]
        base_mass_kg = params_by_color[any_lc]["mass_kg"]
        mode = "total"

    st.plotly_chart(
        plot_mass_vs_energy(
            params_by_color=params_by_color,
            lines_df=lines_df,
            timetable=timetable,
            base_mass_kg=base_mass_kg,
            label_suffix=label_suffix,
            eta_factor=eta_factor,
            mode=mode,
            selected_line=selected_line,
        ),
        use_container_width=True,
    )

# ---------------------------- Battery Results (if Battery Electric) ---------------------------
if train_type == "Battery Electric" and battery_results:
    st.markdown("### 🔋 Battery Sizing Results (per train)")

    batt_df = pd.DataFrame(
        [
            {"Line": lc, "Required_Battery_kWh": battery_results[lc]}
            for lc in sorted(battery_results.keys())
        ]
    )

    st.dataframe(batt_df, use_container_width=True)

    fig_batt = go.Figure(
        data=[
            go.Bar(
                x=batt_df["Line"],
                y=batt_df["Required_Battery_kWh"],
                text=[
                    f"{v:.0f} kWh"
                    for v in batt_df["Required_Battery_kWh"]
                ],
                textposition="auto",
            )
        ]
    )
    fig_batt.update_layout(
        title="Required Battery Capacity per Train (kWh)",
        xaxis_title="Line",
        yaxis_title="Battery Size (kWh)",
        template="plotly_white",
    )
    st.plotly_chart(fig_batt, use_container_width=True)

# ---------------------------- Data Summary Tables ---------------------------
st.markdown(f"### Data Summary Tables ({label_suffix})")

st.markdown("#### Line Energy (kWh / MWh)")
energy_df = pd.DataFrame(
    [
        {
            "Line": lc,
            "Energy_kWh": energy_per_line_total_display[lc],
            "Energy_MWh": energy_per_line_total_display[lc] / 1000.0,
        }
        for lc in sorted(energy_per_line_total_display.keys())
    ]
)
st.dataframe(energy_df, use_container_width=True)

st.markdown("#### Trips per Line (both directions)")
trip_df = pd.DataFrame(
    {"Line": trips_per_line_display.index, "Trips": trips_per_line_display.values}
)
st.dataframe(trip_df, use_container_width=True)

st.markdown("#### Download Results")
csv_energy = energy_df.to_csv(index=False).encode()
st.download_button(
    "Download Line Energy CSV",
    data=csv_energy,
    file_name=f"line_energy_{'week' if days_factor == 7 else 'day'}.csv",
    mime="text/csv",
)

station_rows = []
for (station, t_str), vals in station_power_time.items():
    station_rows.append(
        {"station": station, "time": t_str, "sum_power_kW": float(sum(vals))}
    )
station_power_df = pd.DataFrame(station_rows)
csv_station = station_power_df.to_csv(index=False).encode()
st.download_button(
    "Download Station Power CSV (kW vs time, same for day/week)",
    data=csv_station,
    file_name="station_power_kW.csv",
    mime="text/csv",
)

st.success("Done. Use Time View Mode to switch between per-day and per-week totals.")

# ---------------------------- GROQ Chatbot UI -------------------------------
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []
if "chat_open" not in st.session_state:
    st.session_state["chat_open"] = False

icon_cols = st.columns([0.8, 0.2])
with icon_cols[1]:
    if st.button("💬 Chat", key="toggle_chatbot"):
        st.session_state["chat_open"] = not st.session_state["chat_open"]

if st.session_state["chat_open"]:
    st.markdown("### 🤖 Metro Chatbot (GROQ)")
    st.info(
        "اسألني عن الطاقة، القدرة، عدد الرحلات، أو تكلفة تقديرية للكهرباء.\n"
        "مثال: *كم الطاقة المستهلكة في الأسبوع؟*، "
        "*كم تكلفة الاستهلاك اليومي؟*، "
        "*كم عدد رحلات الخط الأزرق في الأسبوع؟*"
    )

    metro_ctx = build_metro_context(
        energy_per_line_total_display=energy_per_line_total_display,
        station_power_time=station_power_time,
        display_time=display_time,
        max_station_name=max_station_name,
        max_station_power=max_station_power,
        max_station_time=max_station_time,
        peak_time=peak_time,
        peak_system_power=peak_system_power,
        total_energy_kwh_display=total_energy_kwh_display,
        total_energy_mwh_display=total_energy_mwh_display,
        trips_per_line_display=trips_per_line_display.to_dict(),
        total_trips_display=total_trips_display,
        days_factor=days_factor,
        label_suffix=label_suffix,
        emission_factor=emission_factor,
        total_co2_t_display=total_co2_t_display,
        time_view_mode=time_view_mode,
    )
    ctx_json = json.dumps(metro_ctx, ensure_ascii=False)

    for msg in st.session_state["chat_messages"]:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg["content"])

    if prompt := st.chat_input("اكتب سؤالك هنا..."):
        user_text = prompt.strip()
        if user_text:
            st.session_state["chat_messages"].append(
                {"role": "user", "content": user_text}
            )
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(user_text)

            answer = call_groq_chat(
                user_message=user_text,
                context_json=ctx_json,
                chat_history=st.session_state["chat_messages"],
            )

            st.session_state["chat_messages"].append(
                {"role": "assistant", "content": answer}
            )
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(answer)

# ---------------------------------------------------------------------
# 📄 PDF Report Export Section
# ---------------------------------------------------------------------
st.markdown("---")
st.markdown("### Download PDF Report")

if st.button("Generate PDF Report"):

    def safe_text(text):
        if not isinstance(text, str):
            text = str(text)
        return text.encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(
        200,
        10,
        txt=safe_text("Metro Energy and Power Report"),
        ln=True,
        align="C",
    )
    pdf.ln(10)

    pdf.multi_cell(
        0, 8, txt=safe_text(f"Time view mode: {time_view_mode}")
    )
    pdf.multi_cell(
        0,
        8,
        txt=safe_text(
            f"Total energy (kWh): {total_energy_kwh_display:.2f}"
        ),
    )
    pdf.multi_cell(
        0,
        8,
        txt=safe_text(
            f"Total trips ({label_suffix}): {total_trips_display}"
        ),
    )
    if max_station_name is not None:
        pdf.multi_cell(
            0,
            8,
            txt=safe_text(
                f"Peak station: {max_station_name} at {max_station_time} "
                f"with power {max_station_power:.1f} kW"
            ),
        )
    if peak_time is not None:
        pdf.multi_cell(
            0,
            8,
            txt=safe_text(
                f"System peak time: {peak_time} with total power {peak_system_power:.1f} kW"
            ),
        )

    pdf.ln(5)
    pdf.multi_cell(0, 8, txt=safe_text("Line energy summary (kWh):"))

    for _, row in energy_df.iterrows():
        line_name = row["Line"]
        e_kwh = row["Energy_kWh"]
        pdf.multi_cell(
            0, 8, txt=safe_text(f"  - {line_name}: {e_kwh:.2f} kWh")
        )

    pdf.ln(5)
    pdf.multi_cell(0, 8, txt=safe_text("Trips per line:"))
    for _, row in trip_df.iterrows():
        line_name = row["Line"]
        trips_val = row["Trips"]
        pdf.multi_cell(
            0, 8, txt=safe_text(f"  - {line_name}: {trips_val} trips")
        )

    pdf_bytes = pdf.output(dest="S").encode("latin-1")

    st.download_button(
        "Download PDF",
        data=pdf_bytes,
        file_name="metro_report.pdf",
        mime="application/pdf",
    )