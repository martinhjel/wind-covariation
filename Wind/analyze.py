import logging 
from pathlib import Path
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from geopy.distance import geodesic
from scipy.optimize import curve_fit
from sklearn.neighbors import KernelDensity
from Wind.plotly_template import my_template
import seaborn as sns

logger = logging.getLogger("__name__")
logger.setLevel(logging.INFO)

pd.set_option("plotting.backend", "plotly")


def get_corr_figure(df, scale_size=1.0):
    corr = df.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    df_mask = corr.mask(mask).round(2)

    fig = ff.create_annotated_heatmap(
        z=df_mask.to_numpy(),
        x=df_mask.columns.tolist(),
        y=df_mask.columns.tolist(),
        colorscale=px.colors.sequential.matter,
        # colorscale=px.colors.diverging.RdBu,
        # colorscale=px.colors.sequential.Blackbody_r,
        hoverinfo="none",  # Shows hoverinfo for null values
        showscale=True,
        ygap=1,
        xgap=1,
        # zmid=0,
    )

    fig.update_xaxes(side="bottom")

    fig.update_layout(
        #     title_text='Heatmap',
        #     title_x=0.5,
        width=1200 * scale_size,
        height=1000 * scale_size,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_zeroline=False,
        yaxis_zeroline=False,
        #     yaxis_autorange='reversed',
        template="plotly_white",
    )
    return fig


def get_hours_shift_figure(df, df_nve_wind_locations, n_shifts, quantile):
    froya_lat = df_nve_wind_locations[df_nve_wind_locations["location"] == "Frøyabanken"]["lat"].values[0]

    cols_south = df_nve_wind_locations[df_nve_wind_locations["lat"] < froya_lat]["location"].to_list()
    cols_north = df_nve_wind_locations[df_nve_wind_locations["lat"] >= froya_lat]["location"].to_list()

    df["All 15 wind farms"] = df.mean(axis=1)
    df["Farms north of Stadt"] = df[cols_north].mean(axis=1)
    df["Farms south of Stadt"] = df[cols_south].mean(axis=1)

    df_t = df[
        [
            "Utsira nord",
            "Sørlige Nordsjø I",
            "Nordmela",
            "Farms south of Stadt",
            "Farms north of Stadt",
            "All 15 wind farms",
        ]
    ]
    df_shift = pd.concat([df_t.diff(i).quantile(q=quantile) for i in range(n_shifts)], axis=1).T

    df_shift.index = [i for i in range(n_shifts)]

    df_shift.index.name = "hour shift"
    n_cols = len(df_shift.columns)
    colormap = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c"]
    # colormap = ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462']
    # colormap = sns.color_palette("Set3", n_cols).as_hex()
    colors = {col: color for col, color in zip(df_shift.columns, colormap)}

    fig = df_shift.plot(
        title=f"", template=my_template, labels=dict(value="Absolute change in power output"), color_discrete_map=colors
    )

    df = df.drop(columns=["All 15 wind farms", "Farms north of Stadt", "Farms south of Stadt"])

    fig.update_layout(legend_title="", legend=dict(y=0.95, x=0.30), yaxis_range=[0, 1])
    return fig


def get_mean_std_wind_figure(df, resample_period):
    df_res = df.resample(resample_period).mean()

    x = list(df_res.index)
    y = df_res.mean(axis=1).values

    y_upper = y + df_res.std(axis=1).values
    y_lower = y - df_res.std(axis=1).values

    fig = go.Figure(
        [
            go.Scatter(
                x=np.concatenate((x, x[::-1])),  # x, then x reversed
                y=np.concatenate((y_upper, y_lower[::-1])),  # upper, then lower reversed
                fill="toself",
                fillcolor="#A9B7C7",
                line=dict(color="#A9B7C7"),
                hoverinfo="skip",
                showlegend=False,
            ),
            go.Scatter(x=x, y=y, name="Mean wind", line=dict(color="#3a4454"), mode="lines"),
        ]
    )
    fig.update_layout(title=f"Period: {resample_period}", template=my_template, xaxis_title="time", yaxis_title=" ")

    return fig


def get_threshold_figure(df):
    cols = ["Sørlige Nordsjø II"]
    hours = 24 * 20
    threshold_list = []
    for hours in range(hours):
        bins = [i / 100 for i in range(40)]
        cats = pd.cut(df[cols].mean(axis=1).rolling(hours, center=True).mean(), bins=bins).value_counts()
        cats = cats.sort_index(ascending=True)
        cats.name = hours
        threshold_list.append(cats)
    df_tr = pd.concat(threshold_list, axis=1)
    df_tr = df_tr.cumsum()

    a = 0.00011
    b = 1
    n = 10

    def get_color_list(a, b, n):
        return [[a + (i) * ((b - a) / (n - 1)), c] for i, c in enumerate(px.colors.sequential.Viridis)]

    fig = go.Figure(
        data=go.Heatmap(
            z=df_tr.values,
            #         zmin=0,
            #         zmax=100,
            x=df_tr.columns,
            y=[i.left for i in df_tr.index],
            colorscale=[[0, "#FFFFFF"], [0.0001, "#FFFFFF"]] + get_color_list(a, b, n),
        )
    )
    fig.update_layout(title=f"", xaxis_title="consequtive hours", yaxis_title="power threshold [%]")
    fig.show()


def get_corr_distance_df(df, df_locations, resolution="1H"):
    locs = []
    for _, row in df_locations.iterrows():
        locs.append((row["lat"], row["lon"]))

    distances = []
    for i in locs:
        distances.append([])
        for j in locs:
            distances[-1].append(geodesic(i, j).km)

    df_corr = df.resample(resolution).mean().corr()
    mask = np.triu(np.ones_like(df_corr, dtype=bool))
    df_corr = df_corr.mask(mask).round(2)

    df_distance = pd.DataFrame(data=distances, index=df_corr.index, columns=df_corr.columns)
    mask = np.triu(np.ones_like(df_distance, dtype=bool))
    df_distance = df_distance.mask(mask).round(2)

    corr_list = []
    for i, row in df_corr.iterrows():
        for j, v in row.items():
            if not np.isnan(v):
                corr_list.append((f"{i} <-> {j}", v))
    dist_list = []
    for i, row in df_distance.iterrows():
        for j, v in row.items():
            if not np.isnan(v):
                dist_list.append((f"{i} <-> {j}", v))

    df_temp = pd.DataFrame(corr_list)
    df_temp = df_temp.set_index(0)
    df_temp = df_temp.rename(columns={1: "Correlation"})

    df_temp1 = pd.DataFrame(dist_list)
    df_temp1 = df_temp1.set_index(0)
    df_temp1 = df_temp1.rename(columns={1: "Distance [km]"})

    df_corr_dist = pd.concat([df_temp, df_temp1], axis=1)
    df_corr_dist = df_corr_dist.reset_index()
    df_corr_dist = df_corr_dist.rename(columns={0: "Span"})

    return df_corr_dist


def get_exponential_function(df_corr_dist):
    # Fit an exponential function
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c

    popt, pcov = curve_fit(
        func, df_corr_dist["Distance [km]"].values, df_corr_dist["Correlation"].values, p0=[1, 0.005, 0]
    )

    xn = np.linspace(df_corr_dist["Distance [km]"].min(), df_corr_dist["Distance [km]"].max(), 2500)
    return xn, func(xn, *popt), popt


def get_multiple_corr_distance_figure(df, df_locations, resolutions, colors):
    data = []
    for i, res in enumerate(resolutions):
        df_corr_dist = get_corr_distance_df(df, df_locations, resolution=res)
        data.append(
            go.Scatter(
                x=df_corr_dist["Distance [km]"],
                y=df_corr_dist["Correlation"],
                text=df_corr_dist["Span"],
                marker=dict(color=colors[i], size=5, line=dict(width=0)),
                mode="markers",
                name=f"Between two wind farms - {res}",
            )
        )
        xn, yn, popt = get_exponential_function(df_corr_dist)
        data.append(
            go.Scatter(
                x=xn,
                y=yn,
                line=dict(color=colors[i], width=3),
                name=f"Exponential fit with {res} resolution",  # r"$1.05 \exp(\frac{-1}{490.4}x) + 0.02$",
            )
        )

        logger.info(f"Fitted function, {res} resolution, {colors[i]} color, {popt} popt -> (a,b,c) y=a*e^(-b*x)+c")

    fig = go.Figure(data=data)
    fig.update_layout(template=my_template, title=f"", xaxis_title="Distance [km]", yaxis_title="Correlation [-]")
    return fig


def get_corr_distance_figure(df, df_locations, resolution="1H"):
    df_corr_dist = get_corr_distance_df(df, df_locations, resolution)
    xn, yn, popt = get_exponential_function(df_corr_dist)
    data = [
        go.Scatter(
            x=df_corr_dist["Distance [km]"],
            y=df_corr_dist["Correlation"],
            text=df_corr_dist["Span"],
            marker=dict(color="Black", size=5, line=dict(width=0)),
            mode="markers",
            name="Between two wind farms",
        ),
        go.Scatter(
            x=xn,
            y=yn,
            line=dict(color="#5D8CC0", width=3),
            name=f"Exponential fit with {resolution} resolution",  # r"$1.05 \exp(\frac{-1}{490.4}x) + 0.02$",
        ),
    ]

    fig = go.Figure(data=data)

    fig.update_layout(template=my_template, title=f"", xaxis_title="Distance [km]", yaxis_title="Correlation [-]")

    return fig


def get_line_plot_with_mean(df, area, resample_period):
    ind = df.index.map(lambda x: x - pd.Timestamp(year=x.year, month=1, day=1, hour=0, minute=0))

    dff = pd.DataFrame(df[area].values, index=ind, columns=[area])
    dff["Year"] = df.index.year
    dff = dff.reset_index().pivot(index="index", columns="Year").dropna()
    dff.columns = dff.columns.droplevel()
    dff.index = dff.index + pd.Timestamp("1970-1-1")  # Hack to show dates in figure
    years = dff.columns

    dff = dff.resample(resample_period).mean()

    dff = dff.reset_index(names="date")

    fig = px.line(
        dff,
        x="date",
        y=years,
        hover_data={"date": "|%d. %B, %H:%M"},
        #     color_discrete_sequence=px.colors.sequential.Blues,
        color_discrete_sequence=sns.color_palette("mako", len(dff.columns)).as_hex(),
    )
    fig.update_traces(opacity=0.5)

    dff = dff.set_index("date")
    x = list(dff.index)
    y = dff.mean(axis=1).values

    y_upper = y + dff.std(axis=1).values
    y_lower = y - dff.std(axis=1).values

    fig.add_traces(
        [
            go.Scatter(
                x=np.concatenate((x, x[::-1])),  # x, then x reversed
                y=np.concatenate((y_upper, y_lower[::-1])),  # upper, then lower reversed
                fill="toself",
                fillcolor="#A9B7C7",
                line=dict(color="#A9B7C7"),
                hoverinfo="skip",
                showlegend=False,
                name="std",
            ),
            go.Scatter(x=x, y=y, name="Mean", line=dict(color="#3a4454"), mode="lines"),
        ]
    )

    fig.update_xaxes(dtick="M1", tickformat="%j", ticklabelmode="period")

    my_template.layout.legend = dict(
        yanchor=None,
        xanchor=None,  # y = 0.95, x=1.2,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=12, family="Times New Roman"),
    )
    fig.update_layout(
        title=f"Area: {area}, with resample period of {resample_period}",
        template=my_template,
        xaxis_title="Day",
        yaxis_title="Wind power output [-]",
    )

    viz_cols = ["Mean", "std"]

    fig.update_traces(visible="legendonly", selector=lambda t: not t.name in viz_cols)

    fig.update_xaxes(dtick="M1", tickformat="%b")
    return fig


def get_mean_std_wind_yearly_figure(df, resample_period):
    """
    Plot the mean and std for all wind farms. Assume equal build-out.
    """
    dff = df.copy(deep=True)
    dff["Year"] = dff.index.year
    cols = dff.columns
    dff["time"] = df.index.map(lambda x: x - pd.Timestamp(year=x.year, month=1, day=1, hour=0, minute=0))

    dff = dff.pivot(index="time", columns="Year")
    dff = dff.resample(resample_period).mean()

    dff.columns = [f"{i[0]}: {i[1]}" for i in dff.columns]
    dff.index = dff.index + pd.Timestamp("1970-1-1")  # Hack to show dates in figure

    x = list(dff.index)
    y = dff.mean(axis=1).values

    y_upper = y + dff.std(axis=1).values
    y_lower = y - dff.std(axis=1).values

    fig = go.Figure()
    fig.add_traces(
        [
            go.Scatter(
                x=np.concatenate((x, x[::-1])),  # x, then x reversed
                y=np.concatenate((y_upper, y_lower[::-1])),  # upper, then lower reversed
                fill="toself",
                fillcolor="#A9B7C7",
                line=dict(color="#A9B7C7"),
                hoverinfo="skip",
                showlegend=False,
            ),
            go.Scatter(x=x, y=y, name="Mean wind", line=dict(color="#3a4454"), mode="lines"),
        ]
    )

    fig.update_xaxes(dtick="M1", tickformat="%j", ticklabelmode="period")

    fig.update_layout(
        title=f"Resample period {resample_period}", template=my_template, xaxis_title="day", yaxis_title=" "
    )
    return fig


def get_scatter_2d_figure(df, area_a, area_b):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[area_a], y=df[area_b], mode="markers", marker=dict(line_width=1, opacity=0.3)))
    fig.update_layout(
        title=f"", template=my_template, xaxis_title=f"{area_a}", yaxis_title=f"{area_b}", width=600, height=600
    )
    return fig


def get_histogram_2d_figure(df, area_a, area_b):
    fig = go.Figure()
    fig.add_trace(
        go.Histogram2d(
            x=df[area_a],
            y=df[area_b],
            # histnorm='probability',
            colorscale="PuBu",
            xbins=dict(start=0, end=1, size=0.025),
            ybins=dict(start=0, end=1, size=0.025),
            zauto=True,
        )
    )
    fig.update_layout(
        title=f"2D Histogram",
        template=my_template,
        xaxis_title=f"{area_a}",
        yaxis_title=f"{area_b}",
        width=600,
        height=600,
    )
    return fig


def get_scatter_density_2d_figure(df, area_a, area_b):
    fig = ff.create_2d_density(df[area_a], df[area_b], hist_color="rgb(255, 237, 222)", point_size=3)
    return fig


def get_scatter_with_kernel_density_2d_figure(
    df,
    area_a,
    area_b,
    N,
    z_max,
    n_scatter_samples=500,
    bandwidth=0.1,
    rtol=0.01,
    kernel="epanechnikov",
):
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([Y.ravel(), X.ravel()])

    values = np.vstack([df[area_a].values, df[area_b].values])

    # Using scikit learn
    kde = KernelDensity(bandwidth=bandwidth, rtol=rtol, kernel=kernel).fit(values.T)
    Z = np.reshape(np.exp(kde.score_samples(positions.T)), X.shape)

    dff = df.sample(n_scatter_samples)

    data = [
        go.Contour(
            z=Z,
            x=x,
            y=y,
            zmax=z_max,
            zauto=False,
            colorscale="Blues",
            # reversescale=True,
            opacity=0.9,
            contours=go.contour.Contours(showlines=False),
            colorbar=dict(lenmode="fraction", len=0.9, y=0.42),
        ),
        go.Scatter(
            x=dff[area_b],
            y=dff[area_a],
            mode="markers",
            marker=dict(line_width=0, opacity=0.3, color="#778DA9", symbol="x"),
        ),
        go.Histogram(
            x=df[area_b].values, name=f"x ", yaxis="y2", histnorm="probability density", marker_color="rgb(220,220,220)"
        ),
        go.Histogram(
            y=df[area_a].values, name=f"y ", xaxis="x2", histnorm="probability density", marker_color="rgb(220,220,220)"
        ),
    ]

    layout = go.Layout(
        # title="",
        # font=go.layout.Font(family="Georgia, serif", color="#635F5D"),
        showlegend=False,
        autosize=False,
        width=650,
        height=650,
        xaxis=dict(domain=[0, 0.85], range=[0, 1], showgrid=False, nticks=7, title=area_b, zeroline=False),
        yaxis=dict(domain=[0, 0.85], range=[0, 1], showgrid=False, nticks=7, title=area_a),
        margin=go.layout.Margin(l=20, r=20, b=20, t=20),
        xaxis2=dict(domain=[0.87, 1], showgrid=False, nticks=7, title="", visible=False),
        yaxis2=dict(domain=[0.87, 1], showgrid=False, nticks=7, title="", visible=False),
        # paper_bgcolor='rgb(233,233,233)',
        plot_bgcolor="rgb(255,255,255)",
    )

    return go.Figure(data=data, layout=layout)
