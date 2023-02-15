import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from geopy.distance import geodesic
from scipy.optimize import curve_fit
from sklearn.neighbors import KernelDensity

pd.set_option("plotting.backend", "plotly")


def get_corr_figure(df):
    corr = df.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    df_mask = corr.mask(mask).round(2)

    fig = ff.create_annotated_heatmap(
        z=df_mask.to_numpy(),
        x=df_mask.columns.tolist(),
        y=df_mask.columns.tolist(),
        # colorscale=px.colors.diverging.RdBu,
        colorscale=px.colors.sequential.Blackbody_r,
        hoverinfo="none",  # Shows hoverinfo for null values
        showscale=True,
        ygap=1,
        xgap=1,
        zmid=0,
    )

    fig.update_xaxes(side="bottom")

    fig.update_layout(
        #     title_text='Heatmap',
        #     title_x=0.5,
        #     width=1000,
        height=1000,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_zeroline=False,
        yaxis_zeroline=False,
        #     yaxis_autorange='reversed',
        template="plotly_white",
    )
    return fig


def get_hours_shift_figure(df, n_shifts, quantile):
    df["Sum"] = df.mean(axis=1)
    df_shift = pd.concat([df.diff(i).quantile(q=quantile) for i in range(n_shifts)], axis=1).T

    df_shift.index = [i for i in range(n_shifts)]

    df_shift.index.name = "hour shift"

    fig = df_shift.plot(
        title=f"quantile: {quantile}", template="simple_white", labels=dict(value="Absolute change in power output")
    )

    df = df.drop(columns=["Sum"])
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
    fig.update_layout(title=f"Period: {resample_period}", template="simple_white", xaxis_title="time", yaxis_title=" ")

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


def get_corr_distance_figure(df, df_locations):

    locs = []
    for _, row in df_locations.iterrows():
        locs.append((row["lat"], row["lon"]))

    distances = []
    for i in locs:
        distances.append([])
        for j in locs:
            distances[-1].append(geodesic(i, j).km)

    df_corr = df.corr()
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

    # Fit an exponential function
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c

    popt, pcov = curve_fit(
        func, df_corr_dist["Distance [km]"].values, df_corr_dist["Correlation"].values, p0=[1, 0.005, 0]
    )

    xn = np.linspace(df_corr_dist["Distance [km]"].min(), df_corr_dist["Distance [km]"].max(), 2500)

    data = [
        go.Scatter(
            x=df_corr_dist["Distance [km]"],
            y=df_corr_dist["Correlation"],
            text=df_corr_dist["Span"],
            marker_color="#697771",
            mode="markers",
            name="Corelations",
        ),
        go.Scatter(
            x=xn,
            y=func(xn, *popt),
            marker_color="red",
            name=f"{popt[0]:.2f} * exp(-1/({1/popt[1]:.1f}) * x) + {popt[2]:.2f}",
        ),
    ]

    fig = go.Figure(data=data)

    fig.update_layout(
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        template="plotly_white",
        title=f"",
        xaxis_title="distance [km]",
        yaxis_title="correlation",
        width=1000,
        height=600,
    )

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
        color_discrete_sequence=px.colors.sequential.Blues,
    )
    fig.update_traces(opacity=0.3)

    x = list(dff["date"])
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
            ),
            go.Scatter(x=x, y=y, name="Mean wind", line=dict(color="#3a4454"), mode="lines"),
        ]
    )

    fig.update_xaxes(
        dtick="M1",
        tickformat="%j",
        ticklabelmode="period",
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        ),
    )
    fig.update_layout(
        title=f"Area: {area}, with resample period of {resample_period}",
        template="simple_white",
        xaxis_title="day",
        yaxis_title=" ",
    )
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
        title=f"Resample period {resample_period}", template="simple_white", xaxis_title="day", yaxis_title=" "
    )
    return fig


def get_scatter_2d_figure(df, area_a, area_b):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[area_a], y=df[area_b], mode="markers", marker=dict(line_width=1, opacity=0.3)))
    fig.update_layout(
        title=f"", template="simple_white", xaxis_title=f"{area_a}", yaxis_title=f"{area_b}", width=600, height=600
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
        template="simple_white",
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
    df, area_a, area_b, N, n_scatter_samples=500, bandwidth=0.1, rtol=0.01, kernel="epanechnikov"
):

    x = np.linspace(-0.5, 1.5, N)
    y = np.linspace(-0.5, 1.5, N)
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
            colorscale="Viridis",
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
            x=df[area_b].values, name=f"x ", yaxis="y2", histnorm="probability density", marker_color="rgb(200,200,200)"
        ),
        go.Histogram(
            y=df[area_a].values, name=f"y ", xaxis="x2", histnorm="probability density", marker_color="rgb(200,200,200)"
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
        xaxis2=dict(domain=[0.87, 1], showgrid=False, nticks=7, title=""),
        yaxis2=dict(domain=[0.87, 1], showgrid=False, nticks=7, title=""),
        # paper_bgcolor='rgb(233,233,233)',
        plot_bgcolor="rgb(255,255,255)",
    )

    return go.Figure(data=data, layout=layout)
