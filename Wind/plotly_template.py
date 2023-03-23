import plotly.graph_objects as go


my_template = go.layout.Template()

my_template.layout.legend = dict(
    yanchor="top",
    xanchor="right",
    y=0.95,
    x=0.95,
    bgcolor="rgba(255, 255, 255, 0.8)",
    bordercolor="black",
    borderwidth=1,
    font=dict(size=12, family="Times New Roman"),
)

my_template.layout.xaxis = dict(
    zeroline=False,
    zerolinewidth=1,
    zerolinecolor="Black",
    mirror="all",
    showgrid=False,
    ticks="inside",
    showline=True,
    tickfont=dict(size=14, family="Times New Roman"),
    title_font=dict(size=14, family="Times New Roman"),
)
my_template.layout.yaxis = dict(
    zeroline=False,
    zerolinewidth=1,
    zerolinecolor="Black",
    mirror="all",
    showgrid=False,
    ticks="inside",
    showline=True,
    tickfont=dict(size=14, family="Times New Roman"),
    title_font=dict(size=14, family="Times New Roman"),
)
my_template.layout.margin = dict(l=70, r=10, b=50, t=10)
my_template.layout.width = 700
my_template.layout.height = 400
