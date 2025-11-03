from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd, numpy as np, seaborn as sns, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
import base64, plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import statsmodels.api as sm  # required for trendline

# ------------------ THEME ------------------
theme = dbc.themes.FLATLY  # Light elegant Bootstrap theme

# ------------------ LOAD DATA ------------------
df = pd.read_csv("C:/Files/Projects/GDP_Analysis/countries of the world.csv", decimal=",")
if "Region" in df.columns:
    for c in ["GDP ($ per capita)", "Literacy (%)", "Agriculture"]:
        df[c] = df.groupby("Region")[c].transform(lambda x: x.fillna(x.median()))
df["Total GDP"] = df["Population"] * df["GDP ($ per capita)"]

# ------------------ MODEL ------------------
target = "GDP ($ per capita)"
X = df.drop(columns=[target, "Country"], errors="ignore")
for c in X.select_dtypes(include=["object"]).columns:
    X[c] = LabelEncoder().fit_transform(X[c].astype(str))
y = df[target].fillna(df[target].median())

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=120, random_state=42)
model.fit(Xtr, ytr)
ypred = model.predict(Xte)

rmse = np.sqrt(mean_squared_error(yte, ypred))
msle = mean_squared_log_error(np.clip(yte, 0, None), np.clip(ypred, 0, None))

# ------------------ APP ------------------
app = Dash(__name__, external_stylesheets=[theme])
app.title = "GDP Analysis Dashboard"

regions = ["All"] + sorted(df["Region"].dropna().unique().tolist())

# ------------------ LAYOUT ------------------
app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand="🌍 GDP Analysis Dashboard",
        color="info",
        dark=False,
        fluid=True,
        className="mb-4 shadow-sm rounded",
    ),

    dbc.Row([
        dbc.Col([
            html.H5("Select Region", className="fw-semibold text-secondary"),
            dcc.Dropdown(
                id="region",
                options=[{"label": r, "value": r} for r in regions],
                value="All",
                clearable=False,
                style={"background": "#ffffff"}
            )
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Average GDP per Capita", className="text-muted"),
                    html.H3(id="avg_gdp", className="fw-bold text-primary")
                ])
            ], style={"background": "#E3F2FD"}, className="shadow-sm border-0")
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Average Literacy", className="text-muted"),
                    html.H3(id="avg_lit", className="fw-bold text-success")
                ])
            ], style={"background": "#E8F5E9"}, className="shadow-sm border-0")
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Countries in Region", className="text-muted"),
                    html.H3(id="count_ctry", className="fw-bold text-info")
                ])
            ], style={"background": "#E1F5FE"}, className="shadow-sm border-0")
        ], width=3),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="bar_gdp"), width=6),
        dbc.Col(dcc.Graph(id="bar_total"), width=6)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="scatter"), width=6),
        dbc.Col(dcc.Graph(id="map"), width=6)
    ]),

    html.Hr(className="my-4"),
    html.H4("📊 Correlation Heatmap", className="text-center text-secondary mb-3"),
    html.Img(id="heatmap", style={
        "width": "80%",
        "display": "block",
        "margin": "auto",
        "borderRadius": "12px",
        "boxShadow": "0px 0px 10px #ccc"
    }),
    html.Hr(),
    html.H5(f"🤖 Model Evaluation — RMSE: {rmse:.2f} | MSLE: {msle:.4f}",
            className="text-center text-secondary my-3 fw-semibold"),
], fluid=True)


# ------------------ CALLBACK ------------------
@app.callback(
    [Output("bar_gdp", "figure"),
     Output("bar_total", "figure"),
     Output("scatter", "figure"),
     Output("map", "figure"),
     Output("heatmap", "src"),
     Output("avg_gdp", "children"),
     Output("avg_lit", "children"),
     Output("count_ctry", "children")],
    Input("region", "value")
)
def update(region):
    data = df if region == "All" else df[df["Region"] == region]
    avg_gdp = f"${data['GDP ($ per capita)'].mean():,.0f}"
    avg_lit = f"{data['Literacy (%)'].mean():.1f}%" if "Literacy (%)" in data else "N/A"
    count_ctry = str(data["Country"].nunique())

    # --- Bar Charts ---
    bar_gdp = px.bar(
        data.nlargest(20, "GDP ($ per capita)"),
        x="Country", y="GDP ($ per capita)",
        color="GDP ($ per capita)", color_continuous_scale="Blues",
        title="Top 20 Countries by GDP per Capita", template="plotly_white"
    )

    bar_total = px.bar(
        data.nlargest(10, "Total GDP"),
        x="Country", y="Total GDP",
        color="Total GDP", color_continuous_scale="Tealgrn",
        title="Top 10 Countries by Total GDP", template="plotly_white"
    )

    # --- Scatter ---
    scatter = px.scatter(
        data, x="Population", y="GDP ($ per capita)",
        color="Region", hover_name="Country",
        title="GDP per Capita vs Population",
        trendline="ols", template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    # --- World Map ---
    map_fig = px.choropleth(
        data, locations="Country", locationmode="country names",
        color="GDP ($ per capita)", color_continuous_scale="Mint",
        hover_name="Country", title="World GDP per Capita", template="plotly_white"
    )

    # --- Enhanced Correlation Heatmap (Wider and Clean) ---
    num_cols = data.select_dtypes(include=[np.number]).columns
    corr = data[num_cols].corr()

    plt.figure(figsize=(10, 7))  # wider figure for readability
    sns.heatmap(
        corr,
        annot=True,
        cmap="crest",
        fmt=".2f",
        annot_kws={"size": 9},
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Correlation Strength"},
    )
    plt.title("Correlation Matrix of Numerical Features", fontsize=13, pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close()
    buf.seek(0)
    heatmap_src = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    return bar_gdp, bar_total, scatter, map_fig, heatmap_src, avg_gdp, avg_lit, count_ctry


# ------------------ RUN SERVER ------------------
if __name__ == "__main__":
    app.run(debug=False, port=8050)
