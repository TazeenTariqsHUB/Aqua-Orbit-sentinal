import plotly.graph_objects as go

# Example oceans with approximate lat/lon
oceans = [
    {"name": "Pacific Ocean", "lat": 0, "lon": -160},
    {"name": "Atlantic Ocean", "lat": 0, "lon": -30},
    {"name": "Indian Ocean", "lat": -20, "lon": 80},
    {"name": "Southern Ocean", "lat": -60, "lon": 0},
    {"name": "Arctic Ocean", "lat": 80, "lon": 0},
]

fig = go.Figure()

# Add Earth as a surface (using a gradient)
fig.add_surface(
    z=[[0]*360]*180,
    surfacecolor=[[0]*360]*180,
    colorscale=[[0, "rgb(0,0,50)"], [1, "rgb(0,0,150)"]],
    showscale=False,
)

# Add ocean names as scatter points
for ocean in oceans:
    fig.add_trace(go.Scattergeo(
        lon=[ocean["lon"]],
        lat=[ocean["lat"]],
        text=ocean["name"],
        mode="text",
        textfont=dict(size=14, color="white")
    ))

fig.update_geos(
    showcountries=False,
    showland=False,
    showocean=True,
    oceancolor="darkblue",
    projection_type="orthographic"  # makes it look like a globe
)

# Save as HTML
fig.write_html("plotly_globe.html")