import dash
from dash import dcc, html, Input, Output
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 1. Générer les données
np.random.seed(0)
x = np.linspace(-3, 3, 100)
y = x**3 - x + np.random.normal(scale=3, size=x.shape)

# On reshape pour sklearn
X = x.reshape(-1, 1)

# 2. Créer l'app Dash
app = dash.Dash(__name__)
app.title = "Régression Polynomiale"

app.layout = html.Div([
    html.H2("Régression polynomiale interactive", style={'textAlign': 'center'}),
    dcc.Slider(
        id='degree-slider',
        min=1,
        max=10,
        step=1,
        value=3,
        marks={i: str(i) for i in range(1, 11)},
    ),
    dcc.Graph(id='regression-plot')
])


# 3. Callback pour mettre à jour le graphe
@app.callback(
    Output('regression-plot', 'figure'),
    Input('degree-slider', 'value')
)
def update_plot(degree):
    # Créer les features polynomiales
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)

    # Entraîner le modèle
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Prédictions
    y_pred = model.predict(X_poly)

    # Construire la figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Données', marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x, y=y_pred, mode='lines', name=f'Polynôme degré {degree}', line=dict(color='red')))
    
    fig.update_layout(title=f"Régression polynomiale de degré {degree}", xaxis_title="x", yaxis_title="y")
    return fig

# 4. Lancer l'app
if __name__ == '__main__':
    app.run(debug=True)
