# %%
import numpy as np
import pandas as pd
import streamlit as st
from scipy.integrate import odeint
from scipy.optimize import minimize
import plotly.graph_objs as go

class NonlinearTimeSeriesModeling:
    def __init__(self, system_type='lorenz'):
        """
        Initialize with different nonlinear dynamical systems
        """
        self.system_types = {
            'lorenz': self.lorenz_system,
            'rossler': self.rossler_system,
            'vanderpol': self.vanderpol_system
        }
        self.current_system = system_type
    
    def lorenz_system(self, state, t, sigma=10, rho=28, beta=8/3):
        # Interpreted as prey-predator-resource with chaotic dynamics
        x, y, z = state
        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z
        return [dx_dt, dy_dt, dz_dt]
    
    def rossler_system(self, state, t, a=0.2, b=0.2, c=5.7):
        # Interpreted as gene regulation/metabolite oscillations
        x, y, z = state
        dx_dt = -y - z
        dy_dt = x + a * y
        dz_dt = b + z * (x - c)
        return [dx_dt, dy_dt, dz_dt]
    
    def vanderpol_system(self, state, t, mu=1.0):
        # Interpreted as membrane potential and recovery variable in cells
        x, y = state
        dx_dt = y
        dy_dt = mu * (1 - x**2) * y - x
        return [dx_dt, dy_dt]

    def generate_time_series(self, initial_conditions=None, 
                             time_points=None, scenario='Generic'):
        if time_points is None:
            time_points = np.linspace(0, 100, 5000)

        # Set default initial conditions based on system type
        if initial_conditions is None:
            if self.current_system == 'vanderpol':
                initial_conditions = [0.5, 0.0]
            elif self.current_system == 'rossler':
                initial_conditions = [1.0, 0.5, 0.2]
            else:  # lorenz
                initial_conditions = [50.0, 10.0, 100.0]

        solution = odeint(
            self.system_types[self.current_system], 
            initial_conditions, 
            time_points
        )

        num_variables = len(initial_conditions)
        column_names = ['x', 'y', 'z'][:num_variables]
        return pd.DataFrame(solution, columns=column_names)

    def calculate_lyapunov_exponent(self, time_series):
        def divergence_rate(params):
            return np.mean(np.log(np.abs(params)))
        
        result = minimize(
            lambda p: -divergence_rate(time_series.values), 
            x0=[1.0], 
            method='Nelder-Mead'
        )
        return result.x[0]
    
    def prediction_uncertainty(self, time_series, forecast_horizon=10):
        rolling_std = time_series.rolling(window=forecast_horizon).std()
        sensitivity = np.gradient(time_series.values.flatten())
        
        return {
            'uncertainty': rolling_std,
            'sensitivity': sensitivity
        }

def plot_3d_trajectory(x, y, z):
    frames = []
    step = 100
    for i in range(0, len(x), step):
        frames.append(
            go.Frame(
                data=[go.Scatter3d(
                    x=x[:i+1],
                    y=y[:i+1],
                    z=z[:i+1],
                    mode='lines',
                    line=dict(color='blue', width=2)
                )],
                name=str(i)
            )
        )

    fig = go.Figure(
        data=[go.Scatter3d(
            x=[x[0]], y=[y[0]], z=[z[0]],
            mode='markers',
            marker=dict(color='red', size=5)
        )],
        layout=go.Layout(
            scene=dict(
                xaxis_title='State 1',
                yaxis_title='State 2',
                zaxis_title='State 3'
            ),
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                buttons=[dict(
                    label='Play',
                    method='animate',
                    args=[None, {"frame": {"duration": 50, "redraw": True},
                                  "transition": {"duration": 0}}]
                )]
            )]
        ),
        frames=frames
    )

    fig.update_layout(
        scene=dict(aspectmode='cube'),
        margin=dict(l=0, r=0, b=0, t=0),
        sliders=[{
            "steps": [
                {
                    "method": "animate",
                    "args": [[f.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    "label": f.name
                } for f in frames
            ],
            "transition": {"duration": 0},
            "x":0.1,
            "len":0.9
        }]
    )
    return fig

def plot_2d_trajectory(x, y):
    frames = []
    step = 100
    for i in range(0, len(x), step):
        frames.append(
            go.Frame(
                data=[go.Scatter(
                    x=x[:i+1],
                    y=y[:i+1],
                    mode='lines',
                    line=dict(color='blue', width=2)
                )],
                name=str(i)
            )
        )

    fig = go.Figure(
        data=[go.Scatter(
            x=[x[0]],
            y=[y[0]],
            mode='markers',
            marker=dict(color='red', size=5)
        )],
        layout=go.Layout(
            xaxis_title='State 1',
            yaxis_title='State 2',
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                buttons=[dict(
                    label='Play',
                    method='animate',
                    args=[None, {"frame": {"duration": 50, "redraw": True},
                                  "transition": {"duration": 0}}]
                )]
            )]
        ),
        frames=frames
    )

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        sliders=[{
            "steps": [
                {
                    "method": "animate",
                    "args": [[f.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    "label": f.name
                } for f in frames
            ],
            "transition": {"duration": 0},
            "x":0.1,
            "len":0.9
        }]
    )

    return fig

def display_biological_context(system_type):
    if system_type == 'Lorenz':
        st.write("**Biological Interpretation (Lorenz):**")
        st.write("- Interpreted as a predator-prey-resource model.")
        st.write("- **Prey (x):** Population of prey species.")
        st.write("- **Predator (y):** Population of predators.")
        st.write("- **Resource (z):** Available resource (e.g., vegetation).")
        st.write("Parameters (σ, ρ, β) affect interaction strengths and resource dynamics.")
    elif system_type == 'Rossler':
        st.write("**Biological Interpretation (Rössler):**")
        st.write("- Consider gene regulation or metabolic cycles.")
        st.write("- **Signal (x):** Concentration of a transcription factor.")
        st.write("- **Regulator (y):** A protein that modulates the signal.")
        st.write("- **Metabolite (z):** Another molecule influencing the cycle.")
        st.write("Parameters (a, b, c) control feedback loops and oscillatory behavior.")
    elif system_type == 'Van der Pol':
        st.write("**Biological Interpretation (Van der Pol):**")
        st.write("- Models oscillations in systems like cardiac cells or neurons.")
        st.write("- **MembranePotential (x):** The cell's membrane voltage.")
        st.write("- **RecoveryVariable (y):** Represents ion channel dynamics.")
        st.write("Parameter μ controls the nature and strength of the oscillations.")

def rename_for_biology(system_type, df):
    if system_type == 'Lorenz':
        df.columns = ['Prey', 'Predator', 'Resource']
    elif system_type == 'Rossler':
        df.columns = ['Signal', 'Regulator', 'Metabolite']
    elif system_type == 'Van der Pol':
        df.columns = ['MembranePotential', 'RecoveryVariable']
    return df

def display_differential_equations(system_type):
    if system_type == 'Lorenz':
        st.write("### **Differential Equations (Lorenz System)**")
        st.latex(r"\frac{dx}{dt} = \sigma (y - x)")
        st.latex(r"\frac{dy}{dt} = x (\rho - z) - y")
        st.latex(r"\frac{dz}{dt} = xy - \beta z")
        st.write("""
        **How It Works**:
        - `x`, `y`, `z`: State variables (e.g., Prey, Predator, Resource).
        - `σ` (sigma), `ρ` (rho), `β` (beta) control interaction strengths and resource dynamics.
        - Known for chaotic behavior under certain parameter values.
        """)

    elif system_type == 'Rossler':
        st.write("### **Differential Equations (Rössler System)**")
        st.latex(r"\frac{dx}{dt} = -y - z")
        st.latex(r"\frac{dy}{dt} = x + ay")
        st.latex(r"\frac{dz}{dt} = b + z(x - c)")
        st.write("""
        **How It Works**:
        - `x`, `y`, `z`: Variables in a feedback loop (e.g., Signal, Regulator, Metabolite).
        - `a`, `b`, `c`: Parameters controlling feedback and oscillatory/chaotic behavior.
        - Useful for modeling cycles in biological systems.
        """)

    elif system_type == 'Van der Pol':
        st.write("### **Differential Equations (Van der Pol Oscillator)**")
        st.latex(r"\frac{dx}{dt} = y")
        st.latex(r"\frac{dy}{dt} = \mu (1 - x^2)y - x")
        st.write("""
        **How It Works**:
        - `x`: State variable (e.g., membrane potential).
        - `y`: Recovery variable (e.g., ion channel dynamics).
        - `μ` (mu): Controls the strength and character of oscillations.
        - Models oscillatory behavior in cells, neurons, or cardiac systems.
        """)

def main():
    st.title('Nonlinear Time Series Modeling - Generic and Biological Interpretations')
    st.write("Choose a system and scenario. Adjust parameters as needed to reflect realistic conditions in biology.")

    # System and scenario selection
    system_type = st.selectbox('Select Dynamic System', ['Lorenz', 'Rossler', 'Van der Pol'])
    scenario = st.selectbox('Select Scenario', ['Generic', 'Biology'])
    
    # Create model and display differential equations
    modeler = NonlinearTimeSeriesModeling(system_type=system_type.lower().replace(' ', ''))
    display_differential_equations(system_type)

    # Generate time series
    time_series = modeler.generate_time_series()

    # If biology scenario, rename columns and show context
    if scenario == 'Biology':
        time_series = rename_for_biology(system_type, time_series)
        display_biological_context(system_type)
    else:
        st.write("**Generic Interpretation:** Variables represent abstract states in a nonlinear system.")

    lyap_exp = modeler.calculate_lyapunov_exponent(time_series)
    uncertainty_data = modeler.prediction_uncertainty(time_series)
    
    st.write(f"**Lyapunov Exponent:** {lyap_exp:.2f}")
    st.write("A positive exponent suggests sensitivity to initial conditions and chaotic dynamics, limiting long-term predictability.")
    st.write("**Time Series Data:**")
    st.line_chart(time_series)
    
    st.write("**Prediction Uncertainty (Rolling Std):**")
    st.line_chart(uncertainty_data['uncertainty'])

    # Trajectory plots
    if system_type in ['Lorenz', 'Rossler']:
        # 3D system
        st.write(f"**3D Phase Space Trajectory ({system_type} System)**")
        x = time_series[time_series.columns[0]].values
        y = time_series[time_series.columns[1]].values
        z = time_series[time_series.columns[2]].values
        fig = plot_3d_trajectory(x, y, z)
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Van der Pol is 2D
        st.write("**2D Phase Space Trajectory (Van der Pol System)**")
        x = time_series[time_series.columns[0]].values
        y = time_series[time_series.columns[1]].values
        fig = plot_2d_trajectory(x, y)
        st.plotly_chart(fig, use_container_width=True)

    st.write("**How to Pick the Best Parameters?**")
    st.write("- **Data-Driven Parameter Estimation:** Use observed data and optimization techniques to fit parameters.")
    st.write("- **Literature and Experiments:** Use known biological/physical values for more realism.")
    st.write("- **Sensitivity Analysis:** Adjust parameters slightly to see which have the largest effect.")
    st.write("- **Stability vs. Chaos:** Pick parameters that yield stable or chaotic behavior based on your modeling goals.")

if __name__ == "__main__":
    main()