import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def calculate_portfolio_return(stock_pct, bond_pct):
    """Calculate expected portfolio return based on allocation"""
    stock_return = 0.07  # 7% expected stock return
    bond_return = 0.03   # 3% expected bond return
    return (stock_pct / 100 * stock_return) + (bond_pct / 100 * bond_return)

def get_asset_allocation(age, risk_tolerance, goal_type, years_to_goal):
    """Calculate recommended asset allocation based on age, risk tolerance, and goal timeline"""
    # Base allocation: 110 - age rule for retirement, adjusted for other goals
    if goal_type == 'Retirement':
        base_stock_pct = max(30, min(90, 110 - age))
    elif goal_type == 'Home Purchase':
        # More conservative for shorter-term goals
        if years_to_goal < 5:
            base_stock_pct = max(20, 40 - years_to_goal * 4)
        else:
            base_stock_pct = min(70, 40 + years_to_goal * 3)
    elif goal_type == 'Education':
        # Moderate approach for education savings
        if years_to_goal < 5:
            base_stock_pct = 30
        else:
            base_stock_pct = min(80, 50 + years_to_goal * 2)
    else:  # General Wealth Building
        base_stock_pct = max(40, min(85, 100 - age))
    
    # Adjust based on risk tolerance
    if risk_tolerance == 'High':
        stock_pct = min(90, base_stock_pct + 10)
    elif risk_tolerance == 'Low':
        stock_pct = max(20, base_stock_pct - 10)
    else:  # Medium
        stock_pct = base_stock_pct
    
    bond_pct = 100 - stock_pct
    return stock_pct, bond_pct

def calculate_goal_projection(current_assets, annual_contribution, portfolio_return, years_to_goal, inflation_rate=0.02):
    """Calculate projections with compound growth"""
    years = np.arange(0, years_to_goal + 1)
    balances = np.zeros_like(years, dtype=float)
    balances[0] = current_assets
    
    # Calculate year-by-year growth
    for i in range(1, len(years)):
        balances[i] = balances[i-1] * (1 + portfolio_return) + annual_contribution
    
    # Inflation-adjusted balances
    inflation_adjusted_balances = balances / (1 + inflation_rate)**years
    
    return years, balances, inflation_adjusted_balances

def monte_carlo_simulation(current_assets, annual_contribution, stock_pct, bond_pct, years_to_goal, num_simulations=1000, inflation_rate=0.02):
    """Run Monte Carlo simulation for portfolio projections"""
    # Historical volatility (standard deviation)
    stock_volatility = 0.18  # ~18% annual volatility for stocks
    bond_volatility = 0.05   # ~5% annual volatility for bonds
    stock_mean_return = 0.07
    bond_mean_return = 0.03
    
    # Calculate portfolio volatility (simplified - assumes no correlation)
    portfolio_volatility = np.sqrt((stock_pct/100 * stock_volatility)**2 + (bond_pct/100 * bond_volatility)**2)
    portfolio_mean_return = (stock_pct/100 * stock_mean_return) + (bond_pct/100 * bond_mean_return)
    
    # Run simulations
    simulations = np.zeros((num_simulations, years_to_goal + 1))
    simulations[:, 0] = current_assets
    
    for sim in range(num_simulations):
        for year in range(1, years_to_goal + 1):
            # Random return for this year
            annual_return = np.random.normal(portfolio_mean_return, portfolio_volatility)
            simulations[sim, year] = simulations[sim, year-1] * (1 + annual_return) + annual_contribution
    
    # Adjust for inflation
    years = np.arange(0, years_to_goal + 1)
    for sim in range(num_simulations):
        simulations[sim, :] = simulations[sim, :] / (1 + inflation_rate)**years
    
    # Calculate percentiles
    percentile_10 = np.percentile(simulations, 10, axis=0)
    percentile_25 = np.percentile(simulations, 25, axis=0)
    percentile_50 = np.percentile(simulations, 50, axis=0)  # Median
    percentile_75 = np.percentile(simulations, 75, axis=0)
    percentile_90 = np.percentile(simulations, 90, axis=0)
    
    return years, percentile_10, percentile_25, percentile_50, percentile_75, percentile_90, simulations

def create_projection_table(years, balances, current_age, step=10):
    """Create a projection table showing balances at regular intervals"""
    table_data = []
    for i, year in enumerate(years):
        if year % step == 0 or year == years[-1]:
            table_data.append({
                'Year': int(year),
                'Age': int(current_age + year),
                'Balance (Inflation-Adjusted)': f"${balances[i]:,.0f}"
            })
    return pd.DataFrame(table_data)

def main():
    st.set_page_config(
        page_title="Cosmic Robo-Advisor",
        page_icon="🚀",
        layout="wide"
    )
    
    # Custom CSS for space theme
    st.markdown("""
    <style>
    .stApp {
        background-image: url("https://www.publicdomainpictures.net/pictures/180000/velka/weltraum-1465833065YL1.jpg");
        background-size: cover;
        background-attachment: fixed;
        color: white;
    }
    /* Make text more visible */
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: white !important;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 50, 0.7);  /* Semi-transparent dark blue */
        color: white;
    }
    /* Button styling for creativity */
    div.stButton > button {
        background-color: #4B0082;  /* Indigo */
        color: white;
        border-radius: 20px;
        border: 2px solid #9370DB;
    }
    div.stButton > button:hover {
        background-color: #9370DB;
        color: white;
    }
    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: rgba(0, 0, 0, 0.5);
        border-radius: 10px;
        padding: 10px;
    }
    /* Add some glow for human touch */
    h1 {
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
    }
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(75, 0, 130, 0.3);  /* Purple tint */
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🚀 Cosmic Robo-Advisor: Launch Your Financial Future!")
    st.markdown("Navigate the stars of finance with personalized investment strategies and cosmic projections.")
    
    # Sidebar for user inputs
    st.sidebar.header("🌌 Your Galactic Profile")
    
    # Goal selection
    goal_type = st.sidebar.selectbox(
        "Financial Quest",
        ["Retirement", "Home Purchase", "Education", "General Wealth Building"],
        help="Choose your primary financial adventure"
    )
    
    # Input form
    age = st.sidebar.number_input("Current Age", min_value=18, max_value=80, value=30, step=1)
    current_assets = st.sidebar.number_input("Current Assets ($)", min_value=0, value=10000, step=1000)
    annual_income = st.sidebar.number_input("Annual Income ($)", min_value=0, value=60000, step=1000)
    savings_rate = st.sidebar.slider("Annual Savings Rate (%)", min_value=1, max_value=50, value=15) / 100
    
    # Goal-specific inputs
    desired_retirement_income = None  # Initialize for retirement goals
    if goal_type == 'Retirement':
        target_age = st.sidebar.number_input("Target Retirement Age", min_value=age+1, max_value=80, value=67, step=1)
        years_to_goal = target_age - age
        desired_retirement_income = st.sidebar.number_input("Desired Annual Retirement Income ($)", min_value=10000, value=50000, step=5000,
                                                            help="Annual income you want in retirement (today's dollars)")
        # Calculate required nest egg using 4% rule (need 25x annual spending)
        goal_amount = desired_retirement_income * 25
    elif goal_type == 'Home Purchase':
        target_down_payment = st.sidebar.number_input("Target Down Payment ($)", min_value=1000, value=80000, step=5000)
        years_to_goal = st.sidebar.number_input("Years to Purchase", min_value=1, max_value=30, value=5, step=1)
        goal_amount = target_down_payment
    elif goal_type == 'Education':
        target_education_cost = st.sidebar.number_input("Target Education Cost ($)", min_value=1000, value=100000, step=5000)
        years_to_goal = st.sidebar.number_input("Years Until Needed", min_value=1, max_value=30, value=10, step=1)
        goal_amount = target_education_cost
    else:  # General Wealth Building
        target_wealth = st.sidebar.number_input("Target Wealth ($)", min_value=1000, value=500000, step=10000)
        years_to_goal = st.sidebar.number_input("Target Timeline (Years)", min_value=1, max_value=50, value=20, step=1)
        goal_amount = target_wealth
    
    risk_tolerance = st.sidebar.selectbox("Risk Tolerance", ["Low", "Medium", "High"], index=1)
    
    # Calculate derived values
    annual_contribution = annual_income * savings_rate
    
    # Get asset allocation recommendation
    stock_pct, bond_pct = get_asset_allocation(age, risk_tolerance, goal_type, years_to_goal)
    portfolio_return = calculate_portfolio_return(stock_pct, bond_pct)
    
    # Display goal-specific header
    goal_icons = {
        'Retirement': '🏖️',
        'Home Purchase': '🏠',
        'Education': '🎓',
        'General Wealth Building': '💰'
    }
    
    st.header(f"{goal_icons[goal_type]} {goal_type} Odyssey")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Profile Constellation")
        
        # Profile metrics
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Current Age", f"{age} years")
        with metric_cols[1]:
            st.metric("Timeline", f"{years_to_goal} years")
        with metric_cols[2]:
            st.metric("Current Assets", f"${current_assets:,}")
        with metric_cols[3]:
            st.metric("Annual Contribution", f"${annual_contribution:,.0f}")
    
    with col2:
        st.subheader("🎯 Stellar Allocation")
        
        # Portfolio allocation pie chart
        allocation_data = pd.DataFrame({
            'Asset Class': ['Stocks', 'Bonds'],
            'Allocation': [stock_pct, bond_pct],
            'Expected Return': ['7%', '3%']
        })
        
        fig_pie = px.pie(allocation_data, values='Allocation', names='Asset Class',
                        title=f"Portfolio Allocation ({risk_tolerance} Risk)",
                        color_discrete_sequence=['#1f77b4', '#ff7f0e'])
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Investment recommendations
    st.header("💼 Investment Galaxy")
    
    rec_col1, rec_col2 = st.columns(2)
    with rec_col1:
        st.subheader("Asset Allocation")
        st.write(f"**Stocks: {stock_pct}%** (Expected return: 7%)")
        st.write(f"**Bonds: {bond_pct}%** (Expected return: 3%)")
        st.write(f"**Expected Portfolio Return: {portfolio_return*100:.1f}%**")
    
    with rec_col2:
        st.subheader("Investment Approach")
        if goal_type == 'Home Purchase' and years_to_goal < 5:
            st.write("✅ Conservative allocation for near-term goal")
            st.write("✅ Focus on capital preservation")
        else:
            st.write("✅ Diversified ETFs (global stocks + total bond market)")
            st.write("✅ Rebalance portfolio annually")
        st.write("✅ Low-cost index funds recommended")
        st.write("✅ Dollar-cost averaging with regular contributions")
    
    # Tax-Advantaged Account Optimization (for retirement and general wealth)
    if goal_type in ['Retirement', 'General Wealth Building']:
        st.header("💳 Tax Nebula Optimization")
        
        with st.expander("Optimize Your Retirement Account Contributions", expanded=False):
            st.markdown("Maximize tax benefits by strategically allocating contributions across account types.")
            
            # Tax bracket estimation
            tax_bracket = st.selectbox(
                "Current Tax Bracket",
                ["12% (Low)", "22% (Medium)", "24% (Medium-High)", "32% (High)", "35% (High)", "37% (Highest)"],
                index=1
            )
            
            expected_retirement_bracket = st.selectbox(
                "Expected Retirement Tax Bracket",
                ["12% (Low)", "22% (Medium)", "24% (Medium-High)", "32% (High)"],
                index=0
            )
            
            # Extract percentages
            current_tax_rate = float(tax_bracket.split('%')[0]) / 100
            retirement_tax_rate = float(expected_retirement_bracket.split('%')[0]) / 100
            
            # Calculate optimization
            employer_match_pct = st.slider("Employer 401k Match (%)", 0, 10, 3) / 100
            employer_match_limit = st.number_input("Match Limit (% of salary)", 0, 10, 6) / 100
            
            # Account allocation recommendations
            st.subheader("Recommended Contribution Strategy")
            
            # Calculate optimal allocation
            max_employer_match = annual_income * employer_match_limit
            recommended_401k = min(annual_contribution, max_employer_match)
            remaining_after_401k = annual_contribution - recommended_401k
            
            # Roth vs Traditional decision
            if current_tax_rate < retirement_tax_rate:
                roth_recommendation = "Roth IRA/401k"
                roth_reason = "Your current tax rate is lower than expected retirement rate - pay taxes now"
            else:
                roth_recommendation = "Traditional IRA/401k"
                roth_reason = "Your current tax rate is higher - defer taxes to retirement"
            
            # IRA limits (2024)
            ira_limit = 6500 if age < 50 else 7500
            recommended_ira = min(remaining_after_401k, ira_limit)
            remaining_after_ira = remaining_after_401k - recommended_ira
            
            # Display recommendations
            col_tax1, col_tax2 = st.columns(2)
            
            with col_tax1:
                st.markdown("**1. 401(k) Contribution**")
                st.metric("Recommended Amount", f"${recommended_401k:,.0f}")
                st.write(f"**Reason**: Capture {employer_match_pct*100:.0f}% employer match up to {employer_match_limit*100:.0f}% of salary")
            
            with col_tax2:
                st.markdown(f"**2. {roth_recommendation} Contribution**")
                st.metric("Recommended Amount", f"${recommended_ira:,.0f}")
                st.write(f"**Reason**: {roth_reason}")
            
            if remaining_after_ira > 0:
                st.markdown("**3. Taxable Brokerage Account**")
                st.metric("Recommended Amount", f"${remaining_after_ira:,.0f}")
                st.write("**Reason**: Additional savings after maxing out tax-advantaged accounts")
    
    # Financial Projections
    st.header("📈 Financial Projections")
    
    # Calculate projections
    years, balances, inflation_adjusted_balances = calculate_goal_projection(
        current_assets, annual_contribution, portfolio_return, years_to_goal
    )
    
    # Display goal metrics
    st.subheader(f"{goal_type} Goal")
    
    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("Target Amount", f"${goal_amount:,.0f}")
    with metric_cols[1]:
        st.metric("Projected Value (Inflation-Adjusted)", f"${inflation_adjusted_balances[-1]:,.0f}",
                 delta=f"{(inflation_adjusted_balances[-1] - goal_amount)/goal_amount*100:+.1f}%")
    with metric_cols[2]:
        success_threshold = 0.80 if goal_type == 'Retirement' else 0.90
        st.metric("Confidence Level", "Calculating...")
    
    # Monte Carlo Simulation
    st.subheader("Monte Carlo Simulation")
    st.markdown("Explore the range of possible outcomes based on market volatility.")
    
    years, p10, p25, p50, p75, p90, sims = monte_carlo_simulation(
        current_assets, annual_contribution, stock_pct, bond_pct, years_to_goal
    )
    
    # Update confidence level
    success_count = np.sum(sims[:, -1] >= goal_amount)
    success_rate = (success_count / len(sims)) * 100
    confidence_label = "High" if success_rate >= 80 else "Moderate" if success_rate >= 60 else "Low"
    
    metric_cols[2].metric("Confidence Level", confidence_label, delta=f"{success_rate:.1f}%")
    
    # Plot Monte Carlo results
    fig_mc = go.Figure()
    
    fig_mc.add_trace(go.Scatter(
        x=years,
        y=p90,
        mode='lines',
        name='90th Percentile',
        line=dict(color='rgba(31, 119, 180, 0.3)'),
        showlegend=True
    ))
    fig_mc.add_trace(go.Scatter(
        x=years,
        y=p10,
        mode='lines',
        name='10th Percentile',
        fill='tonexty',
        fillcolor='rgba(31, 119, 180, 0.2)',
        line=dict(color='rgba(31, 119, 180, 0.3)'),
        showlegend=True
    ))
    fig_mc.add_trace(go.Scatter(
        x=years,
        y=p75,
        mode='lines',
        name='75th Percentile',
        line=dict(color='rgba(44, 160, 44, 0.3)'),
        showlegend=True
    ))
    fig_mc.add_trace(go.Scatter(
        x=years,
        y=p25,
        mode='lines',
        name='25th Percentile',
        fill='tonexty',
        fillcolor='rgba(44, 160, 44, 0.2)',
        line=dict(color='rgba(44, 160, 44, 0.3)'),
        showlegend=True
    ))
    fig_mc.add_trace(go.Scatter(
        x=years,
        y=p50,
        mode='lines',
        name='Median',
        line=dict(color='#1f77b4', width=3),
        showlegend=True
    ))
    fig_mc.add_trace(go.Scatter(
        x=[years[0], years[-1]],
        y=[goal_amount, goal_amount],
        mode='lines',
        name='Goal Amount',
        line=dict(color='#ff7f0e', dash='dash', width=2),
        showlegend=True
    ))
    
    fig_mc.update_layout(
        title="Projected Portfolio Value (Inflation-Adjusted)",
        xaxis_title="Years",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        height=450,
        yaxis_tickformat='$,.0f',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    
    st.plotly_chart(fig_mc, use_container_width=True)
    
    # Projection table
    st.subheader("Projection Milestones")
    projection_table = create_projection_table(years, inflation_adjusted_balances, age)
    st.dataframe(projection_table, hide_index=True, use_container_width=True)
    
    # Strategy Comparison
    st.header("⚖️ Strategy Comparison")
    
    with st.expander("Compare Different Allocation Strategies", expanded=False):
        st.markdown("See how different stock/bond allocations impact your goal.")
        
        strategies = [
            {'name': 'Aggressive (80/20)', 'stock_pct': 80, 'bond_pct': 20},
            {'name': 'Balanced (60/40)', 'stock_pct': 60, 'bond_pct': 40},
            {'name': 'Conservative (40/60)', 'stock_pct': 40, 'bond_pct': 60},
            {'name': 'Your Strategy', 'stock_pct': stock_pct, 'bond_pct': bond_pct},
            {'name': '100% Stocks', 'stock_pct': 100, 'bond_pct': 0},
            {'name': '100% Bonds', 'stock_pct': 0, 'bond_pct': 100}
        ]
        
        success_rates = []
        for strategy in strategies:
            strategy_name = strategy['name']
            temp_stock_pct = strategy['stock_pct']
            temp_bond_pct = strategy['bond_pct']
            temp_portfolio_return = calculate_portfolio_return(temp_stock_pct, temp_bond_pct)
            
            _, _, _, _, _, _, sims = monte_carlo_simulation(
                current_assets, annual_contribution, temp_stock_pct, temp_bond_pct, years_to_goal
            )
            success_count = np.sum(sims[:, -1] >= goal_amount)
            success_rate = (success_count / len(sims)) * 100
            success_rates.append({
                'Strategy': strategy_name,
                'Success Rate': f"{success_rate:.1f}%",
                'Success Rate (num)': success_rate
            })
        
        success_df = pd.DataFrame(success_rates)
        success_df = success_df.sort_values('Success Rate (num)', ascending=False)
        display_success_df = success_df.drop('Success Rate (num)', axis=1)
        
        st.dataframe(display_success_df, hide_index=True)
        
        # Bar chart of success rates
        fig_success = px.bar(
            success_df,
            x='Strategy',
            y='Success Rate (num)',
            title='Probability of Meeting Goal by Strategy',
            labels={'Success Rate (num)': 'Success Rate (%)'},
            color='Success Rate (num)',
            color_continuous_scale='RdYlGn'
        )
        fig_success.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_success, use_container_width=True)
    
    # Portfolio Rebalancing & Historical Backtesting
    st.header("⚖️ Portfolio Rebalancing & Historical Analysis")
    
    with st.expander("Rebalancing Recommendations & Historical Performance", expanded=False):
        st.markdown("Understand when to rebalance and see how your strategy would have performed historically.")
        
        # Rebalancing section
        st.subheader("📊 Rebalancing Strategy")
        
        # Simulate portfolio drift
        simulated_years = min(5, years_to_goal)
        drift_scenario = st.radio(
            "Market Scenario",
            ["Bull Market (Stocks +20%)", "Bear Market (Stocks -15%)", "Balanced (Expected Returns)"],
            index=2
        )
        
        # Calculate drift
        if drift_scenario == "Bull Market (Stocks +20%)":
            stock_return_drift = 0.20
            bond_return_drift = 0.03
        elif drift_scenario == "Bear Market (Stocks -15%)":
            stock_return_drift = -0.15
            bond_return_drift = 0.03
        else:
            stock_return_drift = 0.07
            bond_return_drift = 0.03
        
        # Current allocation value
        current_stock_value = current_assets * (stock_pct / 100)
        current_bond_value = current_assets * (bond_pct / 100)
        
        # After drift
        after_stock_value = current_stock_value * (1 + stock_return_drift)
        after_bond_value = current_bond_value * (1 + bond_return_drift)
        total_after = after_stock_value + after_bond_value
        
        # New percentages after drift
        new_stock_pct = (after_stock_value / total_after) * 100
        new_bond_pct = (after_bond_value / total_after) * 100
        
        drift_cols = st.columns(3)
        with drift_cols[0]:
            st.metric("Current Allocation", f"{stock_pct}/{bond_pct}", 
                     help="Stock/Bond split")
        with drift_cols[1]:
            st.metric("After Market Movement", f"{new_stock_pct:.1f}/{new_bond_pct:.1f}",
                     delta=f"{new_stock_pct - stock_pct:+.1f}% stocks")
        with drift_cols[2]:
            drift_amount = abs(new_stock_pct - stock_pct)
            if drift_amount > 5:
                st.metric("Rebalance Needed?", "Yes", delta="High drift",
                         delta_color="inverse")
            else:
                st.metric("Rebalance Needed?", "No", delta="On target")
        
        # Rebalancing recommendations
        st.markdown("**Rebalancing Guidelines:**")
        rebalance_col1, rebalance_col2 = st.columns(2)
        
        with rebalance_col1:
            st.write("**When to Rebalance:**")
            st.write("✅ When allocation drifts >5% from target")
            st.write("✅ At least annually (recommended)")
            st.write("✅ After major market movements")
            st.write("✅ When adding new contributions")
        
        with rebalance_col2:
            st.write("**How to Rebalance:**")
            if new_stock_pct > stock_pct + 5:
                stocks_to_sell = after_stock_value - (total_after * stock_pct / 100)
                st.write(f"🔴 Sell ${stocks_to_sell:,.0f} of stocks")
                st.write(f"🟢 Buy ${stocks_to_sell:,.0f} of bonds")
            elif new_stock_pct < stock_pct - 5:
                stocks_to_buy = (total_after * stock_pct / 100) - after_stock_value
                st.write(f"🟢 Buy ${stocks_to_buy:,.0f} of stocks")
                st.write(f"🔴 Sell ${stocks_to_buy:,.0f} of bonds")
            else:
                st.write("✅ Portfolio is balanced")
                st.write("✅ No action needed")
            st.write("💡 Use new contributions to rebalance")
            st.write("💡 Consider tax implications")
        
        # Historical backtesting
        st.subheader("📈 Historical Backtesting")
        st.markdown("See how your allocation would have performed in different historical periods.")
        
        # Historical scenarios (simplified - based on real market data)
        historical_periods = {
            "2008 Financial Crisis": {
                "stocks": [-37.0, 26.5, 15.1, 2.1],  # 2008-2011
                "bonds": [5.2, -11.1, 8.5, 7.8],
                "years": [2008, 2009, 2010, 2011]
            },
            "2010s Bull Market": {
                "stocks": [15.1, 2.1, 16.0, 32.4, 13.7],  # 2010-2014
                "bonds": [8.5, 7.8, 4.2, -2.0, 6.0],
                "years": [2010, 2011, 2012, 2013, 2014]
            },
            "COVID-19 (2020-2021)": {
                "stocks": [-18.0, 18.4, 28.7],  # 2020-2022
                "bonds": [7.5, -1.5, -13.0],
                "years": [2020, 2021, 2022]
            },
            "Recent Years (2020-2024)": {
                "stocks": [-18.0, 18.4, 28.7, -18.1, 26.3],  # 2020-2024
                "bonds": [7.5, -1.5, -13.0, 5.5, 5.5],
                "years": [2020, 2021, 2022, 2023, 2024]
            }
        }
        
        selected_period = st.selectbox(
            "Select Historical Period",
            list(historical_periods.keys())
        )
        
        period_data = historical_periods[selected_period]
        
        # Calculate historical performance
        initial_investment = 10000
        stock_allocation = initial_investment * (stock_pct / 100)
        bond_allocation = initial_investment * (bond_pct / 100)
        
        portfolio_values = [initial_investment]
        stock_values = [stock_allocation]
        bond_values = [bond_allocation]
        
        for i in range(len(period_data["stocks"])):
            stock_return = period_data["stocks"][i] / 100
            bond_return = period_data["bonds"][i] / 100
            
            stock_allocation = stock_allocation * (1 + stock_return)
            bond_allocation = bond_allocation * (1 + bond_return)
            
            total_value = stock_allocation + bond_allocation
            portfolio_values.append(total_value)
            stock_values.append(stock_allocation)
            bond_values.append(bond_allocation)
        
        years_list = [period_data["years"][0] - 1] + period_data["years"]
        
        # Calculate metrics
        total_return = ((portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]) * 100
        annualized_return = (((portfolio_values[-1] / portfolio_values[0]) ** (1/len(period_data["stocks"]))) - 1) * 100
        
        backtest_cols = st.columns(3)
        with backtest_cols[0]:
            st.metric("Initial Investment", f"${initial_investment:,}")
        with backtest_cols[1]:
            st.metric("Final Value", f"${portfolio_values[-1]:,.0f}",
                     delta=f"{total_return:+.1f}%")
        with backtest_cols[2]:
            st.metric("Annualized Return", f"{annualized_return:.1f}%")
        
        # Historical performance chart
        fig_backtest = go.Figure()
        
        fig_backtest.add_trace(go.Scatter(
            x=years_list,
            y=portfolio_values,
            mode='lines+markers',
            name=f'Your Portfolio ({stock_pct}/{bond_pct})',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))
        
        # Add 100% stocks comparison
        stocks_only_values = [initial_investment]
        stocks_only = initial_investment
        for stock_return in period_data["stocks"]:
            stocks_only = stocks_only * (1 + stock_return/100)
            stocks_only_values.append(stocks_only)
        
        fig_backtest.add_trace(go.Scatter(
            x=years_list,
            y=stocks_only_values,
            mode='lines+markers',
            name='100% Stocks',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            marker=dict(size=6)
        ))
        
        # Add 100% bonds comparison
        bonds_only_values = [initial_investment]
        bonds_only = initial_investment
        for bond_return in period_data["bonds"]:
            bonds_only = bonds_only * (1 + bond_return/100)
            bonds_only_values.append(bonds_only)
        
        fig_backtest.add_trace(go.Scatter(
            x=years_list,
            y=bonds_only_values,
            mode='lines+markers',
            name='100% Bonds',
            line=dict(color='#2ca02c', width=2, dash='dash'),
            marker=dict(size=6)
        ))
        
        fig_backtest.update_layout(
            title=f"Historical Performance: {selected_period}",
            xaxis_title="Year",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified',
            height=450,
            yaxis_tickformat='$,.0f',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        
        st.plotly_chart(fig_backtest, use_container_width=True)
        
        # Performance summary
        stocks_return = ((stocks_only_values[-1] - initial_investment) / initial_investment) * 100
        bonds_return = ((bonds_only_values[-1] - initial_investment) / initial_investment) * 100
        
        st.markdown("**Performance Summary:**")
        summary_df = pd.DataFrame({
            'Strategy': [f'Your Portfolio ({stock_pct}/{bond_pct})', '100% Stocks', '100% Bonds'],
            'Final Value': [f"${portfolio_values[-1]:,.0f}", f"${stocks_only_values[-1]:,.0f}", f"${bonds_only_values[-1]:,.0f}"],
            'Total Return': [f"{total_return:.1f}%", f"{stocks_return:.1f}%", f"{bonds_return:.1f}%"],
            'Risk Level': ['Balanced', 'High', 'Low']
        })
        st.dataframe(summary_df, hide_index=True)
        
        st.info(f"""
        **Key Insight**: During {selected_period}, your {stock_pct}/{bond_pct} allocation provided 
        {"better" if total_return > bonds_return else "similar"} returns {"with less volatility than 100% stocks" if stock_pct < 100 else "but with higher volatility"}.
        Diversification helps smooth out market turbulence while still capturing growth.
        """)
    
    # Key assumptions and disclaimers
    st.header("📝 Key Assumptions & Disclaimers")
    
    assumptions_col1, assumptions_col2 = st.columns(2)
    
    with assumptions_col1:
        st.subheader("Assumptions")
        st.write("• Stock market return: 7% annually")
        st.write("• Bond market return: 3% annually")
        st.write("• Inflation rate: 2% annually")
        st.write("• Consistent annual contributions")
        st.write("• No major market disruptions")
    
    with assumptions_col2:
        st.subheader("Important Notes")
        st.write("• Past performance doesn't guarantee future results")
        st.write("• Market returns can vary significantly")
        st.write("• Consider tax implications of investments")
        st.write("• Review and adjust strategy regularly")
        st.write("• Consult a financial advisor for personalized advice")
    
    # Goal-specific insights
    if goal_type == 'Retirement':
        if years_to_goal > 30:
            st.info("💡 **Long time horizon advantage**: With over 30 years until retirement, you can take advantage of compound growth and ride out market volatility.")
        elif years_to_goal < 10:
            st.warning("⚠️ **Limited time horizon**: Consider increasing contributions or adjusting retirement timeline for better outcomes.")
    elif goal_type == 'Home Purchase':
        if years_to_goal < 3:
            st.info("💡 **Short timeline**: Consider a high-yield savings account or CDs for funds needed within 3 years to reduce market risk.")
    
    if savings_rate < 0.10:
        st.warning("⚠️ **Low savings rate**: Consider increasing your savings rate to 10-15% or more for better outcomes.")
    elif savings_rate > 0.20:
        st.success("🎉 **Excellent savings rate**: You're building wealth at an impressive pace!")

if __name__ == "__main__":
    main()
