import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from mf_analyzer import FlexiCapComparator
import pandas as pd
from datetime import datetime
import numpy as np
import plotly
import yfinance
import requests

st.set_page_config(
    page_title="Flexi Cap Funds Analysis",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stPlotlyChart {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stDataFrame td {
        white-space: pre-wrap !important;
        padding: 15px !important;
        line-height: 1.6 !important;
        vertical-align: top !important;
        font-size: 13px !important;
        width: 220px !important;  /* Increased width */
        max-height: none !important;
        height: auto !important;
    }
    .stDataFrame th {
        padding: 15px !important;
        font-weight: bold !important;
        background-color: #f0f2f6 !important;
        width: 220px !important;  /* Increased width */
    }
    /* Force expanded view */
    .stDataFrame [data-testid="StyledDataFrameDataCell"] {
        width: 200px !important;
        min-width: 200px !important;
        max-width: 200px !important;
        overflow: visible !important;
        height: auto !important;
        max-height: none !important;
    }
    /* Remove any collapse/expand behavior */
    .stDataFrame [data-testid="StyledDataFrameDataCell"] > div {
        max-height: none !important;
        overflow: visible !important;
    }
    /* Ensure table takes full width */
    [data-testid="stDataFrame"] {
        width: 100% !important;
        max-width: 100% !important;
    }
    /* Remove scrollbars */
    .stDataFrame div[data-testid="stHorizontalBlock"] {
        overflow: visible !important;
    }
    /* Add new styles for NAV indicators */
    .nav-increase {
        color: #00cc00;  /* Green */
    }
    .nav-decrease {
        color: #ff0000;  /* Red */
    }
    .nav-neutral {
        color: #0066ff;  /* Blue */
    }
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 13px;
        font-family: 'Segoe UI', sans-serif;
    }
    .styled-table thead tr {
        background-color: #f0f2f6;
        text-align: left;
    }
    .styled-table th,
    .styled-table td {
        padding: 15px;
        vertical-align: top;
        border: 1px solid #ddd;
        white-space: pre-line;
        line-height: 1.6;
        width: 200px;
    }
    .nav-value {
        display: block;
        margin-bottom: 5px;
    }
    .nav-return {
        display: block;
    }
    </style>
""", unsafe_allow_html=True)

def create_nav_chart(comparator):
    """Create NAV comparison chart"""
    data = []
    for fund_name, analyzer in comparator.analyzers.items():
        if analyzer.nav_data is not None:
            df = analyzer.nav_data.copy()
            df['Fund'] = fund_name
            data.append(df)
    
    if data:
        df_combined = pd.concat(data)
        fig = px.line(df_combined, x='date', y='nav', color='Fund',
                     title='NAV Comparison',
                     labels={'nav': 'NAV Value', 'date': 'Date'})
        fig.update_layout(height=400)
        return fig
    return None

def create_returns_chart(comparison):
    """Create returns comparison chart"""
    returns = comparison['returns']['monthly_returns']
    fig = go.Figure(data=[
        go.Bar(x=list(returns.keys()), 
               y=[float(v) for v in returns.values()],
               text=[f"{float(v):.2f}%" for v in returns.values()],
               textposition='auto')
    ])
    fig.update_layout(
        title='Monthly Returns Comparison',
        xaxis_title='Fund',
        yaxis_title='Return (%)',
        height=400
    )
    return fig

def get_sector_change_reasons(sector, direction):
    """Get reasons for sectoral changes based on sector and direction"""
    reasons_map = {
        'IT': {
            'increase': [
                'Strong deal pipeline in US market',
                'Increasing tech spending by clients',
                'Digital transformation trends'
            ],
            'decrease': [
                'Client budget constraints',
                'Global tech slowdown',
                'Margin pressure from wage inflation'
            ]
        },
        'Banking': {
            'increase': [
                'Improving credit growth',
                'Better asset quality',
                'Rising interest rate environment'
            ],
            'decrease': [
                'Rising NPAs',
                'Regulatory concerns',
                'Competitive pressure from fintech'
            ]
        },
        'FMCG': {
            'increase': [
                'Rural demand recovery',
                'Moderating input costs',
                'New product launches'
            ],
            'decrease': [
                'High inflation impact',
                'Weak consumer sentiment',
                'Competitive intensity'
            ]
        },
        'Auto': {
            'increase': [
                'Strong order book',
                'Easing chip shortage',
                'New model launches'
            ],
            'decrease': [
                'Rising input costs',
                'EV transition challenges',
                'Demand slowdown'
            ]
        },
        'Pharma': {
            'increase': [
                'US FDA clearances',
                'Specialty product launches',
                'Domestic market growth'
            ],
            'decrease': [
                'US pricing pressure',
                'Regulatory challenges',
                'R&D cost increases'
            ]
        }
    }
    
    # Default reasons if sector not in map
    default_reasons = {
        'increase': [
            'Positive sector outlook',
            'Attractive valuations',
            'Growth opportunities'
        ],
        'decrease': [
            'Valuation concerns',
            'Sector headwinds',
            'Better opportunities elsewhere'
        ]
    }
    
    return reasons_map.get(sector, default_reasons)[direction]

def get_nav_change_reasons(change_percent, sector_changes, portfolio_changes):
    """Analyze investment decisions that led to NAV changes"""
    decisions = []
    
    if change_percent > 0:
        prefix = "📈 Investment decisions that contributed to NAV increase:"
    else:
        prefix = "📉 Investment decisions that led to NAV decrease:"
    
    # Portfolio restructuring decisions
    if portfolio_changes:
        # Analyze new additions
        if portfolio_changes['new_additions']:
            for add in portfolio_changes['new_additions']:
                decisions.append(f"• Strategic addition of {add['name']} ({add['weight']}%) to capture {add['sector']} opportunity")
        
        # Analyze exits
        if portfolio_changes['exits']:
            for exit in portfolio_changes['exits']:
                decisions.append(f"• Timely exit from {exit['name']} (was {exit['prev_weight']}%) based on valuation/risk assessment")
    
    # Sector allocation decisions
    if sector_changes:
        positive_sectors = [(s, p) for s, p in sector_changes.items() if p > 1.0]
        negative_sectors = [(s, p) for s, p in sector_changes.items() if p < -1.0]
        
        if positive_sectors:
            top_sector = max(positive_sectors, key=lambda x: x[1])
            decisions.append(f"• Increased exposure to {top_sector[0]} sector (+{top_sector[1]:.1f}%) anticipating growth")
        
        if negative_sectors:
            worst_sector = min(negative_sectors, key=lambda x: x[1])
            decisions.append(f"• Reduced exposure to {worst_sector[0]} sector ({worst_sector[1]:.1f}%) to manage risk")
    
    # Market positioning decisions
    if abs(change_percent) > 3:
        if change_percent > 0:
            decisions.append("• Maintained higher equity exposure in positive market conditions")
        else:
            decisions.append("• Increased defensive positions in challenging market")
    
    # Add default decision if no specific ones found
    if not decisions:
        decisions.append("• Maintained existing portfolio structure based on long-term strategy")
    
    # Return with HTML line breaks instead of \n
    return f"{prefix}<br>" + "<br>".join(decisions)

def main():
    st.title("📊 Flexi Cap Funds Comparison Agent")
    
    with st.spinner("Analyzing funds..."):
        comparator = FlexiCapComparator()
        analysis = comparator.analyze_all_funds()
        
        if analysis is None:
            st.error("Unable to perform comparison due to insufficient fund data")
            st.warning("""
                Possible reasons:
                - Fund NAV data not publicly available
                - API access issues
                - Invalid fund codes
                
                Please check the logs for specific fund status.
            """)
            return

    # Display fund metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Funds Analyzed",
            len(comparator.funds)
        )
    
    with col2:
        best_return = max(analysis['returns']['monthly_returns'].items(), 
                         key=lambda x: float(x[1]))
        st.metric(
            "Best Performing Fund",
            best_return[0],
            f"{float(best_return[1]):.2f}%"
        )
    
    with col3:
        avg_return = sum(float(v) for v in analysis['returns']['monthly_returns'].values()) / len(analysis['returns']['monthly_returns'])
        st.metric(
            "Average Monthly Return",
            f"{avg_return:.2f}%"
        )

    # NAV Comparison Chart
    st.subheader("NAV Trends")
    nav_data = []
    for fund_name, analyzer in comparator.analyzers.items():
        if analyzer.nav_data is not None:
            df = analyzer.nav_data.copy()
            df['Fund'] = fund_name
            nav_data.append(df)
    
    if nav_data:
        df_combined = pd.concat(nav_data)
        fig = px.line(df_combined, 
                     x='date', 
                     y='nav', 
                     color='Fund',
                     title='NAV Comparison')
        st.plotly_chart(fig, use_container_width=True)

    # Returns and Risk Analysis
    st.subheader("Returns & Risk Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly Returns Chart
        returns_data = analysis['returns']['monthly_returns']
        fig = go.Figure(data=[
            go.Bar(x=list(returns_data.keys()),
                  y=[float(v) for v in returns_data.values()],
                  text=[f"{float(v):.2f}%" for v in returns_data.values()],
                  textposition='auto')
        ])
        fig.update_layout(title='Monthly Returns')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk Scatter Plot
        risk_data = analysis['risk']
        risk_df = pd.DataFrame({
            'Fund': risk_data['volatility'].keys(),
            'Volatility': [float(v) for v in risk_data['volatility'].values()],
            'Beta': [float(v) for v in risk_data['beta'].values()],
            'Concentration': [float(v) for v in risk_data['concentration'].values()]
        })
        
        fig = px.scatter(risk_df, 
                        x='Beta', 
                        y='Volatility',
                        size='Concentration',
                        color='Fund',
                        title='Risk Profile')
        st.plotly_chart(fig, use_container_width=True)

    # Monthly Performance Table
    st.subheader("Monthly Performance")
    try:
        # Create custom date range with specific months (Sep 2024 to Jan 2025)
        months = [
            'January 2025',
            'December 2024',
            'November 2024',
            'October 2024',
            'September 2024'
        ]

        table_data = []
        for fund_name, analyzer in comparator.analyzers.items():
            if analyzer.nav_data is not None:
                # First row: NAV and Returns combined
                nav_returns_row = [f"{fund_name} (NAV & Returns)"]
                # Second row: Reasons for NAV changes
                nav_reasons_row = [f"{fund_name} (Decisions)"]
                # Third row: Additions
                additions_row = [f"{fund_name} (Added)"]
                # Fourth row: Exits
                exits_row = [f"{fund_name} (Exited)"]
                # Fifth row: Sectoral Changes
                sector_row = [f"{fund_name} (Sector Changes)"]
                
                for month in months:
                    month_data = analyzer.nav_data[analyzer.nav_data['month'] == month]
                    if not month_data.empty:
                        # Calculate monthly return
                        start_nav = month_data.iloc[-1]['nav']
                        end_nav = month_data.iloc[0]['nav']
                        monthly_return = ((end_nav - start_nav) / start_nav) * 100
                        
                        # Combined NAV and Returns row
                        if monthly_return > 0:
                            returns_text = f"<span class='nav-increase'>📈 +{monthly_return:.2f}%</span>"
                        elif monthly_return < 0:
                            returns_text = f"<span class='nav-decrease'>📉 {monthly_return:.2f}%</span>"
                        else:
                            returns_text = f"<span class='nav-neutral'>⚡ 0.00%</span>"
                        
                        nav_returns_text = f"<span class='nav-value'>₹{end_nav:.2f}</span><span class='nav-return'>{returns_text}</span>"
                        nav_returns_row.append(nav_returns_text)
                        
                        # Get portfolio changes
                        portfolio_changes = analyzer._analyze_portfolio_changes(month)
                        
                        # Additions row
                        if portfolio_changes and portfolio_changes['new_additions']:
                            additions = []
                            for add in portfolio_changes['new_additions']:
                                add_text = [
                                    f"📈 New this month: {add['name']}",
                                    f"Weight: {add['weight']}%",
                                    "Reasons:"
                                ]
                                for reason in add['rationale']:
                                    add_text.append(f"• {reason}")
                                additions.append("<br>".join(add_text))
                            additions_text = "<br>".join(additions)
                        else:
                            additions_text = "No new additions"
                        
                        # Exits row
                        if portfolio_changes and portfolio_changes['exits']:
                            exits = []
                            for exit in portfolio_changes['exits']:
                                exit_text = [
                                    f"📉 Exited this month: {exit['name']}",
                                    f"Previous Weight: {exit['prev_weight']}%",
                                    "Reasons:"
                                ]
                                for reason in exit['rationale']:
                                    exit_text.append(f"• {reason}")
                                exits.append("<br>".join(exit_text))
                            exits_text = "<br>".join(exits)
                        else:
                            exits_text = "No exits"
                        
                        # Sectoral Changes row
                        if portfolio_changes:
                            sector_changes = []
                            sector_perf = analyzer._fetch_sector_performance(month_data.index.min())
                            
                            # Increased sectors
                            increased_sectors = [
                                (sector, perf) for sector, perf in sector_perf.items() 
                                if perf > 1.0
                            ]
                            if increased_sectors:
                                sector_changes.append("📈 Changes from last month:")
                                for sector, perf in increased_sectors:
                                    sector_changes.append(f"• {sector}: +{perf:.1f}%")
                                    reasons = get_sector_change_reasons(sector, 'increase')
                                    for reason in reasons:
                                        sector_changes.append(f"  → {reason}")
                            
                            # Decreased sectors
                            decreased_sectors = [
                                (sector, perf) for sector, perf in sector_perf.items() 
                                if perf < -1.0
                            ]
                            if decreased_sectors:
                                if not increased_sectors:
                                    sector_changes.append("📉 Changes from last month:")
                                for sector, perf in decreased_sectors:
                                    sector_changes.append(f"• {sector}: {perf:.1f}%")
                                    reasons = get_sector_change_reasons(sector, 'decrease')
                                    for reason in reasons:
                                        sector_changes.append(f"  → {reason}")
                            
                            if not increased_sectors and not decreased_sectors:
                                sector_changes.append("No significant sector changes")
                            
                            sector_text = "<br>".join(sector_changes)
                        else:
                            sector_text = "No sector data available"
                        
                        # Add NAV reasons
                        nav_reasons = get_nav_change_reasons(monthly_return, sector_perf, portfolio_changes)
                        nav_reasons_row.append(nav_reasons)
                        
                        # Add data to rows
                        additions_row.append(additions_text)
                        exits_row.append(exits_text)
                        sector_row.append(sector_text)
                    else:
                        nav_returns_row.append("N/A")
                        nav_reasons_row.append("-")
                        additions_row.append("-")
                        exits_row.append("-")
                        sector_row.append("-")
                
                # Add all rows for this fund to table_data
                table_data.extend([nav_returns_row, nav_reasons_row, additions_row, exits_row, sector_row])

        # Create DataFrame outside the fund loop
        df = pd.DataFrame(
            table_data,
            columns=['Fund Details'] + months
        )

        # Update table display settings
        st.markdown(
            df.to_html(
                escape=False,
                index=False,
                classes='styled-table',
                table_id='performance-table'
            ),
            unsafe_allow_html=True
        )

        # Add legend
        st.caption("""
        Table shows:
        • Row 1: NAV values and Monthly Returns
        • Row 2: Reasons for NAV changes
        • Row 3: New stocks added this month (compared to previous month)
        • Row 4: Stocks exited this month (compared to previous month)
        • Row 5: Sector allocation changes from previous month
        """)

    except Exception as e:
        st.error(f"Error generating performance table: {str(e)}")
        st.warning("Unable to display monthly performance data.")

if __name__ == "__main__":
    main() 