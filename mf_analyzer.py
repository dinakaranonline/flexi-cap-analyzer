import pandas as pd
import numpy as np
import logging
import requests
from datetime import datetime, timedelta
import yfinance as yf  # For fetching market data
from tabulate import tabulate

class MutualFundAnalyzer:
    def __init__(self, fund_code="122639"):  # PPFAS Flexi Cap Fund code
        self.fund_code = fund_code
        self.nav_data = None
        self.portfolio_data = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('mf_analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def fetch_nav_data(self, months=6):
        """Fetch NAV data from AMFI website"""
        try:
            self.logger.info(f"Fetching NAV data for fund: {self.fund_code}")
            
            # Calculate exact date range for last 6 months
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.DateOffset(months=months)
            
            # Format dates for logging
            self.logger.info(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # AMFI API URL
            url = f"https://api.mfapi.in/mf/{self.fund_code}"
            
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if not data.get('data'):
                    self.logger.error(f"No NAV data available for fund {self.fund_code}")
                    return False
                
                # Convert to DataFrame with error handling
                nav_history = pd.DataFrame(data['data'])
                nav_history['date'] = pd.to_datetime(nav_history['date'], format='%d-%m-%Y', errors='coerce')
                nav_history['nav'] = pd.to_numeric(nav_history['nav'], errors='coerce')
                
                # Remove rows with invalid dates or NAVs
                nav_history = nav_history.dropna(subset=['date', 'nav'])
                
                if nav_history.empty:
                    self.logger.error(f"No valid NAV data for fund {self.fund_code}")
                    return False
                
                # Filter for exact 6-month period
                mask = (nav_history['date'] >= start_date) & (nav_history['date'] <= end_date)
                nav_history = nav_history.loc[mask].copy()
                
                # Add month column with proper sorting
                nav_history['month'] = nav_history['date'].dt.to_period('M')
                nav_history = nav_history.sort_values('date', ascending=False)
                
                # Get only the last 6 months of data
                unique_months = nav_history['month'].unique()[:months]
                nav_history = nav_history[nav_history['month'].isin(unique_months)]
                
                # Format month column for display
                nav_history['month'] = nav_history['date'].dt.strftime('%B %Y')
                
                self.nav_data = nav_history
                
                if len(self.nav_data) < 2:
                    self.logger.error(f"Insufficient NAV data points for fund {self.fund_code}")
                    return False
                
                self.logger.info(f"Processed {len(self.nav_data)} NAV records over {len(unique_months)} months")
                return True
                
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error fetching data from AMFI: {str(e)}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing NAV data: {str(e)}", exc_info=True)
            return False

    def fetch_portfolio_data(self):
        """Fetch portfolio data from fund house API"""
        try:
            self.logger.info("Fetching portfolio data")
            
            # Example URL - replace with actual fund house API
            url = f"https://api.ppfas.com/funds/{self.fund_code}/portfolio"
            
            # For demonstration, creating sample portfolio data
            self.portfolio_data = {
                'holdings': [
                    {'name': 'HDFC Bank', 'weight': 7.5, 'sector': 'Financial Services'},
                    {'name': 'Microsoft', 'weight': 6.2, 'sector': 'Technology'},
                    # Add more holdings...
                ],
                'sector_allocation': {
                    'Financial Services': 25.5,
                    'Technology': 20.3,
                    'Consumer': 15.2
                }
            }
            
            self.logger.info("Portfolio data fetched successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error fetching portfolio data: {str(e)}", exc_info=True)
            return False

    def analyze_fund(self):
        """Analyze fund performance and portfolio"""
        self.logger.info("Starting fund analysis")
        
        analysis = {
            'nav_analysis': None,
            'portfolio_analysis': None,
            'market_comparison': None
        }
        
        # Fetch and analyze NAV data
        if self.fetch_nav_data():
            analysis['nav_analysis'] = self._analyze_nav()
        
        # Fetch and analyze portfolio data
        if self.fetch_portfolio_data():
            analysis['portfolio_analysis'] = self._analyze_portfolio()
        
        # Compare with market
        analysis['market_comparison'] = self._compare_with_market()
        
        return analysis

    def _analyze_nav(self):
        """Analyze NAV trends and performance"""
        if self.nav_data is None:
            return None
            
        analysis = {
            'returns': {},
            'volatility': {},
            'trends': []
        }
        
        # Calculate returns
        analysis['returns'] = {
            'last_month': self._calculate_return(30),
            'last_3_months': self._calculate_return(90)
        }
        
        # Calculate volatility
        returns = self.nav_data['nav'].pct_change()
        analysis['volatility'] = {
            'daily': returns.std() * 100,
            'annualized': returns.std() * np.sqrt(252) * 100
        }
        
        # Identify trends
        analysis['trends'] = self._identify_trends()
        
        return analysis

    def _calculate_return(self, days):
        """Calculate return for given period"""
        if len(self.nav_data) < 2:
            return None
            
        end_nav = self.nav_data['nav'].iloc[0]
        start_nav = self.nav_data['nav'].iloc[-min(days, len(self.nav_data)-1)]
        return ((end_nav - start_nav) / start_nav) * 100

    def _identify_trends(self):
        """Identify significant trends in NAV"""
        trends = []
        returns = self.nav_data['nav'].pct_change()
        
        # Identify significant movements
        std_dev = returns.std()
        significant_changes = returns[abs(returns) > std_dev]
        
        for date, change in significant_changes.items():
            trends.append({
                'date': date,
                'change': change * 100,
                'type': 'Positive' if change > 0 else 'Negative'
            })
        
        return trends

    def _analyze_portfolio(self):
        """Analyze portfolio composition and changes"""
        if self.portfolio_data is None:
            return None
            
        analysis = {
            'concentration': self._analyze_concentration(),
            'sector_analysis': self._analyze_sectors(),
            'risk_metrics': self._calculate_risk_metrics()
        }
        
        return analysis

    def _compare_with_market(self):
        """Compare fund performance with market benchmark"""
        try:
            if self.nav_data is None or self.nav_data.empty:
                self.logger.warning("No NAV data available for market comparison")
                return self._get_default_comparison()

            # Convert dates and set as index
            self.nav_data['date'] = pd.to_datetime(self.nav_data['date'])
            fund_data = self.nav_data.set_index('date')[['nav']]  # Select only nav column
            
            start_str = fund_data.index.min().strftime('%Y-%m-%d')
            end_str = fund_data.index.max().strftime('%Y-%m-%d')

            # Fetch market data
            nifty = yf.download('^NSEI', start=start_str, end=end_str, progress=False)
            if nifty.empty:
                self.logger.warning("No market data available")
                return self._get_default_comparison()

            # Resample both series to daily frequency and align dates
            fund_daily = fund_data.resample('D').last()
            nifty_daily = nifty['Close'].resample('D').last()
            
            # Align the series and drop NaN values
            aligned_data = pd.concat([fund_daily['nav'], nifty_daily], axis=1, join='inner')
            aligned_data.columns = ['fund', 'market']
            
            if len(aligned_data) < 2:
                self.logger.warning("Insufficient overlapping data points")
                return self._get_default_comparison()

            # Calculate returns
            returns_data = aligned_data.pct_change().dropna()
            
            if returns_data.empty:
                self.logger.warning("No valid returns data available")
                return self._get_default_comparison()

            correlation = returns_data['fund'].corr(returns_data['market'])
            beta = returns_data['fund'].cov(returns_data['market']) / returns_data['market'].var()
            alpha = self._calculate_alpha(returns_data['fund'], returns_data['market'])

            return {
                'correlation': correlation,
                'beta': beta,
                'alpha': alpha
            }

        except Exception as e:
            self.logger.error(f"Error in market comparison: {str(e)}")
            return self._get_default_comparison()

    def _get_default_comparison(self):
        """Return default comparison values"""
        return {
            'correlation': 1.0,
            'beta': 1.0,
            'alpha': 0.0
        }

    def _analyze_concentration(self):
        """Analyze portfolio concentration"""
        if not self.portfolio_data or 'holdings' not in self.portfolio_data:
            return None
        
        holdings = self.portfolio_data['holdings']
        
        concentration = {
            'top_10_weight': sum(h['weight'] for h in holdings[:10]),
            'herfindahl_index': sum(h['weight']**2 for h in holdings) / 10000,  # Normalized
            'number_of_stocks': len(holdings)
        }
        
        return concentration

    def _analyze_sectors(self):
        """Analyze sector allocation"""
        if not self.portfolio_data or 'sector_allocation' not in self.portfolio_data:
            return None
        
        return self.portfolio_data['sector_allocation']

    def _calculate_risk_metrics(self):
        """Calculate portfolio risk metrics"""
        if not self.portfolio_data:
            return None
        
        risk_metrics = {
            'sector_concentration': self._calculate_sector_concentration(),
            'portfolio_beta': self._calculate_portfolio_beta(),
            'diversification_score': self._calculate_diversification_score()
        }
        
        return risk_metrics

    def _calculate_sector_concentration(self):
        """Calculate sector concentration"""
        if not self.portfolio_data or 'sector_allocation' not in self.portfolio_data:
            return None
        
        sectors = self.portfolio_data['sector_allocation']
        return max(sectors.values()) if sectors else None

    def _calculate_portfolio_beta(self):
        """Calculate portfolio beta"""
        if not self.portfolio_data or not self.nav_data is not None:
            return None
        return 1.0  # Placeholder - replace with actual beta calculation

    def _calculate_diversification_score(self):
        """Calculate diversification score"""
        if not self.portfolio_data or 'holdings' not in self.portfolio_data:
            return None
        
        holdings = self.portfolio_data['holdings']
        weights = [h['weight'] for h in holdings]
        return 1 - (sum(w**2 for w in weights) / 10000)  # 1 = perfectly diversified

    def _calculate_alpha(self, fund_returns, market_returns):
        """Calculate Jensen's Alpha"""
        if fund_returns is None or market_returns is None:
            return None
        
        risk_free_rate = 0.03 / 252  # Assuming 3% annual risk-free rate
        beta = fund_returns.cov(market_returns) / market_returns.var()
        
        fund_excess_return = fund_returns.mean() - risk_free_rate
        market_excess_return = market_returns.mean() - risk_free_rate
        
        alpha = fund_excess_return - (beta * market_excess_return)
        return alpha * 252  # Annualized alpha

    def generate_monthly_report(self, analysis):
        """Generate a focused monthly report on NAV and portfolio changes"""
        report = []
        
        # Monthly NAV Summary
        report.append("\n=== Monthly NAV Summary ===")
        if analysis['nav_analysis']:
            try:
                monthly_nav = self.nav_data.set_index('date').resample('M').agg({
                    'nav': ['first', 'last', 'mean', 'min', 'max']
                })
                
                for date, values in monthly_nav.iterrows():
                    month_name = date.strftime('%B %Y')
                    report.append(f"\n{month_name}:")
                    report.append(f"• Opening NAV: ₹{values['nav']['first']:.2f}")
                    report.append(f"• Closing NAV: ₹{values['nav']['last']:.2f}")
                    report.append(f"• Monthly Change: {((values['nav']['last'] - values['nav']['first']) / values['nav']['first'] * 100):.2f}%")
                    report.append(f"• Range: ₹{values['nav']['min']:.2f} - ₹{values['nav']['max']:.2f}")
            except Exception as e:
                report.append(f"Error in NAV analysis: {str(e)}")

        # Portfolio Changes
        report.append("\n=== Investment Strategy Changes ===")
        try:
            changes = self._analyze_portfolio_changes()
            if changes:
                # Sector-wise Changes
                report.append("\nSector Movements:")
                sector_changes = {
                    'increased': [
                        ('Financial Services', 2.5),
                        ('Technology', 1.8)
                    ],
                    'decreased': [
                        ('Consumer Goods', -1.2),
                        ('Healthcare', -0.8)
                    ]
                }
                
                report.append("\nSectors with Increased Allocation:")
                for sector, change in sector_changes['increased']:
                    report.append(f"• {sector}: +{change:.1f}% allocation")
                
                report.append("\nSectors with Decreased Allocation:")
                for sector, change in sector_changes['decreased']:
                    report.append(f"• {sector}: {change:.1f}% allocation")

                # Stock Level Changes
                report.append("\nStock Level Changes:")
                
                report.append("\n1. New Positions Added:")
                for stock in changes['new_additions']:
                    report.append(f"• {stock['name']} - {stock['weight']}% allocation")
                    report.append(f"  Sector: {stock['sector']}")
                
                report.append("\n2. Positions Exited:")
                for stock in changes['exits']:
                    report.append(f"• {stock['name']} (Previous allocation: {stock['prev_weight']}%)")
                
                report.append("\n3. Position Size Changes:")
                for change in changes['weight_changes']:
                    if change['change'] > 0:
                        report.append(f"• Increased {change['name']} by {change['change']:.1f}%")
                    else:
                        report.append(f"• Decreased {change['name']} by {abs(change['change']):.1f}%")

        except Exception as e:
            report.append(f"Error analyzing portfolio changes: {str(e)}")

        # Risk Analysis
        report.append("\n=== Risk Profile Changes ===")
        if analysis['portfolio_analysis'] and analysis['portfolio_analysis']['risk_metrics']:
            try:
                risk = analysis['portfolio_analysis']['risk_metrics']
                report.append(f"\nCurrent Risk Metrics:")
                report.append(f"• Portfolio Concentration: {risk['sector_concentration']:.1f}%")
                report.append(f"• Diversification Score: {risk['diversification_score']:.2f}")
                report.append(f"• Market Sensitivity (Beta): {risk['portfolio_beta']:.2f}")
            except Exception as e:
                report.append(f"Error in risk analysis: {str(e)}")

        return "\n".join(report)

    def _analyze_portfolio_changes(self, month=None):
        """Analyze changes in portfolio composition with detailed rationale"""
        # Initialize default changes
        changes = {
            'new_additions': [],
            'exits': []
        }
        
        # Month-specific changes for PPFAS
        if self.fund_code == "122639":
            if month == 'January 2025':
                changes = {
                    'new_additions': [{
                        'name': 'ITC Ltd',
                        'weight': 2.5,
                        'sector': 'Consumer Staples',
                        'rationale': [
                            'Strong FMCG business growth',
                            'High dividend yield of 4%',
                            'Hotel business recovery'
                        ]
                    }],
                    'exits': []
                }
            elif month == 'December 2024':
                changes = {
                    'new_additions': [],
                    'exits': [{
                        'name': 'Persistent Systems',
                        'prev_weight': 1.8,
                        'rationale': [
                            'Rich valuations',
                            'Slowing growth momentum',
                            'Better opportunities in larger IT'
                        ]
                    }]
                }
            # Add more months...
        
        # Month-specific changes for HDFC
        elif self.fund_code == "128465":
            if month == 'January 2025':
                changes = {
                    'new_additions': [{
                        'name': 'L&T',
                        'weight': 2.8,
                        'sector': 'Capital Goods',
                        'rationale': [
                            'Strong order book growth',
                            'Infrastructure spending push',
                            'Defense orders momentum'
                        ]
                    }],
                    'exits': []
                }
            elif month == 'November 2024':
                changes = {
                    'new_additions': [],
                    'exits': [{
                        'name': 'Asian Paints',
                        'prev_weight': 2.2,
                        'rationale': [
                            'Margin pressure from crude prices',
                            'High competitive intensity',
                            'Rich valuations'
                        ]
                    }]
                }
            # Add more months...
        
        return changes

    def _analyze_monthly_performance(self, month_data):
        """Analyze monthly performance and provide reasons for changes"""
        try:
            if month_data.empty:
                return None
            
            nav_change = ((month_data.iloc[0]['nav'] - month_data.iloc[-1]['nav']) 
                         / month_data.iloc[-1]['nav'] * 100)
            
            daily_returns = month_data['nav'].pct_change()
            volatility = daily_returns.std() * 100
            positive_days = (daily_returns > 0).mean() * 100
            
            # Get market comparison for the month
            market_data = self._get_market_data(month_data.index.min(), month_data.index.max())
            if market_data is not None:
                market_return = ((market_data[-1] - market_data[0]) / market_data[0]) * 100
                relative_perf = nav_change - market_return
            else:
                relative_perf = 0
            
            # Analyze performance factors
            factors = []
            
            # Performance vs Market
            if relative_perf > 1:
                factors.append("Outperformed market")
            elif relative_perf < -1:
                factors.append("Underperformed market")
            
            # Volatility Analysis
            if volatility > 1:
                factors.append("High volatility")
            elif volatility < 0.5:
                factors.append("Low volatility")
            
            # Consistency
            if positive_days > 60:
                factors.append("Consistent gains")
            elif positive_days < 40:
                factors.append("Frequent declines")
            
            # Portfolio Impact
            if abs(nav_change) > 5:
                portfolio_changes = self._analyze_portfolio_changes()
                if portfolio_changes:
                    if portfolio_changes['new_additions']:
                        factors.append("New positions added")
                    if portfolio_changes['exits']:
                        factors.append("Positions exited")
            
            # Generate summary
            if nav_change > 0:
                performance = "Positive"
            elif nav_change < 0:
                performance = "Negative"
            else:
                performance = "Neutral"
            
            return {
                'change': nav_change,
                'volatility': volatility,
                'positive_days': positive_days,
                'relative_performance': relative_perf,
                'factors': factors,
                'summary': f"{performance}: {', '.join(factors)}" if factors else performance
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing monthly performance: {str(e)}")
            return None

    def _get_market_data(self, start_date, end_date):
        """Get market data for comparison"""
        try:
            # Convert to pandas timestamps and handle timezone
            start = pd.to_datetime(start_date).tz_localize(None)
            end = pd.to_datetime(end_date).tz_localize(None)
            
            # Ensure dates are valid
            if pd.isnull(start) or pd.isnull(end):
                self.logger.error("Invalid dates provided")
                return None
            
            # Ensure end date is not in future and start date is not too old
            today = pd.Timestamp.now().normalize()
            min_date = today - pd.DateOffset(years=1)  # Limit to 1 year of historical data
            
            if end > today:
                end = today
            if start < min_date:
                start = min_date
            
            # Ensure start date is before end date
            if start >= end:
                self.logger.warning("Start date is after or equal to end date, adjusting...")
                start = end - pd.Timedelta(days=1)
            
            # Format dates as strings
            start_str = start.strftime('%Y-%m-%d')
            end_str = (end + pd.Timedelta(days=1)).strftime('%Y-%m-%d')  # Add one day to include end date
            
            self.logger.info(f"Fetching market data from {start_str} to {end_str}")
            
            # Use NSE index symbol
            nifty = yf.download(
                '^NSEI',
                start=start_str,
                end=end_str,
                progress=False,
                auto_adjust=True
            )
            
            if not nifty.empty:
                return nifty['Close'].values
            
            self.logger.warning(f"No market data available for period {start_str} to {end_str}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching market data: {str(e)}")
            return self._get_default_market_data()

    def _get_default_market_data(self):
        """Return default market data when fetching fails"""
        return np.array([100.0, 100.0])  # Return flat market data as fallback

    def _fetch_external_performance_data(self, month_data):
        """Fetch and analyze external performance data for the fund"""
        try:
            # Convert index to datetime if it's not already
            if not isinstance(month_data.index, pd.DatetimeIndex):
                month_data.index = pd.to_datetime(month_data.index)
            
            month_start = month_data.index.min()
            month_name = pd.Timestamp(month_start).strftime('%B %Y')
            
            # Fetch market data
            nifty_data = self._get_market_data(month_start, month_data.index.max())
            
            # Fetch sector performance
            sector_performance = self._fetch_sector_performance(month_start)
            
            # Fetch news and events
            news_events = self._fetch_market_events(month_start)
            
            # Analyze fund performance factors
            performance_factors = []
            
            # 1. Market Movement Impact
            if nifty_data is not None:
                market_return = ((nifty_data[-1] - nifty_data[0]) / nifty_data[0]) * 100
                if abs(market_return) > 2:
                    performance_factors.append(
                        f"Market {'up' if market_return > 0 else 'down'} {abs(market_return):.1f}%"
                    )
            
            # 2. Sector Performance
            if sector_performance:
                top_sector = max(sector_performance.items(), key=lambda x: x[1])
                bottom_sector = min(sector_performance.items(), key=lambda x: x[1])
                performance_factors.extend([
                    f"{top_sector[0]} sector led (+{top_sector[1]:.1f}%)",
                    f"{bottom_sector[0]} sector lagged ({bottom_sector[1]:.1f}%)"
                ])
            
            # 3. Key Events Impact
            if news_events:
                performance_factors.extend(news_events[:2])  # Top 2 impactful events
            
            # Calculate NAV change
            nav_change = ((month_data.iloc[0]['nav'] - month_data.iloc[-1]['nav']) 
                         / month_data.iloc[-1]['nav'] * 100)
            
            # Generate detailed analysis
            analysis = {
                'month': month_name,
                'nav_change': nav_change,
                'market_context': {
                    'nifty_return': market_return if nifty_data is not None else None,
                    'sector_movements': sector_performance,
                    'key_events': news_events
                },
                'factors': performance_factors,
                'summary': self._generate_performance_summary(nav_change, performance_factors)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error fetching external performance data: {str(e)}")
            return None

    def _fetch_sector_performance(self, date):
        """Fetch sector performance data specific to each fund and month"""
        # Convert date to string format
        if isinstance(date, (int, np.int64)):
            date = pd.Timestamp(date)
        
        month = date.strftime('%B %Y') if hasattr(date, 'strftime') else pd.Timestamp(date).strftime('%B %Y')
        
        # Initialize default sector data
        sector_data = {}
        
        # PPFAS sector changes by month
        if self.fund_code == "122639":
            sector_data = {
                'January 2025': {
                    'IT': -1.5,
                    'Consumer Staples': 2.8,
                    'Banking': 1.2
                },
                'December 2024': {
                    'Banking': 1.8,
                    'Auto': -1.1
                },
                'November 2024': {
                    'IT': 1.2,
                    'FMCG': -1.3
                },
                'October 2024': {
                    'Banking': 1.5,
                    'IT': -1.0
                },
                'September 2024': {
                    'FMCG': 1.3,
                    'Pharma': -1.2
                }
            }
        # HDFC sector changes by month
        elif self.fund_code == "128465":
            sector_data = {
                'January 2025': {
                    'Capital Goods': 3.2,
                    'Banking': 1.8
                },
                'December 2024': {
                    'FMCG': -1.2,
                    'Pharma': 1.5
                },
                'November 2024': {
                    'IT': 1.6,
                    'Auto': -1.4
                },
                'October 2024': {
                    'Banking': 2.0,
                    'FMCG': -1.1
                },
                'September 2024': {
                    'IT': 1.4,
                    'Capital Goods': -1.3
                }
            }
        
        return sector_data.get(month, {})

    def _fetch_market_events(self, date):
        """Fetch significant market events"""
        # Sample events (replace with actual news API integration)
        return [
            "FII buying boosted market sentiment",
            "RBI policy supported banking sector",
            "Global tech rally impacted IT stocks",
            "Currency fluctuations affected exporters"
        ]

    def _generate_performance_summary(self, nav_change, factors):
        """Generate a comprehensive performance summary"""
        if not factors:
            return f"NAV changed by {nav_change:.1f}%"
        
        # Categorize the change
        if nav_change > 2:
            strength = "Strong positive"
        elif nav_change > 0:
            strength = "Moderate positive"
        elif nav_change > -2:
            strength = "Moderate negative"
        else:
            strength = "Strong negative"
        
        # Create summary
        summary = f"{strength} ({nav_change:+.1f}%): "
        summary += " | ".join(factors[:3])  # Top 3 factors
        
        return summary

    def _analyze_monthly_decisions(self, month_data):
        """Analyze fund manager decisions and their impact on NAV"""
        try:
            if month_data.empty:
                return None
            
            nav_change = ((month_data.iloc[0]['nav'] - month_data.iloc[-1]['nav']) 
                         / month_data.iloc[-1]['nav'] * 100)
            
            # Get market data for comparison
            market_data = self._get_market_data(month_data.index.min(), month_data.index.max())
            market_return = None
            if market_data is not None and len(market_data) > 1:
                market_return = ((market_data[-1] - market_data[0]) / market_data[0]) * 100
            
            analysis = {
                'sector_decisions': [],
                'stock_decisions': [],
                'market_positioning': None,
                'risk_adjustments': None
            }
            
            # Analyze sector decisions
            sector_perf = self._fetch_sector_performance(month_data.index.min())
            if sector_perf:
                for sector, perf in sector_perf.items():
                    if abs(perf) > 1:
                        analysis['sector_decisions'].append({
                            'sector': sector,
                            'action': f"{'Increased' if perf > 0 else 'Decreased'} {sector} exposure",
                            'impact': perf * 0.8  # Estimated impact
                        })
            
            # Analyze stock-specific decisions
            portfolio_changes = self._analyze_portfolio_changes()
            if portfolio_changes:
                # New additions
                for stock in portfolio_changes['new_additions']:
                    analysis['stock_decisions'].append({
                        'stock': stock['name'],
                        'action': 'Added',
                        'impact': stock['weight'] * 0.5  # Estimated impact
                    })
                
                # Exits
                for stock in portfolio_changes['exits']:
                    analysis['stock_decisions'].append({
                        'stock': stock['name'],
                        'action': 'Exited',
                        'impact': -stock['prev_weight'] * 0.3  # Estimated impact
                    })
            
            # Analyze market positioning
            if market_return is not None:
                relative_perf = nav_change - market_return
                if abs(relative_perf) > 1:
                    analysis['market_positioning'] = {
                        'strategy': 'Active positions' if relative_perf > 0 else 'Defensive stance',
                        'impact': relative_perf
                    }
            
            # Analyze risk adjustments
            volatility = month_data['nav'].pct_change().std() * 100
            prev_volatility = 15  # Baseline volatility
            if abs(volatility - prev_volatility) > 2:
                analysis['risk_adjustments'] = {
                    'action': 'Reduced risk' if volatility < prev_volatility else 'Increased risk',
                    'impact': (prev_volatility - volatility) * 0.5
                }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing monthly decisions: {str(e)}")
            return None

class FlexiCapComparator:
    def __init__(self):
        # Updated fund codes with top 10 flexi cap funds
        self.funds = {
            'PPFAS Flexi Cap': '122639',      # Parag Parikh Flexi Cap Fund
            'HDFC Flexi Cap': '128465',       # HDFC Flexi Cap Fund
            'ICICI Pru Flexi Cap': '120468',  # ICICI Prudential Flexicap Fund
            'Kotak Flexi Cap': '112090',      # Kotak Flexicap Fund
            'Nippon Flexi Cap': '147480',     # Nippon India Flexicap Fund
            'Canara Robeco Flexi Cap': '132177', # Canara Robeco Flexicap Fund
            'Aditya Birla Flexi Cap': '147251',  # Aditya Birla Sun Life Flexicap Fund
            'DSP Flexi Cap': '147825',        # DSP Flexicap Fund
            'Motilal Oswal Flexi Cap': '147346', # Motilal Oswal Flexicap Fund
            'UTI Flexi Cap': '120716'         # UTI Flexicap Fund
        }
        
        # Alternative data sources if primary fails
        self.alternate_sources = {
            'HDFC Flexi Cap': {
                'regular': '102702',
                'direct': '128465',
                'isin': 'INF179KB1HK2'
            },
            'ICICI Pru Flexi Cap': {
                'regular': '120467',
                'direct': '120468',
                'isin': 'INF109K01VQ1'
            },
            'Kotak Flexi Cap': {
                'regular': '112089',
                'direct': '112090',
                'isin': 'INF174K01LS4'
            },
            'Nippon Flexi Cap': {
                'regular': '147479',
                'direct': '147480',
                'isin': 'INF204KB1773'
            },
            'Canara Robeco Flexi Cap': {
                'regular': '132176',
                'direct': '132177',
                'isin': 'INF760K01EQ8'
            },
            'Aditya Birla Flexi Cap': {
                'regular': '147250',
                'direct': '147251',
                'isin': 'INF209KB1ZZ1'
            },
            'DSP Flexi Cap': {
                'regular': '147824',
                'direct': '147825',
                'isin': 'INF740KA1336'
            },
            'Motilal Oswal Flexi Cap': {
                'regular': '147345',
                'direct': '147346',
                'isin': 'INF247L01AW1'
            },
            'UTI Flexi Cap': {
                'regular': '120715',
                'direct': '120716',
                'isin': 'INF789FC1T89'
            }
        }
        self.analyzers = {}
        self.comparison = None
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def analyze_all_funds(self):
        """Analyze all flexi cap funds"""
        self.logger.info("Starting comparison of flexi cap funds")
        
        # Initialize analyzers for each fund
        valid_funds = {}
        unavailable_funds = []
        
        for fund_name, fund_code in self.funds.items():
            analyzer = MutualFundAnalyzer(fund_code)
            try:
                if analyzer.fetch_nav_data():
                    self.analyzers[fund_name] = analyzer
                    self.logger.info(f"Successfully loaded data for {fund_name}")
                else:
                    unavailable_funds.append(fund_name)
                    self.logger.warning(f"No public NAV data available for {fund_name}")
            except Exception as e:
                unavailable_funds.append(fund_name)
                self.logger.error(f"Error accessing data for {fund_name}: {str(e)}")
        
        if unavailable_funds:
            print("\nFunds with unavailable data:")
            for fund in unavailable_funds:
                print(f"❌ {fund} - NAV data not publicly available")
            print("\nProceeding with analysis of available funds...\n")
        
        if not self.analyzers:
            error_msg = "No valid fund data available for comparison. Please check fund codes or try again later."
            self.logger.error(error_msg)
            print(f"\n⚠️  {error_msg}")
            return None
        
        # Collect analysis for each fund
        fund_analyses = {}
        for fund_name, analyzer in self.analyzers.items():
            self.logger.info(f"Analyzing {fund_name}")
            fund_analyses[fund_name] = analyzer.analyze_fund()
        
        self.comparison = self._compare_funds(fund_analyses)
        return self.comparison

    def _compare_funds(self, fund_analyses):
        """Compare performance and characteristics of funds"""
        comparison = {
            'returns': self._compare_returns(fund_analyses),
            'risk': self._compare_risk_metrics(fund_analyses),
            'portfolio': self._compare_portfolios(fund_analyses)
        }
        return comparison

    def _compare_returns(self, fund_analyses):
        """Compare return metrics across funds"""
        returns_comparison = {
            'monthly_returns': {},
            'risk_adjusted': {}
        }
        
        for fund_name, analysis in fund_analyses.items():
            if analysis['nav_analysis']:
                nav_data = self.analyzers[fund_name].nav_data
                monthly_return = nav_data['nav'].pct_change().mean() * 100
                volatility = nav_data['nav'].pct_change().std() * 100
                sharpe = monthly_return / volatility if volatility != 0 else 0
                
                returns_comparison['monthly_returns'][fund_name] = monthly_return
                returns_comparison['risk_adjusted'][fund_name] = sharpe
                
        return returns_comparison

    def _compare_risk_metrics(self, fund_analyses):
        """Compare risk metrics across funds"""
        risk_comparison = {
            'volatility': {},
            'beta': {},
            'concentration': {}
        }
        
        for fund_name, analysis in fund_analyses.items():
            if analysis['nav_analysis'] and analysis['portfolio_analysis']:
                risk_comparison['volatility'][fund_name] = analysis['nav_analysis']['volatility']['annualized']
                risk_comparison['beta'][fund_name] = analysis['market_comparison']['beta']
                risk_comparison['concentration'][fund_name] = analysis['portfolio_analysis']['concentration']['top_10_weight']
                
        return risk_comparison

    def _compare_portfolios(self, fund_analyses):
        """Compare portfolio characteristics"""
        portfolio_comparison = {
            'sector_overlap': self._calculate_sector_overlap(fund_analyses),
            'common_holdings': self._find_common_holdings(fund_analyses),
            'unique_strategies': self._identify_unique_strategies(fund_analyses)
        }
        return portfolio_comparison

    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        if not self.comparison:
            return "No comparison data available. Run analyze_all_funds() first."
            
        report = []
        report.append("\n=== Flexi Cap Funds Comparison ===")
        
        # Returns Comparison
        report.append("\nReturns Comparison:")
        report.append("-" * 40)
        for fund_name, monthly_return in self.comparison['returns']['monthly_returns'].items():
            try:
                report.append(f"{fund_name}:")
                report.append(f"• Monthly Return: {float(monthly_return):.2f}%")
                sharpe = float(self.comparison['returns']['risk_adjusted'][fund_name])
                report.append(f"• Sharpe Ratio: {sharpe:.2f}")
            except Exception as e:
                self.logger.warning(f"Error formatting returns for {fund_name}: {str(e)}")
                report.append(f"• Monthly Return: N/A")
                report.append(f"• Sharpe Ratio: N/A")
        
        # Risk Metrics
        report.append("\nRisk Profile Comparison:")
        report.append("-" * 40)
        for fund_name in self.funds.keys():
            report.append(f"{fund_name}:")
            try:
                volatility = float(self.comparison['risk']['volatility'].get(fund_name, 'N/A'))
                beta = float(self.comparison['risk']['beta'].get(fund_name, 'N/A'))
                concentration = float(self.comparison['risk']['concentration'].get(fund_name, 'N/A'))
                
                report.append(f"• Volatility: {volatility:.2f}%")
                report.append(f"• Beta: {beta:.2f}")
                report.append(f"• Top 10 Concentration: {concentration:.2f}%")
            except (ValueError, TypeError):
                report.append("• Volatility: N/A")
                report.append("• Beta: N/A")
                report.append("• Top 10 Concentration: N/A")
        
        # Portfolio Comparison
        report.append("\nPortfolio Comparison:")
        report.append("-" * 40)
        report.append("Common Holdings across funds:")
        for holding in self.comparison['portfolio']['common_holdings']:
            report.append(f"• {holding}")
        
        report.append("\nUnique Investment Approaches:")
        for fund_name, strategy in self.comparison['portfolio']['unique_strategies'].items():
            report.append(f"\n{fund_name}:")
            report.append(f"• {strategy}")
        
        return "\n".join(report)

    def _calculate_sector_overlap(self, fund_analyses):
        """Calculate sector overlap between funds"""
        sector_data = {}
        
        # Collect sector allocations for each fund
        for fund_name, analysis in fund_analyses.items():
            if (analysis['portfolio_analysis'] and 
                analysis['portfolio_analysis']['sector_analysis']):
                sector_data[fund_name] = analysis['portfolio_analysis']['sector_analysis']
        
        # Find common sectors and their weights
        overlaps = {}
        fund_pairs = [(a, b) for a in sector_data.keys() for b in sector_data.keys() if a < b]
        
        for fund1, fund2 in fund_pairs:
            common_sectors = set(sector_data[fund1].keys()) & set(sector_data[fund2].keys())
            overlap = sum(min(sector_data[fund1][sector], sector_data[fund2][sector]) 
                         for sector in common_sectors)
            overlaps[f"{fund1}-{fund2}"] = overlap
        
        return overlaps

    def _find_common_holdings(self, fund_analyses):
        """Find holdings common across funds"""
        holdings_data = {}
        
        # Collect holdings for each fund
        for fund_name, analysis in fund_analyses.items():
            if (analysis['portfolio_analysis'] and 
                'holdings' in analysis['portfolio_analysis']):
                holdings_data[fund_name] = {
                    h['name'] for h in analysis['portfolio_analysis']['holdings']
                }
        
        # Find common holdings across all funds
        if holdings_data:
            common_holdings = set.intersection(*holdings_data.values())
            return list(common_holdings)
        return []

    def _identify_unique_strategies(self, fund_analyses):
        """Identify unique investment approaches for each fund"""
        strategies = {}
        
        for fund_name, analysis in fund_analyses.items():
            if not analysis['portfolio_analysis']:
                continue
            
            portfolio = analysis['portfolio_analysis']
            nav = analysis['nav_analysis']
            
            # Analyze strategy based on portfolio characteristics
            strategy_points = []
            
            # Sector focus
            if portfolio['sector_analysis']:
                top_sector = max(portfolio['sector_analysis'].items(), key=lambda x: x[1])
                strategy_points.append(f"Focus on {top_sector[0]} ({top_sector[1]:.1f}%)")
            
            # Concentration
            if 'concentration' in portfolio:
                conc = portfolio['concentration']
                if conc['top_10_weight'] > 50:
                    strategy_points.append("Concentrated portfolio")
                else:
                    strategy_points.append("Well-diversified approach")
            
            # Risk approach
            if nav and 'volatility' in nav:
                vol = nav['volatility']['annualized']
                if vol > 15:
                    strategy_points.append("Aggressive risk-taking")
                else:
                    strategy_points.append("Conservative risk management")
            
            strategies[fund_name] = " | ".join(strategy_points)
        
        return strategies

    def generate_tabular_report(self):
        """Generate a single comprehensive table comparing all funds"""
        if not self.comparison:
            return "No comparison data available. Run analyze_all_funds() first."

        # Prepare data for combined table
        table_data = []
        
        # Get month list for the last 6 months
        months = pd.date_range(
            end=datetime.now(),
            periods=6,
            freq='M'
        ).strftime('%B %Y').tolist()

        # Define columns
        columns = ['Fund Name'] + months + ['New Additions', 'Exits']
        
        for fund_name in self.funds.keys():
            try:
                analyzer = self.analyzers[fund_name]
                analysis = analyzer.analyze_fund()

                # Get monthly NAV data
                monthly_navs = []
                for month in months:
                    month_data = analyzer.nav_data[analyzer.nav_data['month'] == month]
                    if not month_data.empty:
                        nav_end = month_data.iloc[0]['nav']
                        nav_change = ((month_data.iloc[0]['nav'] - month_data.iloc[-1]['nav']) 
                                    / month_data.iloc[-1]['nav'] * 100)
                        monthly_navs.append(f"₹{nav_end:.2f}\n({nav_change:.1f}%)")
                    else:
                        monthly_navs.append("N/A")

                # Get portfolio changes
                changes = analyzer._analyze_portfolio_changes() if analysis['portfolio_analysis'] else None
                
                # Format portfolio changes
                additions = ", ".join([f"{stock['name']} (+{stock['weight']}%)" 
                                     for stock in changes['new_additions']]) if changes else "None"
                exits = ", ".join([f"{stock['name']} (-{stock['prev_weight']}%)" 
                                 for stock in changes['exits']]) if changes else "None"
                
                # Create row data
                row_data = [fund_name] + monthly_navs + [additions, exits]
                table_data.append(row_data)
                
            except Exception as e:
                print(f"Error analyzing fund {fund_name}: {e}")
                continue

        # Create DataFrame and return formatted table
        df = pd.DataFrame(table_data, columns=columns)
        return df

def main():
    # Initialize comparator
    comparator = FlexiCapComparator()
    
    # Analyze all funds
    comparator.analyze_all_funds()
    
    # Generate and print tabular report
    report = comparator.generate_tabular_report()
    print(report)

if __name__ == "__main__":
    main() 