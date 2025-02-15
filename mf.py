import pandas as pd

def _compare_with_market(self):
    """Compare fund performance with market benchmark"""
    try:
        if self.nav_data is None or self.nav_data.empty:
            self.logger.warning("No NAV data available for market comparison")
            return self._get_default_comparison()

        # Ensure 'date' is in the DataFrame and is a datetime type
        if 'date' not in self.nav_data.columns:
            self.logger.error("Date column is missing from NAV data")
            return self._get_default_comparison()

        self.nav_data['date'] = pd.to_datetime(self.nav_data['date'], errors='coerce')

        start_date = self.nav_data['date'].min()
        end_date = self.nav_data['date'].max()

        # Check for NaT values
        if pd.isna(start_date) or pd.isna(end_date):
            self.logger.warning("Invalid dates for market comparison")
            return self._get_default_comparison()

        # Continue with the rest of your logic... 
    except Exception as e:
        self.logger.error(f"Error in market comparison: {str(e)}")
        return self._get_default_comparison() 