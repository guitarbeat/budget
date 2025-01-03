import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from fuzzywuzzy import fuzz
import logging
from datetime import datetime
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
REQUIRED_COLUMNS = ['Date', 'Amount']
OPTIONAL_COLUMNS = ['Description']
DATE_TOLERANCE_DAYS = 3
AMOUNT_TOLERANCE = 0.01
DESCRIPTION_MATCH_THRESHOLD = 85
DISCREPANCY_TYPES = {
    'left_only': '🏦 In Bank Only',
    'right_only': '📊 In Budget Only',
    'amount_mismatch': '💰 Amount Mismatch',
    'date_mismatch': '📅 Date Mismatch'
}

# Application Steps Configuration
STEPS = {
    1: {
        "name": "Upload Files",
        "icon": "📤",
        "description": "Upload bank and budget files",
        "tab_label": "📤 Upload Files"
    },
    2: {
        "name": "Map Columns",
        "icon": "🗺️",
        "description": "Match file columns",
        "tab_label": "🗺️ Column Mapping"
    },
    3: {
        "name": "Process",
        "icon": "🔄",
        "description": "Process and standardize data",
        "tab_label": "🔄 Process Data"
    },
    4: {
        "name": "Analyze",
        "icon": "📊",
        "description": "View matching results",
        "tab_label": "📊 Results"
    }
}

# Cache configuration
CACHE_TTL = 3600  # 1 hour in seconds

@st.cache_data(ttl=CACHE_TTL)
def load_and_process_csv(file_content, filename: str) -> pd.DataFrame:
    """Load and process CSV file with caching"""
    try:
        df = pd.read_csv(file_content)
        df['Source'] = filename
        return df
    except Exception as e:
        logger.error(f"Error loading CSV file {filename}: {str(e)}")
        raise ValueError(f"Failed to load {filename}. Please ensure it's a valid CSV file.")

@st.cache_data(ttl=CACHE_TTL)
def combine_transaction_dfs(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Combine multiple transaction DataFrames with caching"""
    try:
        return pd.concat(dfs, ignore_index=True, sort=False)
    except Exception as e:
        logger.error(f"Error combining DataFrames: {str(e)}")
        raise ValueError("Failed to combine transaction files. Please check file formats.")

# Column Mapping Class
class ColumnMapper:
    def __init__(self):
        self.required_cols = REQUIRED_COLUMNS
        self.optional_cols = OPTIONAL_COLUMNS
        
        # Known file format mappings
        self.format_mappings = {
            'apple_card': {
                'Date': 'Transaction Date',
                'Amount': 'Amount (USD)',
                'Description': ['Description', 'Merchant']  # Will concatenate these
            },
            'ynab': {
                'Date': 'Date',
                'Amount': ['Inflow', 'Outflow'],  # Special handling for YNAB
                'Description': ['Memo', 'Payee']  # Will concatenate these
            }
        }

    def detect_file_format(self, df: pd.DataFrame) -> str:
        """Detect the file format based on columns"""
        columns = set(df.columns)
        
        if 'Transaction Date' in columns and 'Amount (USD)' in columns:
            return 'apple_card'
        elif 'Inflow' in columns and 'Outflow' in columns:
            return 'ynab'
        return 'unknown'

    def display_available_columns(self, transaction_df: pd.DataFrame, budget_df: pd.DataFrame) -> None:
        st.subheader("Available Columns")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Bank Transaction Columns:")
            st.write(sorted(transaction_df.columns.tolist()))
            format_type = self.detect_file_format(transaction_df)
            st.info(f"Detected format: {format_type}")
        
        with col2:
            st.write("Budget Columns:")
            st.write(sorted(budget_df.columns.tolist()))
            format_type = self.detect_file_format(budget_df)
            st.info(f"Detected format: {format_type}")

    def create_mapping_interface(self, transaction_df: pd.DataFrame, budget_df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str], bool]:
        st.subheader("Column Mapping")
        st.info("Map your file columns to standard names. Multiple columns can be selected for combined fields.")
        
        # Detect file formats
        transaction_format = self.detect_file_format(transaction_df)
        budget_format = self.detect_file_format(budget_df)
        
        # Create mappings with detected formats
        col1, col2 = st.columns(2)
        
        with col1:
            transaction_mapping = self._create_single_mapping(
                transaction_df,
                "Bank Transaction",
                self.format_mappings.get(transaction_format, {})
            )
        
        with col2:
            budget_mapping = self._create_single_mapping(
                budget_df,
                "Budget",
                self.format_mappings.get(budget_format, {})
            )
        
        has_ynab_amount = budget_format == 'ynab'
        
        # Validate and clean mappings
        missing_cols = []
        missing_cols.extend(self._validate_mapping(transaction_mapping, "Bank Transactions"))
        
        if not has_ynab_amount:
            missing_cols.extend(self._validate_mapping(budget_mapping, "Budget"))
        
        if missing_cols:
            for msg in missing_cols:
                st.error(msg)
            return None, None, False
        
        # Clean mappings
        transaction_mapping = self._clean_mapping(transaction_mapping)
        budget_mapping = self._clean_mapping(budget_mapping)
        
        return transaction_mapping, budget_mapping, has_ynab_amount

    def _create_single_mapping(
        self,
        df: pd.DataFrame,
        mapping_name: str,
        default_mappings: Dict[str, Any]
    ) -> Dict[str, Any]:
        st.write(f"{mapping_name} Column Mapping:")
        mapping = {}
        cols = [""] + df.columns.tolist()
        
        # Create columns for each required and optional column
        columns = st.columns(len(self.required_cols + self.optional_cols))
        
        for idx, col in enumerate(self.required_cols + self.optional_cols):
            with columns[idx]:
                is_optional = col in self.optional_cols
                default_value = default_mappings.get(col, "")
                
                # Create a unique key for each widget
                widget_key = f"{mapping_name}_{col}"
                
                # Handle multiple column mappings
                if isinstance(default_value, list):
                    mapping[col] = st.multiselect(
                        f"{col} Column{' (Optional)' if is_optional else ''} (can select multiple)",
                        options=cols,
                        default=[c for c in default_value if c in cols],
                        help="Select one or more columns to combine",
                        key=widget_key
                    )
                else:
                    mapping[col] = st.selectbox(
                        f"{col} Column{' (Optional)' if is_optional else ''}",
                        options=cols,
                        index=next((i for i, c in enumerate(cols) if c == default_value), 0),
                        key=widget_key
                    )
        
        return mapping

    def _clean_mapping(self, mapping: Dict[str, Any]) -> Dict[str, Any]:
        """Clean mapping by removing empty values and handling lists"""
        cleaned = {}
        for k, v in mapping.items():
            if isinstance(v, list):
                # Keep only non-empty values from lists
                v = [x for x in v if x and x != ""]
                if v:  # Only add if we have values
                    cleaned[k] = v
            elif v and v != "":
                cleaned[k] = v
        return cleaned

    def _validate_mapping(self, mapping: Dict[str, Any], source_name: str) -> List[str]:
        missing_cols = []
        for col in self.required_cols:
            value = mapping.get(col)
            if not value or (isinstance(value, list) and not value):
                missing_cols.append(f"Missing required column '{col}' in {source_name} mapping")
        return missing_cols

# Data Processing Functions
def process_dataframe(df: pd.DataFrame, process_date: bool = True, process_amount: bool = True) -> pd.DataFrame:
    """Process a DataFrame by standardizing date and amount columns with error handling"""
    df = df.copy()
    try:
        if process_date and 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            # Handle invalid dates
            invalid_dates = df['Date'].isna()
            if invalid_dates.any():
                logger.warning(f"Found {invalid_dates.sum()} invalid dates")
                st.warning(f"⚠️ Found {invalid_dates.sum()} invalid dates. These rows will be excluded.")
                df = df[~invalid_dates]
        
        if process_amount and 'Amount' in df.columns:
            # Remove currency symbols and commas
            df['Amount'] = df['Amount'].replace(r'[$,]', '', regex=True)
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').round(2)
            # Handle invalid amounts
            invalid_amounts = df['Amount'].isna()
            if invalid_amounts.any():
                logger.warning(f"Found {invalid_amounts.sum()} invalid amounts")
                st.warning(f"⚠️ Found {invalid_amounts.sum()} invalid amounts. These rows will be excluded.")
                df = df[~invalid_amounts]
    except Exception as e:
        logger.error(f"Error processing DataFrame: {str(e)}")
        raise ValueError("Failed to process data. Please check column formats.")
    
    return df

def find_all_discrepancies(bank_df: pd.DataFrame, budget_df: pd.DataFrame, date_tolerance: int = DATE_TOLERANCE_DAYS) -> pd.DataFrame:
    # Find bank transactions not in budget
    bank_results = []
    for _, bank_row in bank_df.iterrows():
        matches = find_matching_transactions(bank_row, budget_df, date_tolerance)
        bank_results.append(process_transaction_result(bank_row, matches, DISCREPANCY_TYPES))
    
    # Find budget transactions not in bank
    budget_results = []
    for _, budget_row in budget_df.iterrows():
        matches = find_matching_transactions(budget_row, bank_df, date_tolerance)
        if matches.empty:
            budget_results.append({
                'Date': budget_row['Date'],
                'Bank Amount': None,
                'Bank Description': None,
                'Source': 'YNAB',
                'Status': DISCREPANCY_TYPES['right_only'],
                'Budget Amount': budget_row['Amount'],
                'Budget Description': budget_row.get('Description', ''),
                'Budget Payee': budget_row.get('Payee', ''),
                'Amount Difference': None,
                'Days Difference': None,
                'Action_Required': 'Review Missing Bank Transaction'
            })
    
    # Combine results
    return pd.DataFrame(bank_results + budget_results)

def get_date_range_str(df: pd.DataFrame, date_col: str = 'Date') -> str:
    try:
        dates = pd.to_datetime(df[date_col])
        return f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
    except Exception:
        return "Date range unavailable"

def process_ynab_amount(row: pd.Series) -> float:
    try:
        outflow = str(row.get('Outflow', '0')).replace('$', '').replace(',', '').strip()
        inflow = str(row.get('Inflow', '0')).replace('$', '').replace(',', '').strip()
        outflow = float(outflow) if outflow else 0
        inflow = float(inflow) if inflow else 0
        return inflow - outflow
    except (ValueError, TypeError):
        return 0.0

def standardize_columns(df: pd.DataFrame, column_mapping: Dict[str, Any], invert_amounts: bool = False) -> pd.DataFrame:
    df = df.copy()
    
    for target_col, source_cols in column_mapping.items():
        if isinstance(source_cols, list):
            if target_col == 'Amount':
                if 'Inflow' in source_cols and 'Outflow' in source_cols:
                    df[target_col] = df.apply(process_ynab_amount, axis=1)
                else:
                    df[target_col] = df[source_cols].fillna(0).astype(float).sum(axis=1)
            else:
                df[target_col] = df[source_cols].fillna('').apply(lambda x: ' - '.join(filter(None, x)), axis=1)
        else:
            if target_col == 'Amount':
                df[target_col] = df[source_cols].replace(r'[\$,]', '', regex=True).astype(float)
            else:
                df[target_col] = df[source_cols]
    
    if invert_amounts and 'Amount' in df.columns:
        df['Amount'] = -df['Amount']
    
    return df

@st.cache_data(ttl=CACHE_TTL)
def find_matching_transactions(bank_row: pd.Series, budget_df: pd.DataFrame, date_tolerance: int = DATE_TOLERANCE_DAYS) -> pd.DataFrame:
    """Find matching transactions with improved matching logic and caching"""
    try:
        bank_date = pd.to_datetime(bank_row['Date'])
        bank_amount = float(bank_row['Amount'])
        
        # Use vectorized operations for better performance
        date_mask = (
            (budget_df['Date'] >= bank_date - pd.Timedelta(days=date_tolerance)) &
            (budget_df['Date'] <= bank_date + pd.Timedelta(days=date_tolerance))
        )
        amount_mask = np.abs(budget_df['Amount'] - bank_amount) < AMOUNT_TOLERANCE
        
        # First try exact amount matches within date window
        exact_matches = budget_df[date_mask & amount_mask]
        if not exact_matches.empty:
            return exact_matches
        
        # If no exact matches, try fuzzy matching on description
        if 'Description' in bank_row and 'Memo' in budget_df.columns:
            date_window = budget_df[date_mask].copy()
            bank_desc = str(bank_row['Description']).lower()
            
            # Vectorized description matching
            date_window['desc_match'] = date_window['Memo'].fillna('').str.lower().apply(
                lambda x: fuzz.ratio(bank_desc, x)
            )
            
            potential_matches = date_window[date_window['desc_match'] >= DESCRIPTION_MATCH_THRESHOLD]
            if not potential_matches.empty:
                return potential_matches.sort_values('desc_match', ascending=False)
        
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error finding matches: {str(e)}")
        return pd.DataFrame()

def process_transaction_result(
    bank_row: pd.Series,
    matches: pd.DataFrame,
    discrepancy_types: Dict[str, str]
) -> Dict[str, Any]:
    if matches.empty:
        return {
            'Date': bank_row['Date'],
            'Bank Amount': bank_row['Amount'],
            'Bank Description': bank_row.get('Description', ''),
            'Source': bank_row.get('Source', 'Unknown'),
            'Status': discrepancy_types['left_only'],
            'Budget Amount': None,
            'Budget Description': None,
            'Budget Payee': None,
            'Amount Difference': None,
            'Days Difference': None,
            'Action_Required': 'Add to YNAB'
        }
    
    best_match = matches.iloc[0]
    amount_diff = bank_row['Amount'] - best_match['Amount']
    days_diff = (bank_row['Date'] - best_match['Date']).days
    
    return {
        'Date': bank_row['Date'],
        'Bank Amount': bank_row['Amount'],
        'Bank Description': bank_row.get('Description', ''),
        'Source': bank_row.get('Source', 'Unknown'),
        'Status': '✅ Matched' if abs(amount_diff) < 0.01 else discrepancy_types['amount_mismatch'],
        'Budget Amount': best_match['Amount'],
        'Budget Description': best_match.get('Memo', ''),
        'Budget Payee': best_match.get('Payee', ''),
        'Amount Difference': amount_diff,
        'Days Difference': days_diff,
        'Action_Required': 'None' if abs(amount_diff) < 0.01 else 'Review Amount'
    }

# UI Components
def display_dataframe(df: pd.DataFrame, title: Optional[str] = None, height: int = 400) -> None:
    if title:
        st.header(title)
    
    df = df.copy()
    if 'Amount' in df.columns:
        df['Amount'] = df['Amount'].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else '')
    if 'Date' in df.columns:
        df['Date'] = df['Date'].apply(lambda x: pd.to_datetime(x).strftime('%Y-%m-%d') if pd.notnull(x) else '')
    
    st.dataframe(
        data=df,
        use_container_width=True,
        height=height,
        hide_index=True,
        column_config={
            'Status': st.column_config.TextColumn(
                'Status',
                help='Transaction matching status',
                width='medium'
            ),
            'Action_Required': st.column_config.TextColumn(
                'Action Required',
                help='Required action for this transaction',
                width='medium'
            )
        }
    )

def display_metrics(metrics: List[Tuple[str, Any, Optional[str]]], num_columns: int = 3) -> None:
    cols = st.columns(num_columns)
    for i, (label, value, delta) in enumerate(metrics):
        with cols[i % num_columns]:
            formatted_value = f"{value:,}" if isinstance(value, (int, float)) else str(value)
            delta_color = "normal" if delta and delta.endswith("% of total") else "off"
            st.metric(
                label=label,
                value=formatted_value,
                delta=delta,
                delta_color=delta_color
            )

def create_date_range_filter(df: pd.DataFrame, date_col: str = 'Date') -> Tuple[pd.Timestamp, pd.Timestamp]:
    df_dates = pd.to_datetime(df[date_col])
    min_date = df_dates.min().date()
    max_date = df_dates.max().date()
    selected_dates = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    return pd.to_datetime(selected_dates[0]), pd.to_datetime(selected_dates[1])

def create_amount_range_filter(df: pd.DataFrame, bank_col: str = 'Bank Amount', budget_col: str = 'Budget Amount') -> Tuple[float, float]:
    """Create a range filter that works with both bank and budget amounts"""
    amounts = pd.concat([
        df[bank_col].dropna(),
        df[budget_col].dropna()
    ])
    min_amount = float(amounts.min())
    max_amount = float(amounts.max())
    amount_range = st.slider(
        "Amount Range",
        min_value=min_amount,
        max_value=max_amount,
        value=(min_amount, max_amount)
    )
    return amount_range

def display_df_preview(df: pd.DataFrame, title: str, show_columns: bool = True) -> None:
    st.subheader(title)
    if show_columns:
        st.write("Columns:", sorted(df.columns.tolist()))
    st.dataframe(df.head(), use_container_width=True)

def setup_page():
    """Configure the Streamlit page with enhanced settings"""
    st.set_page_config(
        page_title="Budget Match Pro",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def render_sidebar():
    """Render an enhanced sidebar with more information and controls"""
    with st.sidebar:
        # App Logo and Title with better styling
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("https://www.svgrepo.com/show/530438/bank.svg", width=50)
        with col2:
            st.title("Budget Match Pro")
        
        # Progress Section with better visualization
        st.subheader("📊 Progress")
        
        # Create a progress tracker
        steps_completed = 0
        total_steps = len(STEPS)
        
        if 'combined_transactions_df' in st.session_state:
            steps_completed += 1
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.markdown("Bank Transactions")
            with col2:
                st.markdown("✅")
            with col3:
                st.markdown(f"_{len(st.session_state.combined_transactions_df)}_")
        
        if 'budget_df' in st.session_state:
            steps_completed += 1
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.markdown("Budget Entries")
            with col2:
                st.markdown("✅")
            with col3:
                st.markdown(f"_{len(st.session_state.budget_df)}_")
        
        if 'column_mappings' in st.session_state:
            steps_completed += 1
            st.success("Column mapping completed", icon="✅")
        
        if 'processed_data' in st.session_state:
            steps_completed += 1
            st.success("Data processing completed", icon="✅")
        
        # Progress bar
        progress = steps_completed / total_steps
        st.progress(progress, text=f"Progress: {int(progress * 100)}%")
        
        st.divider()
        
        # Settings section
        with st.expander("⚙️ Settings", expanded=False):
            st.slider(
                "Description Match Threshold",
                min_value=50,
                max_value=100,
                value=DESCRIPTION_MATCH_THRESHOLD,
                help="Adjust how strict the description matching should be"
            )
            st.slider(
                "Date Tolerance (days)",
                min_value=0,
                max_value=7,
                value=DATE_TOLERANCE_DAYS,
                help="Number of days to look for matching transactions"
            )
            st.checkbox(
                "Dark Mode",
                value=True,
                help="Toggle dark/light mode"
            )
        
        # Help section
        with st.expander("❓ Help", expanded=False):
            st.markdown("""
                ### Quick Start
                1. Upload your bank transactions
                2. Upload your budget file
                3. Map the columns
                4. Process and analyze
                
                ### Tips
                - Use the filters to find specific transactions
                - Export results for further analysis
                - Use fuzzy matching for better results
            """)
        
        # About section
        with st.expander("ℹ️ About", expanded=False):
            st.markdown("""
                Compare your bank transactions with your budget entries.
                
                ### Features
                - 📤 Multi-file upload support
                - 🗺️ Smart column mapping
                - 🔍 Advanced matching
                - 📊 Detailed analysis
                - 💾 Export capabilities
            """)
            st.divider()
            st.caption("Need help? Contact support.")

def display_summary_metrics(results_df: pd.DataFrame, transactions_df: pd.DataFrame, budget_df: pd.DataFrame):
    """Display enhanced summary metrics with visualizations"""
    total_bank = len(transactions_df)
    total_budget = len(budget_df)
    total_matched = len(results_df[results_df['Status'] == '✅ Matched'])
    total_bank_only = len(results_df[results_df['Status'] == DISCREPANCY_TYPES['left_only']])
    total_budget_only = len(results_df[results_df['Status'] == DISCREPANCY_TYPES['right_only']])
    total_mismatched = len(results_df[results_df['Status'] == DISCREPANCY_TYPES['amount_mismatch']])
    
    # Summary metrics with enhanced styling
    st.markdown("### 📊 Summary")
    
    # Create two rows of metrics for better organization
    row1_cols = st.columns(3)
    with row1_cols[0]:
        st.metric(
            "Total Transactions",
            total_bank,
            f"{total_bank - total_budget:+d} vs budget",
            help="Total number of bank transactions"
        )
    with row1_cols[1]:
        st.metric(
            "Matched Transactions",
            total_matched,
            f"{(total_matched/total_bank*100):.1f}%",
            help="Transactions that match perfectly"
        )
    with row1_cols[2]:
        st.metric(
            "Total Amount",
            f"${results_df['Bank Amount'].sum():,.2f}",
            f"${(results_df['Bank Amount'].sum() - results_df['Budget Amount'].sum()):,.2f}",
            help="Total amount of all transactions"
        )
    
    row2_cols = st.columns(3)
    with row2_cols[0]:
        st.metric(
            "In Bank Only",
            total_bank_only,
            f"{(total_bank_only/total_bank*100):.1f}%",
            help="Transactions found only in bank records"
        )
    with row2_cols[1]:
        st.metric(
            "In Budget Only",
            total_budget_only,
            f"{(total_budget_only/total_budget*100):.1f}%",
            help="Transactions found only in budget"
        )
    with row2_cols[2]:
        st.metric(
            "Amount Mismatches",
            total_mismatched,
            f"{(total_mismatched/total_bank*100):.1f}%",
            help="Transactions with amount discrepancies"
        )
    
    # Add charts
    st.markdown("### 📈 Insights")
    chart_cols = st.columns(2)
    
    with chart_cols[0]:
        # Daily transaction volume chart
        daily_volume = results_df.groupby('Date').size().reset_index()
        daily_volume.columns = ['Date', 'Count']
        
        st.line_chart(
            daily_volume.set_index('Date'),
            use_container_width=True
        )
        st.caption("Daily Transaction Volume")
    
    with chart_cols[1]:
        # Status distribution pie chart
        status_counts = results_df['Status'].value_counts()
        st.bar_chart(status_counts)
        st.caption("Transaction Status Distribution")

def create_enhanced_filters(results_df: pd.DataFrame) -> pd.DataFrame:
    """Create enhanced filtering interface with more options"""
    st.markdown("### 🔍 Filters")
    
    # Quick filters
    quick_filters = st.columns(4)
    with quick_filters[0]:
        if st.button("🏦 Bank Only", use_container_width=True):
            st.session_state.quick_filter = DISCREPANCY_TYPES['left_only']
    with quick_filters[1]:
        if st.button("📊 Budget Only", use_container_width=True):
            st.session_state.quick_filter = DISCREPANCY_TYPES['right_only']
    with quick_filters[2]:
        if st.button("💰 Mismatches", use_container_width=True):
            st.session_state.quick_filter = DISCREPANCY_TYPES['amount_mismatch']
    with quick_filters[3]:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.quick_filter = None
    
    # Main filters
    with st.expander("Advanced Filters", expanded=True):
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            status_filter = st.multiselect(
                "Status",
                options=sorted(results_df['Status'].unique()),
                default=get_default_status_filter(results_df),
                help="Filter by transaction status"
            )
        
        with col2:
            date_range = st.date_input(
                "Date Range",
                value=[results_df['Date'].min(), results_df['Date'].max()],
                help="Filter by date range"
            )
        
        with col3:
            amount_range = st.slider(
                "Amount Range",
                min_value=float(results_df['Bank Amount'].min()),
                max_value=float(results_df['Bank Amount'].max()),
                value=(float(results_df['Bank Amount'].min()), float(results_df['Bank Amount'].max())),
                help="Filter by amount range"
            )
        
        col1, col2 = st.columns(2)
        with col1:
            description_search = st.text_input(
                "Search Description",
                help="Search in transaction descriptions"
            )
            source_filter = st.multiselect(
                "Source",
                options=sorted(results_df['Source'].unique()),
                default=sorted(results_df['Source'].unique()),
                help="Filter by transaction source"
            )
        
        with col2:
            exclude_matched = st.checkbox(
                "Exclude Matched",
                help="Hide perfectly matched transactions"
            )
            show_large_discrepancies = st.checkbox(
                "Show Large Discrepancies Only",
                help="Show only transactions with significant amount differences"
            )
    
    # Apply filters
    filtered_df = apply_enhanced_filters(
        results_df,
        status_filter,
        date_range,
        amount_range,
        description_search,
        source_filter,
        exclude_matched,
        show_large_discrepancies
    )
    
    return filtered_df

def apply_enhanced_filters(
    df: pd.DataFrame,
    status_filter: List[str],
    date_range: Tuple[datetime, datetime],
    amount_range: Tuple[float, float],
    description_search: str,
    source_filter: List[str],
    exclude_matched: bool,
    show_large_discrepancies: bool
) -> pd.DataFrame:
    """Apply all filters to the DataFrame"""
    mask = (
        df['Status'].isin(status_filter) &
        (df['Date'].dt.date >= date_range[0]) &
        (df['Date'].dt.date <= date_range[1]) &
        (df['Source'].isin(source_filter)) &
        (
            (df['Bank Amount'].between(amount_range[0], amount_range[1])) |
            (df['Budget Amount'].between(amount_range[0], amount_range[1]))
        )
    )
    
    if description_search:
        description_mask = (
            df['Bank Description'].str.contains(description_search, case=False, na=False) |
            df['Budget Description'].str.contains(description_search, case=False, na=False) |
            df['Budget Payee'].str.contains(description_search, case=False, na=False)
        )
        mask &= description_mask
    
    if exclude_matched:
        mask &= (df['Status'] != '✅ Matched')
    
    if show_large_discrepancies:
        mask &= (abs(df['Amount Difference']) > AMOUNT_TOLERANCE * 10)
    
    return df[mask]

def get_default_status_filter(results_df: pd.DataFrame) -> List[str]:
    """Get default status filter based on current state"""
    if 'quick_filter' in st.session_state and st.session_state.quick_filter:
        if st.session_state.quick_filter in results_df['Status'].unique():
            return [st.session_state.quick_filter]
        st.session_state.quick_filter = None
    
    return [DISCREPANCY_TYPES['left_only']] if DISCREPANCY_TYPES['left_only'] in results_df['Status'].unique() else [results_df['Status'].unique()[0]]

def display_enhanced_results(filtered_df: pd.DataFrame):
    """Display results with enhanced formatting and features"""
    st.markdown("### 📋 Results")
    
    # Action buttons
    action_col1, action_col2, action_col3 = st.columns(3)
    with action_col1:
        if st.button("📋 Copy as YNAB Format", type="primary", use_container_width=True):
            ynab_format = filtered_df.apply(
                lambda x: f"{x['Date'].strftime('%Y-%m-%d')},{x['Amount']:.2f},{x['Description']}",
                axis=1
            ).str.cat(sep='\n')
            st.code(ynab_format, language=None)
            st.info("👆 Copy the above text and paste into YNAB's import dialog")
    
    with action_col2:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            "💾 Export to CSV",
            csv,
            "transactions.csv",
            "text/csv",
            use_container_width=True
        )
    
    with action_col3:
        st.button("🔄 Refresh Results", use_container_width=True)
    
    # Results table
    st.subheader(f"Showing {len(filtered_df)} transactions")
    
    # Configure columns for better display
    display_columns = {
        'Date': st.column_config.DateColumn(
            'Date',
            format='YYYY-MM-DD',
            width='medium'
        ),
        'Bank Amount': st.column_config.NumberColumn(
            'Bank Amount',
            format='$%.2f',
            width='medium'
        ),
        'Bank Description': st.column_config.TextColumn(
            'Bank Description',
            width='large'
        ),
        'Budget Amount': st.column_config.NumberColumn(
            'Budget Amount',
            format='$%.2f',
            width='medium'
        ),
        'Budget Description': st.column_config.TextColumn(
            'Budget Description',
            width='medium'
        ),
        'Budget Payee': st.column_config.TextColumn(
            'Budget Payee',
            width='medium'
        ),
        'Amount Difference': st.column_config.NumberColumn(
            'Difference',
            format='$%.2f',
            help='Positive means bank amount is larger',
            width='small'
        ),
        'Days Difference': st.column_config.NumberColumn(
            'Days',
            help='Positive means bank date is later',
            width='small'
        ),
        'Status': st.column_config.TextColumn(
            'Status',
            width='medium'
        ),
        'Source': st.column_config.TextColumn(
            'Source',
            width='small'
        ),
        'Action_Required': st.column_config.TextColumn(
            'Action Required',
            width='medium'
        )
    }
    
    # Sort the DataFrame to group similar transactions together
    filtered_df = filtered_df.sort_values(['Date', 'Status', 'Bank Amount', 'Budget Amount'])
    
    st.dataframe(
        filtered_df,
        use_container_width=True,
        column_config=display_columns,
        hide_index=True
    )

def process_transaction_files(uploaded_files):
    """Process uploaded transaction files"""
    transaction_dfs = []
    all_columns = set()
    
    with st.spinner("Processing transaction files..."):
        for file in uploaded_files:
            try:
                df = load_and_process_csv(file, file.name)
                transaction_dfs.append(df)
                all_columns.update(df.columns)
            except Exception as e:
                st.error(f"Error reading file {file.name}: {str(e)}")
                continue
    
    if transaction_dfs:
        try:
            combined_transactions_df = combine_transaction_dfs(transaction_dfs)
            st.session_state.combined_transactions_df = combined_transactions_df
            st.session_state.all_columns = sorted(list(all_columns))
            
            st.success(f"Successfully loaded {len(transaction_dfs)} files with {len(combined_transactions_df)} total transactions")
            
            # Display metrics
            metrics = [
                ("Total Files", len(transaction_dfs), None),
                ("Total Transactions", len(combined_transactions_df), None),
                ("Date Range", get_date_range_str(combined_transactions_df), None)
            ]
            display_metrics(metrics)
            
            # Display preview
            with st.expander("Preview Combined Transactions", expanded=True):
                display_dataframe(combined_transactions_df, height=300)
            
            # Add download button
            st.download_button(
                "📥 Download Combined Transactions CSV",
                combined_transactions_df.to_csv(index=False),
                "combined_transactions.csv",
                "text/csv",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Error combining transaction files: {str(e)}")

def process_budget_file(uploaded_file):
    """Process uploaded budget file"""
    try:
        budget_df = load_and_process_csv(uploaded_file, uploaded_file.name)
        st.session_state.budget_df = budget_df
        st.success(f"Successfully loaded budget file with {len(budget_df)} entries")
        
        # Display preview
        with st.expander("Preview Budget Data", expanded=True):
            display_dataframe(budget_df, height=300)
    except Exception as e:
        st.error(f"Error processing budget file: {str(e)}")

def handle_column_mapping():
    """Handle the column mapping step"""
    if 'combined_transactions_df' not in st.session_state or 'budget_df' not in st.session_state:
        st.warning("Please upload both transaction files and budget file in the Upload Files tab.")
        return
    
    st.session_state.column_mapper.display_available_columns(
        st.session_state.combined_transactions_df,
        st.session_state.budget_df
    )
    
    transaction_mapping, budget_mapping, has_ynab_amount = st.session_state.column_mapper.create_mapping_interface(
        st.session_state.combined_transactions_df,
        st.session_state.budget_df
    )
    
    if transaction_mapping is not None:
        st.session_state.column_mappings = {
            'transaction_mapping': transaction_mapping,
            'budget_mapping': budget_mapping,
            'has_ynab_amount': has_ynab_amount
        }
        st.success("✅ Column mappings updated! Proceed to Process Data tab.")

def handle_data_processing():
    """Handle the data processing step"""
    if 'column_mappings' not in st.session_state:
        st.warning("Please complete column mapping first.")
        return
    
    if 'combined_transactions_df' not in st.session_state or 'budget_df' not in st.session_state:
        st.warning("Please upload all required files first.")
        return
    
    try:
        with st.spinner("Processing data..."):
            transaction_mapping = st.session_state.column_mappings['transaction_mapping']
            budget_mapping = st.session_state.column_mappings['budget_mapping']
            has_ynab_amount = st.session_state.column_mappings['has_ynab_amount']
            
            # Add invert amount controls
            col1, col2 = st.columns(2)
            with col1:
                invert_transaction_amounts = st.checkbox(
                    "Invert Transaction Amounts",
                    help="Check this if charges are showing as positive instead of negative"
                )
            with col2:
                invert_budget_amounts = st.checkbox(
                    "Invert Budget Amounts",
                    help="Check this if budget amounts have incorrect signs"
                )
            
            standardized_transactions_df = standardize_columns(
                st.session_state.combined_transactions_df.copy(),
                transaction_mapping,
                invert_amounts=invert_transaction_amounts
            )
            
            standardized_budget_df = st.session_state.budget_df.copy()
            if has_ynab_amount:
                standardized_budget_df['Amount'] = standardized_budget_df.apply(process_ynab_amount, axis=1)
                budget_mapping['Amount'] = 'Amount'
            
            standardized_budget_df = standardize_columns(
                standardized_budget_df,
                budget_mapping,
                invert_amounts=invert_budget_amounts
            )
            
            # Validate required columns
            for df_name, df in [("Bank Transactions", standardized_transactions_df), ("Budget", standardized_budget_df)]:
                missing = [col for col in ['Date', 'Amount'] if col not in df.columns]
                if missing:
                    raise ValueError(f"Missing required columns in {df_name}: {', '.join(missing)}")
            
            st.session_state.processed_data = {
                'transactions': standardized_transactions_df,
                'budget': standardized_budget_df
            }
            
            # Display previews
            col1, col2 = st.columns(2)
            with col1:
                with st.expander("Preview Processed Bank Transactions", expanded=True):
                    display_df_preview(standardized_transactions_df, "Processed Bank Transactions")
            with col2:
                with st.expander("Preview Processed Budget Data", expanded=True):
                    display_df_preview(standardized_budget_df, "Processed Budget Data")
            
            st.success("Data processing complete! View results in the Results tab.")
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        logger.error(f"Data processing error: {str(e)}", exc_info=True)

def handle_results():
    """Handle the results analysis step"""
    if 'processed_data' not in st.session_state:
        st.info("Complete the data processing in previous tabs to view results.")
        return
    
    transactions_df = process_dataframe(st.session_state.processed_data['transactions'].copy())
    budget_df = process_dataframe(st.session_state.processed_data['budget'].copy())
    
    # Process transactions in both directions
    results_df = find_all_discrepancies(transactions_df, budget_df, DATE_TOLERANCE_DAYS)
    
    # Display summary metrics
    display_summary_metrics(results_df, transactions_df, budget_df)
    
    # Create and apply filters
    filtered_df = create_enhanced_filters(results_df)
    
    # Display results
    display_enhanced_results(filtered_df)

# Update the main application to use the new components
def main():
    setup_page()
    
    if 'column_mapper' not in st.session_state:
        st.session_state.column_mapper = ColumnMapper()
    
    render_sidebar()
    
    with st.container():
        st.title("Budget Match Pro")
        
        # Create tabs for different sections
        tabs = st.tabs([step_info["tab_label"] for step_info in STEPS.values()])
        
        with tabs[0]:
            # Upload Files tab
            st.header(f"Step 1: {STEPS[1]['name']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Bank Transactions")
                uploaded_transaction_files = st.file_uploader(
                    "Upload Transactions CSV Files",
                    type="csv",
                    accept_multiple_files=True,
                    help="Select one or more transaction CSV files to process"
                )
            
            with col2:
                st.subheader("Budget Data")
                uploaded_budget_file = st.file_uploader(
                    "Upload Budget CSV File",
                    type="csv",
                    help="Select your budget CSV file to compare against transactions",
                    key="budget_uploader"
                )
            
            if uploaded_transaction_files:
                process_transaction_files(uploaded_transaction_files)
            
            if uploaded_budget_file:
                process_budget_file(uploaded_budget_file)
        
        with tabs[1]:
            # Column Mapping tab
            st.header(f"Step 2: {STEPS[2]['name']}")
            handle_column_mapping()
        
        with tabs[2]:
            # Process Data tab
            st.header(f"Step 3: {STEPS[3]['name']}")
            handle_data_processing()
        
        with tabs[3]:
            # Results tab
            st.header(f"Step 4: {STEPS[4]['name']}")
            handle_results()

if __name__ == "__main__":
    main()
