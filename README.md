# Sysdig Report Studio

A Streamlit application for building reports (primarily aimed at executive high level reports) from Sysdig vulnerability data (SysQL). Design reports with various chart types, preview them live, then generate PDFs on demand or on a schedule.

## Features

- **Visual Report Builder**: Design reports using a drag-and-drop style interface
- **Multiple Chart Types**: Vulnerability history trends, traffic lights, bar charts, pie charts, and tables
- **Live Preview**: See your report as you build it
- **PDF Generation**: Export professional PDF reports
- **Scheduling**: Optionally schedule automated report generation (daily, weekly, monthly)
- **SysQL Support**: Query Sysdig data using SysQL syntax

## Installation

### Prerequisites

- Python 3.10+
- A Sysdig account with API access

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sysdig-report-studio.git
   cd sysdig-report-studio
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

The app will open in your browser at `http://localhost:8501`.

## Configuration

### API Token

You'll need a Sysdig API token with read access to vulnerability data. Enter this in the sidebar when running the app.

For local development, you can set defaults in `app.py`:

```python
# Default values - change these for your environment
DEFAULT_CUSTOMER_NAME = "Acme Corp"
DEFAULT_API_TOKEN = ""  # Leave empty for security, or set for local dev
```

**Security Note**: Never commit API tokens to version control. If deploying to Kubernetes, consider refactoring to read the token from a Kubernetes Secret or environment variable instead of hardcoding it.

### Regions

The app supports multiple Sysdig regions. Add or modify regions in the `SYSDIG_REGIONS` dictionary in `config.py`:

```python
SYSDIG_REGIONS = {
    "APJ": "app.au1.sysdig.com",
    "US East": "secure.sysdig.com",
    "EU": "eu1.app.sysdig.com",
    "EU North": "app.eu2.sysdig.com",
    "US West": "us2.app.sysdig.com",
    "India": "app.in1.sysdig.com",
    "US West (GCP)": "app.us4.sysdig.com",
    "ME Central": "app.me2.sysdig.com"
}
```

### Data Storage

By default, the SQLite database and generated PDFs are stored in `./data/`. Override this with the `SYSDIG_REPORT_DATA_DIR` environment variable:

```bash
export SYSDIG_REPORT_DATA_DIR=/path/to/persistent/storage
```

This is useful for Kubernetes deployments with persistent volumes.

## Usage

1. **Configure**: Enter your customer name, select your Sysdig region, and provide your API token in the sidebar
2. **Design Elements**: Choose a chart type, write a SysQL query, and click "Fetch & Preview"
3. **Build Report**: Add elements to your report template, arrange and resize as needed
4. **Save**: Give your report a name and save it
5. **Generate**: Click "Generate" to create a PDF, or configure a schedule for automated generation

## Project Structure

```
sysdig-report-studio/
├── app.py              # Main Streamlit application
├── charts.py           # Chart rendering (Plotly)
├── config.py           # Shared configuration (regions, etc.)
├── database.py         # SQLite data layer
├── pdf_generator.py    # PDF generation (ReportLab)
├── scheduler.py        # Background scheduler for automated reports
├── requirements.txt    # Python dependencies
└── data/               # SQLite DB and generated PDFs (gitignored)
```

## Future Improvements

- Read API token from environment variable or Kubernetes Secret
- Email delivery of scheduled reports
- Additional chart types
- Report templates/presets
- Create k8s manifests

## Author

Aaron Miles (aaron.miles@sysdig.com)