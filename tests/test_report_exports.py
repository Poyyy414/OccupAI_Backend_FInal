from io import BytesIO
from zipfile import ZipFile

from backend import main as m


SAMPLE_REPORT = {
    "generated_at_ph": "2026-11-16 10:00:00 PHT",
    "daily_revenue": [
        {
            "date": "2026-11-10",
            "label": "Tuesday",
            "revenue_php": 150.0,
            "discount_php": 20.0,
            "transaction_count": 3,
        },
        {
            "date": "2026-11-11",
            "label": "Wednesday",
            "revenue_php": 100.0,
            "discount_php": 0.0,
            "transaction_count": 2,
        },
    ],
    "monthly_revenue": [
        {
            "month": "2026-11",
            "label": "November",
            "revenue_php": 250.0,
            "discount_php": 20.0,
            "transaction_count": 5,
        }
    ],
}


def test_weekly_report_rows_match_dashboard_grouping():
    rows, columns = m._revenue_export_rows(SAMPLE_REPORT, "weekly")

    assert rows == [{
        "week_start": "2026-11-09",
        "revenue_php": 250.0,
        "discount_php": 20.0,
        "transaction_count": 5,
        "net_revenue_php": 230.0,
    }]
    assert columns[-1] == ("transaction_count", "Payments")


def test_xlsx_export_is_a_valid_office_archive():
    output = m._build_revenue_xlsx(SAMPLE_REPORT, "daily")
    with ZipFile(BytesIO(output.getvalue())) as archive:
        names = set(archive.namelist())
    assert "xl/workbook.xml" in names
    assert "xl/worksheets/sheet1.xml" in names


def test_docx_export_is_a_valid_office_archive():
    output = m._build_revenue_docx(SAMPLE_REPORT, "monthly")
    with ZipFile(BytesIO(output.getvalue())) as archive:
        names = set(archive.namelist())
    assert "word/document.xml" in names
    assert "[Content_Types].xml" in names
