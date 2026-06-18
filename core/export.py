from __future__ import annotations

import csv
from pathlib import Path

try:
    import pandas as pd
except Exception:
    pd = None


def export_results(rois: list[dict], output_path: str) -> None:
    """Export ROI metrics to CSV or Excel (.xlsx)."""
    columns = [
        "ROI ID",
        "Тип",
        "Площадь (мм²)",
        "Количество ядер",
        "Плотность (ядра/мм²)",
    ]

    normalized_rows = []
    for row in rois:
        normalized_rows.append({col: row.get(col) for col in columns})

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    ext = output.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        if pd is None:
            raise RuntimeError("Для экспорта в Excel требуется pandas")
        df = pd.DataFrame(normalized_rows, columns=columns)
        df.to_excel(output, index=False)
        return

    if ext not in {".csv", ""}:
        raise ValueError("Поддерживается экспорт только в CSV или Excel (.xlsx)")

    csv_path = output if ext == ".csv" else output.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in normalized_rows:
            writer.writerow(row)


def export_batch_results(rows: list[dict], output_path: str) -> None:
    columns = [
        "Файл",
        "ROI ID",
        "Название ROI",
        "Тип",
        "Площадь (мм²)",
        "Количество ядер",
        "Плотность (ядра/мм²)",
    ]

    normalized_rows = []
    for row in rows:
        normalized_rows.append({col: row.get(col) for col in columns})

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    ext = output.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        if pd is None:
            raise RuntimeError("Для экспорта в Excel требуется pandas")
        df = pd.DataFrame(normalized_rows, columns=columns)
        df.to_excel(output, index=False)
        return

    if ext not in {".csv", ""}:
        raise ValueError("Поддерживается экспорт только в CSV или Excel (.xlsx)")

    csv_path = output if ext == ".csv" else output.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in normalized_rows:
            writer.writerow(row)
