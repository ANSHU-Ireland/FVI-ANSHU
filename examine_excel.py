#!/usr/bin/env python3
"""
Script to examine the Excel file structure and prepare data for FVI platform.
"""
import pandas as pd
import sys
import os
from pathlib import Path

def examine_excel_file(file_path):
    """Examine the structure of the Excel file."""
    print(f"Examining Excel file: {file_path}")
    
    try:
        # Read the Excel file
        xl = pd.ExcelFile(file_path)
        print(f"\nSheet names: {xl.sheet_names}")
        
        # Examine each sheet
        for sheet_name in xl.sheet_names:
            print(f"\n=== Sheet: {sheet_name} ===")
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Show first few rows
            print("\nFirst 5 rows:")
            print(df.head())
            
            # Show data types
            print("\nData types:")
            print(df.dtypes)
            
            # Check for missing values
            print("\nMissing values:")
            print(df.isnull().sum())
            
            print("\n" + "="*50)
            
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

def main():
    excel_file = "/workspaces/FVI-ANSHU/FVI Scoring Metrics_Coal.xlsx"
    
    if not os.path.exists(excel_file):
        print(f"Excel file not found: {excel_file}")
        return
    
    examine_excel_file(excel_file)

if __name__ == "__main__":
    main()
