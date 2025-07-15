import pandas as pd
import numpy as np
import sys

def add_moving_averages(input_file, output_file=None):
    """
    Add moving averages (5, 10, 20, 40, 55) to stock data CSV file.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file (optional)
    """
    
    # Read the CSV file
    try:
        df = pd.read_csv(input_file)
        print(f"Successfully loaded {len(df)} rows from {input_file}")
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Check if required columns exist
    required_columns = ['close']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return
    
    # Sort by timestamp to ensure proper chronological order
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)
        print("Data sorted by timestamp")
    
    # Define moving average periods
    ma_periods = [5, 10, 20, 40, 55]
    
    # Calculate moving averages with your naming convention
    print("Calculating moving averages...")
    ma_columns = []
    for i, period in enumerate(ma_periods):
        if i == 0:
            column_name = 'MA'  # First MA column
        else:
            column_name = f'MA.{i}'  # MA.1, MA.2, MA.3, MA.4
        
        df[column_name] = df['close'].rolling(window=period, min_periods=1).mean()
        ma_columns.append(column_name)
        print(f"Added {column_name} (MA_{period})")
    
    # Round moving averages to reasonable decimal places
    df[ma_columns] = df[ma_columns].round(4)
    
    # Determine output filename
    if output_file is None:
        if input_file.endswith('.csv'):
            output_file = input_file.replace('.csv', '_with_ma.csv')
        else:
            output_file = input_file + '_with_ma.csv'
    
    # Save the updated dataframe
    try:
        df.to_csv(output_file, index=False)
        print(f"Successfully saved updated data to {output_file}")
        print(f"Added columns: {', '.join(ma_columns)}")
    except Exception as e:
        print(f"Error saving file: {e}")
        return
    
    # Display summary statistics
    print("\n--- Summary Statistics ---")
    print(f"Total rows: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}" if 'timestamp' in df.columns else "No timestamp column")
    print(f"Close price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Show first few rows with moving averages
    print("\n--- First 10 rows with Moving Averages ---")
    display_columns = ['timestamp', 'close'] + ma_columns
    available_columns = [col for col in display_columns if col in df.columns]
    print(df[available_columns].head(10).to_string(index=False))
    
    return df

def main():
    """
    Main function to run the script from command line
    """
    # Your file paths
    input_file = "C:/Users/Hayden/Downloads/nvda_5min.csv"
    output_file = "C:/Users/Hayden/Downloads/nvda_5min_with_ma.csv"  # Fixed typo
    
    print(f"Processing: {input_file}")
    print(f"Output will be saved to: {output_file}")
    print()
    
    # Process the file
    result_df = add_moving_averages(input_file, output_file)
    
    if result_df is not None:
        print("\n✓ Moving averages successfully added to your NVDA stock data!")
        print("The new columns are: MA, MA.1, MA.2, MA.3, MA.4")
        print("These correspond to MA_5, MA_10, MA_20, MA_40, MA_55 periods respectively")

if __name__ == "__main__":
    main()
